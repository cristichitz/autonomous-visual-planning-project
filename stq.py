import numpy as np
import torch
import collections
from typing import Any, Dict, MutableMapping, Optional, Sequence, Text, Union


class STQuality(object):
    """Metric class for the Segmentation and Tracking Quality (STQ).

    The metric computes the geometric mean of two terms.
    - Association Quality: This term measures the quality of the track ID
        assignment for `thing` classes. It is formulated as a weighted IoU
        measure.
    - Segmentation Quality: This term measures the semantic segmentation quality.
        The standard class IoU measure is used for this.

    Example usage:

    stq_obj = segmentation_tracking_quality.STQuality(num_classes, things_list,
    ignore_label, max_instances_per_category, offset)
    stq_obj.update_state(y_true_1, y_pred_1)
    stq_obj.update_state(y_true_2, y_pred_2)
    ...
    result = stq_obj.result().numpy()
    """
    def __init__(self,
                 num_classes: int,
                 things_list: Sequence[int],
                 ignore_label: int,
                 max_instances_per_category: int,
                 offset: int,
                 name='stq'
                 ):
        self._name = name
        self._num_classes = num_classes
        self._ignore_label = ignore_label
        self._things_list = things_list
        self._max_instances_per_category = max_instances_per_category
        
        if ignore_label >= num_classes:
            self._confusion_matrix_size = num_classes + 1
            self._include_indices = np.arange(self._num_classes)
        else:
            self._confusion_matrix_size = num_classes + 1
            self._include_indices = np.array(
                [i for i in range(num_classes) if i != self._ignore_label])
        

        self._iou_confusion_matrix_per_sequence = collections.OrderedDict()
        self._predictions = collections.OrderedDict()
        self._ground_truth = collections.OrderedDict()
        self._intersections = collections.OrderedDict()
        self._sequence_length = collections.OrderedDict()
        self._offset = offset
        lower_bound = num_classes * max_instances_per_category
        if offset < lower_bound:
            raise ValueError('Please choose an offset that is higher than num_classes * max_instances')
    
    def _update_dict_stats(self,
                           stat_dict:MutableMapping[int, torch.Tensor],
                           id_array: torch.Tensor):
        """Updates a given dict with corresponding counts"""
        if id_array.numel() == 0:
            return
        unique_ids, counts = torch.unique(id_array, return_counts=True)
        unique_ids = unique_ids.cpu().numpy()
        counts = counts.cpu().numpy()
        
        for uid, count in zip(unique_ids, counts):
            stat_dict[uid] = stat_dict.get(uid, 0) + count

    @torch.no_grad()
    def update_state(self,
                     y_true: torch.Tensor, 
                     y_pred: torch.Tensor, 
                     sequence_id: str = "0"):
        """Accumulates the segmentation and tracking quality statistics.

        Args:
        y_true: The ground-truth panoptic label map for a particular video frame
            (defined as semantic_map * max_instances_per_category + instance_map).
        y_pred: The predicted panoptic label map for a particular video frame
            (defined as semantic_map * max_instances_per_category + instance_map).
        sequence_id: The optional ID of the sequence the frames belong to. When no
            sequence is given, all frames are considered to belong to the same
            sequence (default: 0).
        weights: The weights for each pixel with the same shape of `y_true`.
        """
        y_true = y_true.long()
        y_pred = y_pred.long()

        semantic_label = y_true // self._max_instances_per_category
        semantic_prediction = y_pred // self._max_instances_per_category

        if self._ignore_label > self._num_classes:
            semantic_label = torch.where(
                semantic_label != self._ignore_label,
                semantic_label, torch.tensor(self._num_classes, device=y_true.device)
            )
            semantic_prediction = torch.where(
                semantic_prediction != self._ignore_label, 
                semantic_prediction, 
                torch.tensor(self._num_classes, device=y_pred.device)
            )
        
        sem_label_flat = semantic_label.flatten()
        sem_pred_flat = semantic_prediction.flatten()

        confusion_matrix_indices = sem_label_flat * self._confusion_matrix_size + sem_pred_flat
        conf_matrix = torch.bincount(confusion_matrix_indices, minlength=self._confusion_matrix_size**2)
        conf_matrix = conf_matrix.view(self._confusion_matrix_size, self._confusion_matrix_size).cpu()

        if sequence_id not in self._iou_confusion_matrix_per_sequence:
            self._iou_confusion_matrix_per_sequence[sequence_id] = conf_matrix
            self._predictions[sequence_id] = {}
            self._ground_truth[sequence_id] = {}
            self._intersections[sequence_id] = {}
            self._sequence_length[sequence_id] = 1
        else:
            self._iou_confusion_matrix_per_sequence[sequence_id] += conf_matrix

        instance_label = y_true % self._max_instances_per_category
        
        label_mask = torch.zeros_like(semantic_label, dtype=torch.bool)
        prediction_mask = torch.zeros_like(semantic_prediction, dtype=torch.bool)
        for things_class_id in self._things_list:
            label_mask |= (semantic_label == things_class_id)
            prediction_mask |= (semantic_prediction == things_class_id)

        is_crowd = (instance_label == 0) & label_mask
        label_mask = (label_mask & (~is_crowd))
        prediction_mask = (prediction_mask & (~is_crowd))

        seq_preds = self._predictions[sequence_id]
        seq_gts = self._ground_truth[sequence_id]
        seq_intersects = self._intersections[sequence_id]

        self._update_dict_stats(seq_preds, y_pred[prediction_mask])
        self._update_dict_stats(seq_gts, y_true[label_mask])
        

        non_crowd_intersection = label_mask & prediction_mask
        intersetion_ids = (y_true[non_crowd_intersection] * self._offset + y_pred[non_crowd_intersection])

        self._update_dict_stats(seq_intersects, intersetion_ids)

    def result(self) -> Dict[Text, Any]:
        """Computes the segmentation and tracking quality.

        Returns:
        A dictionary containing:
            - 'STQ': The total STQ score.
            - 'AQ': The total association quality (AQ) score.
            - 'IoU': The total mean IoU.
            - 'STQ_per_seq': A list of the STQ score per sequence.
            - 'AQ_per_seq': A list of the AQ score per sequence.
            - 'IoU_per_seq': A list of mean IoU per sequence.
            - 'Id_per_seq': A list of sequence Ids to map list index to sequence.
            - 'Length_per_seq': A list of the length of each sequence.
        """
        # Compute association quality(AQ)
        num_tubes_per_seq = [0] * len(self._ground_truth)
        aq_per_seq = [0] * len(self._ground_truth)
        iou_per_seq = [0] * len(self._ground_truth)
        id_per_seq = [''] * len(self._ground_truth)

        for index, sequence_id in enumerate(self._ground_truth):
            outer_sum = 0.0
            predictions = self._predictions[sequence_id]
            ground_truth = self._ground_truth[sequence_id]
            intersections = self._intersections[sequence_id]
            num_tubes_per_seq[index] = len(ground_truth)
            id_per_seq[index] = sequence_id

            for gt_id, gt_size in ground_truth.items():
                inner_sum = 0.0
                for pr_id, pr_size in predictions.items():
                    tpa_key = self._offset * gt_id + pr_id
                    if tpa_key in intersections:
                        tpa = intersections[tpa_key]
                        fpa = pr_size - tpa
                        fna = gt_size - tpa
                        inner_sum += tpa * (tpa / (tpa + fpa + fna))
                
                outer_sum += 1.0 / gt_size * inner_sum
            aq_per_seq[index] = outer_sum
        
        aq_mean = np.sum(aq_per_seq) / np.maximum(np.sum(num_tubes_per_seq), 1e-15)
        aq_per_seq = aq_per_seq / np.maximum(num_tubes_per_seq, 1e-15)

        # Compute IoU scores.
        # The rows correspond to ground-truth and the columns to predictions.
        total_confusion = np.zeros(
            (self._confusion_matrix_size, self._confusion_matrix_size),
            dtype=np.float64)
        for index, confusion in enumerate(self._iou_confusion_matrix_per_sequence.values()):
            confusion = confusion.numpy()
            removal_matrix = np.zeros_like(confusion)
            removal_matrix[self._include_indices, :] = 1.0
            total_confusion += confusion

            # `intersections` corresponds to true positives.
            intersections = confusion.diagonal()
            fps = confusion.sum(axis=0) - intersections
            fns = confusion.sum(axis=1) - intersections
            unions = intersections + fps + fns

            num_classes = np.count_nonzero(unions)
            ious = (intersections.astype(np.double) /
                    np.maximum(unions, 1e-15).astype(np.double))
            iou_per_seq[index] = np.sum(ious) / num_classes
        
        # 'intersection' corresponds to true positives
        intersections = total_confusion.diagonal()
        fps = total_confusion.sum(axis=0) - intersections
        fns = total_confusion.sum(axis=1) - intersections
        unions = intersections + fps + fns

        num_classes = np.count_nonzero(unions)
        ious = (intersections.astype(np.double) /
              np.maximum(unions, 1e-15).astype(np.double))
        iou_mean = np.sum(ious) / num_classes

        st_quality = np.sqrt(aq_mean * iou_mean)
        st_quality_per_seq = np.sqrt(aq_per_seq * iou_per_seq)
        return {'STQ': st_quality,
                'AQ': aq_mean,
                'IoU': float(iou_mean),
                'STQ_per_seq': st_quality_per_seq,
                'AQ_per_seq': aq_per_seq,
                'IoU_per_seq': iou_per_seq,
                'ID_per_seq': id_per_seq,
                'Length_per_seq': list(self._sequence_length.values()),
                }


    def reset_states(self):
        """Resets all states that accumulated data."""

        self._iou_confusion_matrix_per_sequence = collections.OrderedDict()
        self._predictions = collections.OrderedDict()
        self._ground_truth = collections.OrderedDict()
        self._intersections = collections.OrderedDict()
        self._sequence_length = collections.OrderedDict()