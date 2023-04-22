import numpy as np
from abc import ABCMeta, abstractmethod
import torch
from torch.nn import functional as F
from mmdet3d.core.bbox.structures import limit_period

class BaseBBoxCoder(metaclass=ABCMeta):
    """Base bounding box coder."""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def encode(self, bboxes, gt_bboxes):
        """Encode deltas between bboxes and ground truth boxes."""


    @abstractmethod
    def decode(self, bboxes, bboxes_pred):
        """Decode the predicted bboxes according to prediction and base
        boxes."""


class FCOS3DBBoxCoder(BaseBBoxCoder):
    """Bounding box coder for FCOS3D.
    Args:
        base_depths (tuple[tuple[float]]): Depth references for decode box
            depth. Defaults to None.
        base_dims (tuple[tuple[float]]): Dimension references for decode box
            dimension. Defaults to None.
        code_size (int): The dimension of boxes to be encoded. Defaults to 7.
        norm_on_bbox (bool): Whether to apply normalization on the bounding
            box 2D attributes. Defaults to True.
    """

    def __init__(self,
                 base_depths=None,
                 base_dims=None,
                 code_size=7,
                 norm_on_bbox=True):
        super(FCOS3DBBoxCoder, self).__init__()
        self.base_depths = base_depths
        self.base_dims = base_dims
        self.bbox_code_size = code_size
        self.norm_on_bbox = norm_on_bbox

    def encode(self, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels):
        pass

    def decode(self, bbox, scale, stride, training, cls_score=None):
        """Decode regressed results into 3D predictions.
        Note that offsets are not transformed to the projected 3D centers.
        Args:
            bbox (torch.Tensor): Raw bounding box predictions in shape
                [N, C, H, W].
            scale (tuple[`Scale`]): Learnable scale parameters.
            stride (int): Stride for a specific feature level.
            training (bool): Whether the decoding is in the training
                procedure.
            cls_score (torch.Tensor): Classification score map for deciding
                which base depth or dim is used. Defaults to None.
        Returns:
            torch.Tensor: Decoded boxes.
        """
        scale_offset, scale_depth, scale_size = scale[0:3]

        clone_bbox = bbox.clone()
        bbox[:, :2] = scale_offset(clone_bbox[:, :2]).float()
        bbox[:, 2] = scale_depth(clone_bbox[:, 2]).float()
        bbox[:, 3:6] = scale_size(clone_bbox[:, 3:6]).float()

        if self.base_depths is None:
            bbox[:, 2] = bbox[:, 2].exp()
        elif len(self.base_depths) == 1:  
            mean = self.base_depths[0][0]
            std = self.base_depths[0][1]
            bbox[:, 2] = mean + bbox.clone()[:, 2] * std
        else: 
            assert len(self.base_depths) == cls_score.shape[1], \
                'The number of multi-class depth priors should be equal to ' \
                'the number of categories.'
            indices = cls_score.max(dim=1)[1]
            depth_priors = cls_score.new_tensor(
                self.base_depths)[indices, :].permute(0, 3, 1, 2)
            mean = depth_priors[:, 0]
            std = depth_priors[:, 1]
            bbox[:, 2] = mean + bbox.clone()[:, 2] * std

        bbox[:, 3:6] = bbox[:, 3:6].exp()
        if self.base_dims is not None:
            assert len(self.base_dims) == cls_score.shape[1], \
                'The number of anchor sizes should be equal to the number ' \
                'of categories.'
            indices = cls_score.max(dim=1)[1]
            size_priors = cls_score.new_tensor(
                self.base_dims)[indices, :].permute(0, 3, 1, 2) 
            bbox[:, 3:6] = size_priors * bbox.clone()[:, 3:6]

        assert self.norm_on_bbox is True, 'Setting norm_on_bbox to False '\
            'has not been thoroughly tested for FCOS3D.'
        
        return bbox


    @staticmethod
    def decode_yaw(bbox, centers2d, dir_cls, dir_offset, cam2img):
        """Decode yaw angle and change it from local to global.i.
        Args:
            bbox (torch.Tensor): Bounding box predictions in shape
                [N, C] with yaws to be decoded.
            centers2d (torch.Tensor): Projected 3D-center on the image planes
                corresponding to the box predictions.
            dir_cls (torch.Tensor): Predicted direction classes.
            dir_offset (float): Direction offset before dividing all the
                directions into several classes.
            cam2img (torch.Tensor): Camera intrinsic matrix in shape [4, 4].
        Returns:
            torch.Tensor: Bounding boxes with decoded yaws.
        """
        if bbox.shape[0] > 0:
            dir_rot = limit_period(bbox[..., 6] - dir_offset, 0, np.pi)
            bbox[..., 6] = \
                dir_rot + dir_offset + np.pi * dir_cls.to(bbox.dtype)

        bbox[:, 6] = torch.atan2(centers2d[:, 0] - cam2img[0, 2],
                                 cam2img[0, 0]) + bbox[:, 6]

        return bbox
    

class PGDBBoxCoder(FCOS3DBBoxCoder):
    """Bounding box coder for PGD."""

    def encode(self, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels):
        pass

    def decode_2d(self,
                  bbox,
                  scale,
                  stride,
                  max_regress_range,
                  training,
                  pred_keypoints=False,
                  pred_bbox2d=True):
        """Decode regressed 2D attributes.
        Args:
            bbox (torch.Tensor): Raw bounding box predictions in shape
                [N, C, H, W].
            scale (tuple[`Scale`]): Learnable scale parameters.
            stride (int): Stride for a specific feature level.
            max_regress_range (int): Maximum regression range for a specific
                feature level.
            training (bool): Whether the decoding is in the training
                procedure.
            pred_keypoints (bool, optional): Whether to predict keypoints.
                Defaults to False.
            pred_bbox2d (bool, optional): Whether to predict 2D bounding
                boxes. Defaults to False.
        Returns:
            torch.Tensor: Decoded boxes.
        """
        clone_bbox = bbox.clone()
        if pred_keypoints:  
            scale_kpts = scale[3]            
            bbox[:, self.bbox_code_size:self.bbox_code_size + 16] = \
                torch.tanh(scale_kpts(clone_bbox[
                    :, self.bbox_code_size:self.bbox_code_size + 16]).float())

        if pred_bbox2d:
            scale_bbox2d = scale[-1]
            bbox[:, -4:] = scale_bbox2d(clone_bbox[:, -4:]).float()

        if self.norm_on_bbox:
            if pred_bbox2d:
                bbox[:, -4:] = F.relu(bbox.clone()[:, -4:])
            if not training:
                if pred_keypoints:
                    bbox[
                        :, self.bbox_code_size:self.bbox_code_size + 16] *= \
                           max_regress_range                    
        else:
            if pred_bbox2d:
                bbox[:, -4:] = bbox.clone()[:, -4:].exp()
        return bbox

    def decode_prob_depth(self, depth_cls_preds, depth_range, depth_unit,
                          division, num_depth_cls):
        """Decode probabilistic depth map.
        Args:
            depth_cls_preds (torch.Tensor): Depth probabilistic map in shape
                [..., self.num_depth_cls] (raw output before softmax).
            depth_range (tuple[float]): Range of depth estimation.
            depth_unit (int): Unit of depth range division.
            division (str): Depth division method. Options include 'uniform',
                'linear', 'log', 'loguniform'.
            num_depth_cls (int): Number of depth classes.
        Returns:
            torch.Tensor: Decoded probabilistic depth estimation.
        """
        if division == 'uniform': 
            depth_multiplier = depth_unit * \
                depth_cls_preds.new_tensor(
                    list(range(num_depth_cls))).reshape([1, -1])
            prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) *
                                depth_multiplier).sum(dim=-1)  
            return prob_depth_preds
        elif division == 'linear':
            split_pts = depth_cls_preds.new_tensor(list(
                range(num_depth_cls))).reshape([1, -1])
            depth_multiplier = depth_range[0] + (
                depth_range[1] - depth_range[0]) / \
                (num_depth_cls * (num_depth_cls - 1)) * \
                (split_pts * (split_pts+1))
            prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) *
                                depth_multiplier).sum(dim=-1)
            return prob_depth_preds
        elif division == 'log':
            split_pts = depth_cls_preds.new_tensor(list(
                range(num_depth_cls))).reshape([1, -1])
            start = max(depth_range[0], 1)
            end = depth_range[1]
            depth_multiplier = (np.log(start) +
                                split_pts * np.log(end / start) /
                                (num_depth_cls - 1)).exp()
            prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) *
                                depth_multiplier).sum(dim=-1)
            return prob_depth_preds
        elif division == 'loguniform':
            split_pts = depth_cls_preds.new_tensor(list(
                range(num_depth_cls))).reshape([1, -1])
            start = max(depth_range[0], 1)
            end = depth_range[1]
            log_multiplier = np.log(start) + \
                split_pts * np.log(end / start) / (num_depth_cls - 1)
            prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) *
                                log_multiplier).sum(dim=-1).exp()
            return prob_depth_preds
        else:
            raise NotImplementedError    

