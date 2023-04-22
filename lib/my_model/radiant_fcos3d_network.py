import numpy as np
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import torch
from torch import nn as nn
from mmcv.runner import auto_fp16, force_fp32
from mmcv.cnn import bias_init_with_prob, normal_init, ConvModule, Scale, build_conv_layer, build_norm_layer 
from mmdet.core import multi_apply
from mmdet3d.core import bbox3d2result, box3d_multiclass_nms, limit_period, xywhr2xyxyr
from lib.my_model.base_module import BaseModule
from lib.my_model.resnet import ResNet
from lib.my_model.fpn import FPN
from lib.my_model.focal_loss import FocalLoss
from lib.my_model.smooth_l1_loss import SmoothL1Loss
from lib.my_model.cross_entropy_loss import CrossEntropyLoss

loss_registry = dict(FocalLoss=FocalLoss, SmoothL1Loss=SmoothL1Loss, CrossEntropyLoss=CrossEntropyLoss)
INF = 1e8

class BaseDetector(BaseModule, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self, init_cfg=None):
        super(BaseDetector, self).__init__(init_cfg)
        self.fp16_enabled = False

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass


    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]


    async def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    def forward_test(self, imgs, img_metas, **kwargs):

        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')
                
        assert len(imgs) == len(img_metas) == 1

        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        return self.simple_test(imgs[0], img_metas[0], **kwargs)
       

    @auto_fp16(apply_to=('img', )) 
    def forward(self, img, img_metas, return_loss=True, model_mlp=None, **kwargs):
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, model_mlp=model_mlp, **kwargs)


    def _parse_losses(self, losses):       
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars


    def train_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        return outputs


    def val_step(self, data, optimizer=None):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        return outputs
    
    
    def create_data_step(self, data):
        return self.forward_create_data(**data)
    

class FusionConvBlock(BaseModule):
    
    def __init__(self,
             inplanes,
             planes,
             outplanes,
             conv_cfg=None,
             norm_cfg=dict(type='BN'),
             init_cfg=None):

        super(FusionConvBlock, self).__init__(init_cfg)
                
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(norm_cfg, outplanes, postfix=3)
        
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            bias=True)      
        self.add_module(self.norm1_name, norm1)
        
        self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                padding=1,
                bias=True)        
        self.add_module(self.norm2_name, norm2)
        
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            outplanes,
            kernel_size=1,
            bias=True)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        
    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)
        
    def forward(self, x):
        """Forward function."""
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.relu(out)

        return out


class SingleStageDetector(BaseDetector):

    def __init__(self,
                 backbone_img,            
                 backbone_other,          
                 neck_img=None,           
                 neck_fusion=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained_img=None,     
                 pretrained_other=None,   
                 eval_mono=None,         
                 init_cfg=None):          
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained_img:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone_img.pretrained = pretrained_img
            
        self.backbone_img = ResNet(**backbone_img)
        self.backbone_other = ResNet(**backbone_other)
        
        if neck_img is not None:
            self.neck_img = FPN(**neck_img)
            
        if neck_fusion is not None:
            self.neck_fusion = FPN(**neck_fusion)
            
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        
        self.bbox_head = FCOSFusion3DHead(**bbox_head)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
                
        im_feat_channels = [512, 1024, 2048]
        other_feat_channels=[128, 256, 512]
        
        self.fusion_convs = nn.ModuleList()
        
        for im_chn, other_chn in zip(im_feat_channels, other_feat_channels):
            inplanes = im_chn + other_chn
            planes = int(im_chn/4)
            outplanes = im_chn
            
            self.fusion_convs.append( 
                FusionConvBlock(inplanes, planes, outplanes) )

        self.eval_mono = eval_mono
            

    def extract_feat(self, img, radar_map):
        
        x_img = self.backbone_img(img)
        x_other = self.backbone_other(radar_map)
        
        x_cat = []
        for x_i, x_o in zip(x_img, x_other):
            x_cat.append(torch.cat([x_i, x_o], dim=1))
        
        for i in range(3):
            x_cat[i+1] = self.fusion_convs[i](x_cat[i+1])
            
        x_img = self.neck_img(x_img)
        x_cat = self.neck_fusion(x_cat)
                  
        return x_img, x_cat


    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs


class SingleStageMono3DDetector(SingleStageDetector):
    """Base class for monocular 3D single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def extract_feats(self, imgs):
        """Directly extract features from the backbone+neck."""
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    def forward_train(self,
                      img,
                      img_metas,
                      radar_map,
                      gt_bboxes,
                      gt_labels,   
                      gt_bboxes_3d,
                      gt_labels_3d,
                      centers2d,
                      depths,
                      radar_pts,
                      attr_labels=None,
                      gt_bboxes_ignore=None):
        
        x_img, x_cat = self.extract_feat(img, radar_map)
        losses = self.bbox_head.forward_train(x_img, x_cat, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_3d, gt_labels_3d,
                                              centers2d, depths, attr_labels,
                                              gt_bboxes_ignore, radar_pts)
        return losses
    
    
    def simple_test(self, img, img_metas, model_mlp, radar_map, radar_pts, rescale=False): 

        x_img, x_cat = self.extract_feat(img, radar_map[0])        
        outs = self.bbox_head(x_img, x_cat)
        
        if self.eval_mono:
            bbox_outputs = self.bbox_head.get_bboxes(
                *outs, img_metas, radar_pts[0][0], rescale=rescale)     
        else:
            bbox_outputs = self.bbox_head.get_bboxes_fusion(
                *outs, img_metas, radar_pts[0][0], rescale=rescale, model_mlp=model_mlp)     

        bbox_img = [
            bbox3d2result(bboxes, scores, labels, attrs)
            for bboxes, scores, labels, attrs in bbox_outputs
        ]

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox

        return bbox_list


    def forward_create_data(self,
                      img,
                      img_metas,
                      radar_map,
                      gt_bboxes,
                      gt_labels,   
                      gt_bboxes_3d,
                      gt_labels_3d,
                      centers2d,
                      depths,
                      radar_pts,
                      attr_labels=None,
                      gt_bboxes_ignore=None):

        
        x_img, x_cat = self.extract_feat(img, radar_map)
        data = self.bbox_head.forward_create_data(x_img, x_cat, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_3d,
                                              gt_labels_3d, centers2d, depths,
                                              attr_labels, gt_bboxes_ignore,
                                              radar_pts)
        return data    
    

class FCOSFusion3D(SingleStageMono3DDetector):
    def __init__(self,
                 backbone_img,
                 backbone_other,                 
                 neck_img,
                 neck_fusion,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained_img=None,
                 pretrained_other=None,
                 eval_mono=None):
        super(FCOSFusion3D, self).__init__(backbone_img,
                                         backbone_other,
                                         neck_img,
                                         neck_fusion,
                                         bbox_head, 
                                         train_cfg,
                                         test_cfg,
                                         pretrained_img,
                                         pretrained_other,
                                         eval_mono)
        
  
class BaseMono3DDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for Monocular 3D DenseHeads."""

    def __init__(self, init_cfg=None):
        super(BaseMono3DDenseHead, self).__init__(init_cfg=init_cfg)

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    def forward_train(self,
                      x_img,
                      x_cat,
                      img_metas, 
                      gt_bboxes,
                      gt_labels=None, 
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      centers2d=None,
                      depths=None,
                      attr_labels=None,
                      gt_bboxes_ignore=None,
                      radar_pts=None,
                      proposal_cfg=None,
                      **kwargs):
       
        outs = self(x_img, x_cat)
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_bboxes_3d,
                              gt_labels_3d, centers2d, depths, attr_labels,
                              img_metas, radar_pts)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)  
        
        return losses


    def forward_create_data(self,
                            x_img,
                            x_cat,
                            img_metas,  
                            gt_bboxes,
                            gt_labels=None,  
                            gt_bboxes_3d=None,
                            gt_labels_3d=None,
                            centers2d=None,
                            depths=None,
                            attr_labels=None,
                            gt_bboxes_ignore=None,
                            radar_pts=None,
                            proposal_cfg=None,
                            **kwargs):
       
        outs = self(x_img, x_cat)
        inputs = outs + (gt_bboxes, gt_labels, gt_bboxes_3d,
                              gt_labels_3d, centers2d, depths, attr_labels,
                              img_metas, radar_pts)
        
        data_label = self.create_mlp_data(*inputs, gt_bboxes_ignore=gt_bboxes_ignore) 
        
        return data_label


class AnchorFreeMono3DHead(BaseMono3DDenseHead):
   
    _version = 1

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()
        
        self._init_radar_cls_convs()
        self._init_radar_reg_convs()

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels   
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
            
    def _init_radar_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.radar_reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.radar_feat_channels   
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.radar_reg_convs.append(
                ConvModule(               
                    chn,
                    self.radar_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))        
    

    def _init_branch(self, conv_channels=(64), conv_strides=(1)):  
        """Initialize conv layers as a prediction branch."""
        conv_before_pred = nn.ModuleList()
        if isinstance(conv_channels, int):
            conv_channels = [self.feat_channels] + [conv_channels]
            conv_strides = [conv_strides]
        else:
            conv_channels = [self.feat_channels] + list(conv_channels)
            conv_strides = list(conv_strides)
        for i in range(len(conv_strides)):
            conv_before_pred.append(
                ConvModule(
                    conv_channels[i],
                    conv_channels[i + 1],
                    3,
                    stride=conv_strides[i],
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

        return conv_before_pred


    def init_weights(self):
        super().init_weights()
        bias_cls = bias_init_with_prob(0.01)
        if self.use_direction_classifier:
            normal_init(self.conv_dir_cls, std=0.01, bias=bias_cls)
        if self.pred_attrs:
            normal_init(self.conv_attr, std=0.01, bias=bias_cls)


    def forward_single(self, x_img, x_cat):
        
        cls_feat = x_img
        reg_feat = x_img    
        radar_cls_feat = x_cat
        radar_reg_feat = x_cat
                
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        clone_cls_feat = cls_feat.clone()
        for conv_cls_prev_layer in self.conv_cls_prev:
            clone_cls_feat = conv_cls_prev_layer(clone_cls_feat)
        cls_score = self.conv_cls(clone_cls_feat)
        
        for rd_cls_layer in self.radar_cls_convs:
            radar_cls_feat = rd_cls_layer(radar_cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = []
        for i in range(len(self.group_reg_dims)):  
            clone_reg_feat = reg_feat.clone()
            if len(self.reg_branch[i]) > 0:    
                for conv_reg_prev_layer in self.conv_reg_prevs[i]:
                    clone_reg_feat = conv_reg_prev_layer(clone_reg_feat)
            bbox_pred.append(self.conv_regs[i](clone_reg_feat))
        bbox_pred = torch.cat(bbox_pred, dim=1)

        dir_cls_pred = None
        if self.use_direction_classifier:
            clone_reg_feat = reg_feat.clone()
            for conv_dir_cls_prev_layer in self.conv_dir_cls_prev:
                clone_reg_feat = conv_dir_cls_prev_layer(clone_reg_feat)
            dir_cls_pred = self.conv_dir_cls(clone_reg_feat)

        attr_pred = None
        if self.pred_attrs: 
            clone_cls_feat = cls_feat.clone()
            for conv_attr_prev_layer in self.conv_attr_prev:
                clone_cls_feat = conv_attr_prev_layer(clone_cls_feat)
            attr_pred = self.conv_attr(clone_cls_feat)
                   
        for rd_reg_layer in self.radar_reg_convs:
            radar_reg_feat = rd_reg_layer(radar_reg_feat)
               
        return cls_score, bbox_pred, dir_cls_pred, attr_pred, cls_feat, \
            reg_feat, radar_cls_feat, radar_reg_feat

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points of a single scale level."""
        h, w = featmap_size  
        x_range = torch.arange(w, dtype=dtype, device=device)
        y_range = torch.arange(h, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)  
        if flatten:  
            y = y.flatten()
            x = x.flatten()
        return y, x

    def get_points(self, featmap_sizes, dtype, device, flatten=False):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self._get_points_single(featmap_sizes[i], self.strides[i],  
                                        dtype, device, flatten))            
        return mlvl_points  


class AnchorFreeMono3DHead2(AnchorFreeMono3DHead):
   
    _version = 1

    def __init__(
            self,
            num_classes,
            in_channels,
            feat_channels=256,
            cls_feat_channels=64,           
            stacked_convs=4,
            strides=(4, 8, 16, 32, 64), 
            dcn_on_last_conv=False,
            conv_bias='auto',
            background_label=None,
            use_direction_classifier=True,
            diff_rad_by_sin=True,
            dir_offset=0,
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            loss_dir=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_attr=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            bbox_code_size=9,  
            pred_attrs=False,
            num_attrs=9,  
            pred_velo=False,
            pred_bbox2d=False,
            group_reg_dims=(2, 1, 3, 1, 2), 
            cls_branch=(128, 64),
            reg_branch=(
                (128, 64), 
                (128, 64), 
                (64, ), 
                (64, ), 
                () 
            ),
            dir_branch=(64, ),
            attr_branch=(64, ),
            conv_cfg=None,
            norm_cfg=None,
            train_cfg=None,
            test_cfg=None,
            init_cfg=None):
        super(AnchorFreeMono3DHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.cls_feat_channels = cls_feat_channels
        
        self.radar_feat_channels = feat_channels         
        self.radar_cls_feat_channels = cls_feat_channels  
                
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.use_direction_classifier = use_direction_classifier
        self.diff_rad_by_sin = diff_rad_by_sin
        self.dir_offset = dir_offset
        
        self.loss_cls = self.build_loss(loss_cls)
        self.loss_bbox = self.build_loss(loss_bbox)
        self.loss_dir = self.build_loss(loss_dir)
        
        self.bbox_code_size = bbox_code_size
        self.group_reg_dims = list(group_reg_dims)
        self.cls_branch = cls_branch
        self.reg_branch = reg_branch
        assert len(reg_branch) == len(group_reg_dims), 'The number of '\
            'element in reg_branch and group_reg_dims should be the same.'
        self.pred_velo = pred_velo
        self.pred_bbox2d = pred_bbox2d
        self.out_channels = []
        for reg_branch_channels in reg_branch: 
            if len(reg_branch_channels) > 0:
                self.out_channels.append(reg_branch_channels[-1])
            else:
                self.out_channels.append(-1)  
        self.dir_branch = dir_branch
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.background_label = (
            num_classes if background_label is None else background_label)

        assert (self.background_label == 0
                or self.background_label == num_classes)
        self.pred_attrs = pred_attrs
        self.attr_background_label = -1
        self.num_attrs = num_attrs
        if self.pred_attrs:
            self.attr_background_label = num_attrs
            
            self.loss_attr = self.build_loss(loss_attr)
            
            self.attr_branch = attr_branch

        self._init_layers()
        if init_cfg is None:
            self.init_cfg = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=dict(
                    type='Normal', name='conv_cls', std=0.01, bias_prob=0.01))
                            
    def build_loss(self, cfg):
        cfg_ = cfg.copy()
        loss_type = cfg_.pop('type')
        loss_cls = loss_registry.get(loss_type)
        loss = loss_cls(**cfg_)
        return loss
       
    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):  
            chn = self.in_channels if i == 0 else self.cls_feat_channels 
            if self.dcn_on_last_conv and i == self.stacked_convs - 1: 
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg   
            self.cls_convs.append(
                ConvModule(       
                    chn,
                    self.cls_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,        
                    norm_cfg=self.norm_cfg,   
                    bias=self.conv_bias))    

    def _init_radar_cls_convs(self):
        """Initialize classification conv layers of the radar head."""
        self.radar_cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):  
            chn = self.in_channels if i == 0 else self.radar_cls_feat_channels 
            if self.dcn_on_last_conv and i == self.stacked_convs - 1: 
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg   
            self.radar_cls_convs.append(
                ConvModule(       
                    chn,
                    self.radar_cls_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,       
                    norm_cfg=self.norm_cfg,  
                    bias=self.conv_bias))    
                         
    def _init_cls_branch(self, conv_channels=(64), conv_strides=(1)):
        """Initialize conv layers as a prediction branch."""
        conv_before_pred = nn.ModuleList()
        if isinstance(conv_channels, int):
            conv_channels = [self.cls_feat_channels] + [conv_channels]
            conv_strides = [conv_strides]
        else: 
            conv_channels = [self.cls_feat_channels] + list(conv_channels)  
            conv_strides = list(conv_strides)
        for i in range(len(conv_strides)):
            conv_before_pred.append(
                ConvModule(
                    conv_channels[i],
                    conv_channels[i + 1],
                    3,
                    stride=conv_strides[i],
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,  
                    bias=self.conv_bias))

        return conv_before_pred
        
    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls_prev = self._init_cls_branch(   
            conv_channels=self.cls_branch,                      
            conv_strides=(1, ) * len(self.cls_branch))         
        self.conv_cls = nn.Conv2d(self.cls_branch[-1], self.cls_out_channels, 1) 
        self.conv_reg_prevs = nn.ModuleList()  
        self.conv_regs = nn.ModuleList()       
        for i in range(len(self.group_reg_dims)):  
            reg_dim = self.group_reg_dims[i]
            reg_branch_channels = self.reg_branch[i]   
            out_channel = self.out_channels[i]   
            if len(reg_branch_channels) > 0: 
                self.conv_reg_prevs.append(
                    self._init_branch(
                        conv_channels=reg_branch_channels,
                        conv_strides=(1, ) * len(reg_branch_channels)))
                self.conv_regs.append(nn.Conv2d(out_channel, reg_dim, 1))
            else:
                self.conv_reg_prevs.append(None)  
                self.conv_regs.append(
                    nn.Conv2d(self.feat_channels, reg_dim, 1))
        if self.use_direction_classifier: 
            self.conv_dir_cls_prev = self._init_branch(
                conv_channels=self.dir_branch,  
                conv_strides=(1, ) * len(self.dir_branch))
            self.conv_dir_cls = nn.Conv2d(self.dir_branch[-1], 2, 1)
        if self.pred_attrs: 
            self.conv_attr_prev = self._init_branch(
                conv_channels=self.attr_branch,
                conv_strides=(1, ) * len(self.attr_branch))
            self.conv_attr = nn.Conv2d(self.attr_branch[-1], self.num_attrs, 1)

    
class FCOSFusion3DHead(AnchorFreeMono3DHead2):

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 48), (48, 96), (96, 192), (192, 384),
                                 (384, INF)),
                 center_sampling=True,
                 center_sample_radius=1.5,
                 norm_on_bbox=True,
                 centerness_on_reg=True,
                 centerness_alpha=2.5,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_dir=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_attr=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 centerness_branch = (64, ),
                 radarOffset_branch = (128,),
                 radarDepthOffset_branch = (128,),
                 radarClass_branch = (64,),                 
                 init_cfg=None,
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.centerness_alpha = centerness_alpha
        self.centerness_branch = centerness_branch
        self.radarOffset_branch = radarOffset_branch
        self.radarDepthOffset_branch = radarDepthOffset_branch
        self.radarClass_branch = radarClass_branch
        
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_dir=loss_dir,
            loss_attr=loss_attr,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)

        self.loss_centerness = self.build_loss(loss_centerness)
        if init_cfg is None:
            self.init_cfg = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=dict(
                    type='Normal', name='conv_cls', std=0.01, bias_prob=0.01))
    
    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness_prev = self._init_branch(    
            conv_channels=self.centerness_branch,          
            conv_strides=(1, ) * len(self.centerness_branch))
        self.conv_centerness = nn.Conv2d(self.centerness_branch[-1], 1, 1) 
        self.scales = nn.ModuleList([ 
            nn.ModuleList([Scale(1.0) for _ in range(3)]) for _ in self.strides 
        ]) 
                        
        self.conv_radarOffset_prev = self._init_branch(     
           conv_channels=self.radarOffset_branch,           
           conv_strides=(1, ) * len(self.radarOffset_branch))
        self.conv_radarOffset = nn.Conv2d(self.radarOffset_branch[-1], 2, 1)
        
        self.conv_radarDepthOffset_prev = self._init_branch(     
           conv_channels=self.radarDepthOffset_branch,           
           conv_strides=(1, ) * len(self.radarDepthOffset_branch))
        self.conv_radarDepthOffset = nn.Conv2d(self.radarDepthOffset_branch[-1], 1, 1)
               
        self.conv_radarClass_prev = self._init_cls_branch(    
           conv_channels=self.radarClass_branch,                    
           conv_strides=(1, ) * len(self.radarClass_branch))        
        self.conv_radarClass = nn.Conv2d(self.radarClass_branch[-1], self.cls_out_channels, 1) 
               
        self.radar_scales = nn.ModuleList([ 
            nn.ModuleList([Scale(1.0) for _ in range(2)]) for _ in self.strides 
        ]) 
        
        
    def forward(self, feats_img, feats_cat):   
        return multi_apply(self.forward_single, feats_img, feats_cat, self.scales, self.radar_scales,
                           self.strides)

    def forward_single(self, x_img, x_cat, scale, radar_scale, stride):
        
        cls_score, bbox_pred, dir_cls_pred, attr_pred, cls_feat, reg_feat,\
        radar_cls_feat, radar_reg_feat = super().forward_single(x_img, x_cat)

        if self.centerness_on_reg:  
            clone_reg_feat = reg_feat.clone()
            for conv_centerness_prev_layer in self.conv_centerness_prev:
                clone_reg_feat = conv_centerness_prev_layer(clone_reg_feat)
            centerness = self.conv_centerness(clone_reg_feat)
        else:
            clone_cls_feat = cls_feat.clone()
            for conv_centerness_prev_layer in self.conv_centerness_prev:
                clone_cls_feat = conv_centerness_prev_layer(clone_cls_feat)
            centerness = self.conv_centerness(clone_cls_feat)
                       
        clone_reg_feat = radar_reg_feat.clone()  
        for conv_radarOffset_prev_layer in self.conv_radarOffset_prev:
            clone_reg_feat = conv_radarOffset_prev_layer(clone_reg_feat)
        radarOffset = self.conv_radarOffset(clone_reg_feat)
        
        clone_reg_feat = radar_reg_feat.clone()
        for conv_radarDepthOffset_prev_layer in self.conv_radarDepthOffset_prev:
            clone_reg_feat = conv_radarDepthOffset_prev_layer(clone_reg_feat)
        radarDepthOffset = self.conv_radarDepthOffset(clone_reg_feat)

        clone_cls_feat = radar_cls_feat.clone()
        for conv_radarClass_prev_layer in self.conv_radarClass_prev:
            clone_cls_feat = conv_radarClass_prev_layer(clone_cls_feat)
        radarClass = self.conv_radarClass(clone_cls_feat) 
                
        scale1, scale2 = radar_scale[0:2]
        radarOffset = scale1(radarOffset).float()
        radarDepthOffset = scale2(radarDepthOffset).float()        
            
        scale_offset, scale_depth, scale_size = scale[0:3]

        clone_bbox_pred = bbox_pred.clone() 
        bbox_pred[:, :2] = scale_offset(clone_bbox_pred[:, :2]).float()
        bbox_pred[:, 2] = scale_depth(clone_bbox_pred[:, 2]).float()
        bbox_pred[:, 3:6] = scale_size(clone_bbox_pred[:, 3:6]).float()

        bbox_pred[:, 2] = bbox_pred[:, 2].exp()     
        bbox_pred[:, 3:6] = bbox_pred[:, 3:6].exp() + 1e-6  

        return cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, radarOffset, radarDepthOffset, radarClass

    @staticmethod
    def add_sin_difference(boxes1, boxes2):
        """Convert the rotation difference to difference in sine function.

        Args:
            boxes1 (torch.Tensor): Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2 (torch.Tensor): Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.

        Returns:
            tuple[torch.Tensor]: ``boxes1`` and ``boxes2`` whose 7th \
                dimensions are changed.
        """
        rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(
            boxes2[..., 6:7])
        rad_tg_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(boxes2[...,
                                                                         6:7])
        boxes1 = torch.cat(
            [boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                           dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(reg_targets,
                             dir_offset=0,
                             num_bins=2,
                             one_hot=True):
        """Encode direction to 0 ~ num_bins-1.

        Args:
            reg_targets (torch.Tensor): Bbox regression targets.
            dir_offset (int): Direction offset.
            num_bins (int): Number of bins to divide 2*PI.
            one_hot (bool): Whether to encode as one hot.

        Returns:
            torch.Tensor: Encoded direction targets.
        """
        rot_gt = reg_targets[..., 6]
        offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot /
                                      (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        if one_hot:
            dir_targets = torch.zeros(
                *list(dir_cls_targets.shape),
                num_bins,
                dtype=reg_targets.dtype,
                device=dir_cls_targets.device)
            dir_targets.scatter_(dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets


    def loss(self,
             cls_scores,
             bbox_preds,
             dir_cls_preds,
             attr_preds,
             centernesses,
             radarOffsets,
             radarDepthOffsets,
             radarClasses,
             gt_bboxes,
             gt_labels,
             gt_bboxes_3d,
             gt_labels_3d,
             centers2d,
             depths,
             attr_labels,
             img_metas,
             radar_pts,
             gt_bboxes_ignore=None):

        def transform_pred(prds):
            out_channels = prds[0].shape[1]
            batch_size = prds[0].shape[0]
            
            prds_tmp = [ prd.permute(0, 2, 3, 1).reshape(batch_size, -1, out_channels) for prd in prds ]
            
            prds_tmp2 = []        
            for im_idx in range(batch_size):
                prds_tmp2.append(torch.cat([tmp_one_level[im_idx] for tmp_one_level in prds_tmp], dim=0))        
            prds_tmp = torch.cat(prds_tmp2, dim=0)
            return prds_tmp
        
        assert len(cls_scores) == len(bbox_preds) == len(centernesses) == len(attr_preds) 
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]  
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype, 
                                           bbox_preds[0].device) 
        labels_3d, bbox_targets_3d, centerness_targets, attr_targets = \
            self.get_targets( 
                all_level_points, gt_bboxes, gt_labels, gt_bboxes_3d,
                gt_labels_3d, centers2d, depths, attr_labels)   

        radar_labels, radar_delta_offsets_tgt, radar_delta_depths_tgt, radar_depths_tgt, msk_positive, indices_pts = \
        self.get_radar_targets(all_level_points, gt_bboxes, gt_labels, gt_bboxes_3d, centers2d, depths, radar_pts)
               
        num_imgs = cls_scores[0].size(0) 
        
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ] 
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, sum(self.group_reg_dims))
            for bbox_pred in bbox_preds
        ] 
        flatten_dir_cls_preds = [
            dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)
            for dir_cls_pred in dir_cls_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores) 
        flatten_bbox_preds = torch.cat(flatten_bbox_preds) 
        flatten_dir_cls_preds = torch.cat(flatten_dir_cls_preds)  
        flatten_centerness = torch.cat(flatten_centerness) 
        flatten_labels_3d = torch.cat(labels_3d) 
        flatten_bbox_targets_3d = torch.cat(bbox_targets_3d) 
        flatten_centerness_targets = torch.cat(centerness_targets)  
       
        bg_class_ind = self.num_classes       
        pos_inds = torch.nonzero( (flatten_labels_3d >= 0) & (flatten_labels_3d < bg_class_ind), as_tuple=False).reshape(-1)  
        
        num_pos = len(pos_inds)  

        loss_cls = self.loss_cls(   
            flatten_cls_scores,
            flatten_labels_3d,
            avg_factor=num_pos + num_imgs) 

        pos_bbox_preds = flatten_bbox_preds[pos_inds] 
        pos_dir_cls_preds = flatten_dir_cls_preds[pos_inds]  
        pos_centerness = flatten_centerness[pos_inds] 
        
        if self.pred_attrs:  
            flatten_attr_preds = [
                attr_pred.permute(0, 2, 3, 1).reshape(-1, self.num_attrs)
                for attr_pred in attr_preds
            ]
            flatten_attr_preds = torch.cat(flatten_attr_preds)  
            flatten_attr_targets = torch.cat(attr_targets)  
            pos_attr_preds = flatten_attr_preds[pos_inds] 

        if num_pos > 0:
            pos_bbox_targets_3d = flatten_bbox_targets_3d[pos_inds]  
            pos_centerness_targets = flatten_centerness_targets[pos_inds]  
                       
            if self.pred_attrs: 
                pos_attr_targets = flatten_attr_targets[pos_inds]  
            bbox_weights = pos_centerness_targets.new_ones(    
                len(pos_centerness_targets), sum(self.group_reg_dims))  
            equal_weights = pos_centerness_targets.new_ones(    
                pos_centerness_targets.shape)

            code_weight = self.train_cfg.get('code_weight', None)  
            if code_weight:
                assert len(code_weight) == sum(self.group_reg_dims)  
                bbox_weights = bbox_weights * bbox_weights.new_tensor(  
                    code_weight)

            if self.use_direction_classifier:  
                pos_dir_cls_targets = self.get_direction_target(  
                    pos_bbox_targets_3d, self.dir_offset, one_hot=False)  

            if self.diff_rad_by_sin: 
                pos_bbox_preds, pos_bbox_targets_3d = self.add_sin_difference(  
                    pos_bbox_preds, pos_bbox_targets_3d)

            loss_offset = self.loss_bbox(  
                pos_bbox_preds[:, :2],
                pos_bbox_targets_3d[:, :2],
                weight=bbox_weights[:, :2],
                avg_factor=equal_weights.sum())
            loss_depth = self.loss_bbox(
                pos_bbox_preds[:, 2],
                pos_bbox_targets_3d[:, 2],
                weight=bbox_weights[:, 2],
                avg_factor=equal_weights.sum())
            loss_size = self.loss_bbox(
                pos_bbox_preds[:, 3:6],
                pos_bbox_targets_3d[:, 3:6],
                weight=bbox_weights[:, 3:6],
                avg_factor=equal_weights.sum())
            loss_rotsin = self.loss_bbox(  
                pos_bbox_preds[:, 6],
                pos_bbox_targets_3d[:, 6],
                weight=bbox_weights[:, 6],
                avg_factor=equal_weights.sum())
            loss_velo = None
            if self.pred_velo:  
                loss_velo = self.loss_bbox(
                    pos_bbox_preds[:, 7:9],
                    pos_bbox_targets_3d[:, 7:9],
                    weight=bbox_weights[:, 7:9],
                    avg_factor=equal_weights.sum())

            loss_centerness = self.loss_centerness(pos_centerness,    
                                                   pos_centerness_targets)
          
            loss_dir = None           
            if self.use_direction_classifier:  
                loss_dir = self.loss_dir(  
                    pos_dir_cls_preds,    
                    pos_dir_cls_targets,  
                    equal_weights,
                    avg_factor=equal_weights.sum())

            loss_attr = None
            if self.pred_attrs:
                loss_attr = self.loss_attr(
                    pos_attr_preds,   
                    pos_attr_targets,  
                    pos_centerness_targets,
                    avg_factor=pos_centerness_targets.sum())
            
        else:
            loss_offset = pos_bbox_preds[:, :2].sum() 
            loss_depth = pos_bbox_preds[:, 2].sum()
            loss_size = pos_bbox_preds[:, 3:6].sum()
            loss_rotsin = pos_bbox_preds[:, 6].sum()
            loss_velo = None
            if self.pred_velo:
                loss_velo = pos_bbox_preds[:, 7:9].sum()
            loss_centerness = pos_centerness.sum()
            loss_dir = None
            if self.use_direction_classifier:
                loss_dir = pos_dir_cls_preds.sum()
            loss_attr = None
            if self.pred_attrs:
                loss_attr = pos_attr_preds.sum()
            
        loss_dict = dict(
            loss_cls=loss_cls,
            loss_offset=loss_offset,
            loss_depth=loss_depth,
            loss_size=loss_size,
            loss_rotsin=loss_rotsin,
            loss_centerness=loss_centerness)

        if loss_velo is not None:
            loss_dict['loss_velo'] = loss_velo

        if loss_dir is not None:
            loss_dict['loss_dir'] = loss_dir

        if loss_attr is not None:
            loss_dict['loss_attr'] = loss_attr
             
        radarOffsets = transform_pred(radarOffsets) 
        radarDepthOffsets = transform_pred(radarDepthOffsets)  
        radarClasses = transform_pred(radarClasses)         
        radarOffsets = radarOffsets[indices_pts[msk_positive]]    
        radarDepthOffsets = radarDepthOffsets[indices_pts[msk_positive]].squeeze(1)   
        radarClasses = radarClasses[indices_pts]            
        num_pos_radar = torch.sum(msk_positive)
                
        num_radar = len(radarClasses)
        if num_radar > 0:
            loss_radarClass = self.loss_cls(  
                                    radarClasses,
                                    radar_labels,
                                    avg_factor=num_pos_radar + num_imgs) 
        else:
            loss_radarClass = radarClasses.sum()
        
        
        if num_pos_radar > 0:    
            loss_radarOffset = self.loss_bbox(  
                                    radarOffsets,
                                    radar_delta_offsets_tgt,
                                    weight=torch.ones_like(radarOffsets),
                                    avg_factor=radarOffsets.shape[0])
            
            code_weight = self.train_cfg.get('code_weight', None)            
            loss_radarDepthOffset = self.loss_bbox(
                                    radarDepthOffsets,
                                    radar_delta_depths_tgt,
                                    weight=torch.ones_like(radarDepthOffsets) * code_weight[2], 
                                    avg_factor=radarDepthOffsets.shape[0])           
        else:
            loss_radarOffset = radarOffsets.sum()  
            loss_radarDepthOffset = radarDepthOffsets.sum()
            
            
        loss_dict['loss_radarClass'] = loss_radarClass
        loss_dict['loss_radarOffset'] = loss_radarOffset
        loss_dict['loss_radarDepthOffset'] = loss_radarDepthOffset
         
        return loss_dict


    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'dir_cls_preds', 'attr_preds',
                  'centernesses', 'radarOffsets', 'radarDepths','radarDepthOffsets',
                  'radarClasses'))
    def get_bboxes(self,
                   cls_scores,  
                   bbox_preds,  
                   dir_cls_preds, 
                   attr_preds,   
                   centernesses, 
                   radarOffsets,  
                   radarDepthOffsets, 
                   radarClasses,
                   img_metas,
                   radar_pts,   
                   cfg=None,
                   rescale=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * 2. (bin = 2)
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(dir_cls_preds) == \
            len(centernesses) == len(attr_preds)  
        num_levels = len(cls_scores) 
         
        if self.norm_on_bbox:
            for i in range(num_levels):
                bbox_preds[i][:, :2] *= self.strides[i]

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]  
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,  
                                      bbox_preds[0].device) 
        result_list = []
        for img_id in range(len(img_metas)): 
            cls_score_list = [  
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [   
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            if self.use_direction_classifier: 
                dir_cls_pred_list = [  
                    dir_cls_preds[i][img_id].detach()
                    for i in range(num_levels)
                ]
            else:
                dir_cls_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [2, *cls_scores[i][img_id].shape[1:]], 0).detach()
                    for i in range(num_levels)
                ]
            if self.pred_attrs: 
                attr_pred_list = [  
                    attr_preds[i][img_id].detach() for i in range(num_levels)
                ]
            else:
                attr_pred_list = [ 
                    cls_scores[i][img_id].new_full(
                        [self.num_attrs, *cls_scores[i][img_id].shape[1:]],
                        self.attr_background_label).detach()
                    for i in range(num_levels)
                ]
            centerness_pred_list = [  
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            input_meta = img_metas[img_id]
            det_bboxes = self._get_bboxes_single(  
                cls_score_list, bbox_pred_list, dir_cls_pred_list,
                attr_pred_list, centerness_pred_list, mlvl_points, input_meta,
                cfg, rescale) 
            result_list.append(det_bboxes)  
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           dir_cls_preds,
                           attr_preds,
                           centernesses,
                           mlvl_points,
                           input_meta,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * bbox_code_size, H, W).
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on a single scale level with shape \
                (num_points * 2, H, W)
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 2).
            input_meta (dict): Metadata of input image.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            tuples[Tensor]: Predicted 3D boxes, scores, labels and attributes.
        """
        view = np.array(input_meta['cam2img']) 
        scale_factor = input_meta['scale_factor'] 
        cfg = self.test_cfg if cfg is None else cfg  
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)  
        mlvl_centers2d = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_attr_scores = []
        mlvl_centerness = []

        for cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, \
                points in zip(cls_scores, bbox_preds, dir_cls_preds,
                              attr_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape( 
                -1, self.cls_out_channels).sigmoid()
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2) 
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]  
            attr_pred = attr_pred.permute(1, 2, 0).reshape(-1, self.num_attrs)  
            attr_score = torch.max(attr_pred, dim=-1)[1]  
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid() 

            bbox_pred = bbox_pred.permute(1, 2,                
                                          0).reshape(-1,
                                                     sum(self.group_reg_dims)) 
            bbox_pred = bbox_pred[:, :self.bbox_code_size] 
            nms_pre = cfg.get('nms_pre', -1)  
            if nms_pre > 0 and scores.shape[0] > nms_pre: 
                max_scores, _ = (scores * centerness[:, None]).max(dim=1) 
                _, topk_inds = max_scores.topk(nms_pre) 
                points = points[topk_inds, :]           
                bbox_pred = bbox_pred[topk_inds, :]     
                scores = scores[topk_inds, :]           
                dir_cls_pred = dir_cls_pred[topk_inds, :]  
                centerness = centerness[topk_inds]         
                dir_cls_score = dir_cls_score[topk_inds]   
                attr_score = attr_score[topk_inds]         
           
            bbox_pred[:, :2] = points - bbox_pred[:, :2]    
            if rescale:
                bbox_pred[:, :2] /= bbox_pred[:, :2].new_tensor(scale_factor) 
            pred_center2d = bbox_pred[:, :3].clone()  
            bbox_pred[:, :3] = self.pts2Dto3D(bbox_pred[:, :3], view) 
            mlvl_centers2d.append(pred_center2d)
            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_attr_scores.append(attr_score)
            mlvl_centerness.append(centerness)

        mlvl_centers2d = torch.cat(mlvl_centers2d)  
        mlvl_bboxes = torch.cat(mlvl_bboxes)         
        mlvl_dir_scores = torch.cat(mlvl_dir_scores) 

        if mlvl_bboxes.shape[0] > 0:
            dir_rot = limit_period(mlvl_bboxes[..., 6] - self.dir_offset, 0,
                                   np.pi)
            mlvl_bboxes[..., 6] = (                 
                dir_rot + self.dir_offset +
                np.pi * mlvl_dir_scores.to(mlvl_bboxes.dtype))

        cam_intrinsic = mlvl_centers2d.new_zeros((4, 4))
        cam_intrinsic[:view.shape[0], :view.shape[1]] = \
            mlvl_centers2d.new_tensor(view)
        mlvl_bboxes[:, 6] = torch.atan2(  
            mlvl_centers2d[:, 0] - cam_intrinsic[0, 2],
            cam_intrinsic[0, 0]) + mlvl_bboxes[:, 6]
        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d']( 
            mlvl_bboxes, box_dim=self.bbox_code_size,
            origin=(0.5, 0.5, 0.5)).bev)

        mlvl_scores = torch.cat(mlvl_scores) 
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)  
        mlvl_attr_scores = torch.cat(mlvl_attr_scores)  
        mlvl_centerness = torch.cat(mlvl_centerness)  
       
        mlvl_nms_scores = mlvl_scores * mlvl_centerness[:, None] 
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms, 
                                       mlvl_nms_scores, cfg.score_thr,  
                                       cfg.max_per_img, cfg, mlvl_dir_scores,  
                                       mlvl_attr_scores)
        bboxes, scores, labels, dir_scores, attrs = results  
        attrs = attrs.to(labels.dtype) 
        bboxes = input_meta['box_type_3d'](  
            bboxes, box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5))
        
        if not self.pred_attrs:
            attrs = None

        return bboxes, scores, labels, attrs


    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'dir_cls_preds', 'attr_preds',
              'centernesses', 'radarOffsets', 'radarDepths','radarDepthOffsets',
               'radarClasses'))
    def get_bboxes_fusion(self,
                   cls_scores,  
                   bbox_preds, 
                   dir_cls_preds, 
                   attr_preds,   
                   centernesses, 
                   radarOffsets,  
                   radarDepthOffsets, 
                   radarClasses,
                   img_metas,
                   radar_pts,  
                   cfg=None,
                   rescale=None,
                   model_mlp=None):

        assert len(cls_scores) == len(bbox_preds) == len(dir_cls_preds) == \
            len(centernesses) == len(attr_preds) 
        num_levels = len(cls_scores)  
        
        if self.norm_on_bbox:
            for i in range(num_levels):
                bbox_preds[i][:, :2] *= self.strides[i]
                radarOffsets[i] *= self.strides[i]

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]  
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,  
                                      bbox_preds[0].device)               
        model_mlp.to(cls_scores[0].device) 
        result_list = []
        for img_id in range(len(img_metas)): 
            cls_score_list = [  
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [   
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            if self.use_direction_classifier: 
                dir_cls_pred_list = [  
                    dir_cls_preds[i][img_id].detach()
                    for i in range(num_levels)
                ]
            else:
                dir_cls_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [2, *cls_scores[i][img_id].shape[1:]], 0).detach()
                    for i in range(num_levels)
                ]
            if self.pred_attrs: 
                attr_pred_list = [ 
                    attr_preds[i][img_id].detach() for i in range(num_levels)
                ]
            else:
                attr_pred_list = [ 
                    cls_scores[i][img_id].new_full(
                        [self.num_attrs, *cls_scores[i][img_id].shape[1:]],
                        self.attr_background_label).detach()
                    for i in range(num_levels)
                ]
            centerness_pred_list = [  
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            input_meta = img_metas[img_id]
                        
            radarOffset_pred_list = [radarOffsets[i][img_id].detach() for i in range(num_levels)]  
            radarDepthOffset_pred_list = [radarDepthOffsets[i][img_id].detach() for i in range(num_levels)]  
            radarClass_pred_list = [radarClasses[i][img_id].detach() for i in range(num_levels)]  
                       
            radarOffset_pred_list = torch.cat([prd.permute(1, 2, 0).reshape(-1, 2) for prd in radarOffset_pred_list], dim=0)
            radarDepthOffset_pred_list = torch.cat([prd.permute(1, 2, 0).reshape(-1, 1) for prd in radarDepthOffset_pred_list], dim=0)
            n_classes = radarClass_pred_list[0].shape[0]
            radarClass_pred_list = torch.cat([prd.permute(1, 2, 0).reshape(-1, n_classes) for prd in radarClass_pred_list], dim=0)
            radarClass_pred_list = radarClass_pred_list.sigmoid()
            
            num_points_per_lvl = [center.size(0) for center in mlvl_points]  
            concat_points = torch.cat(mlvl_points, dim=0)  
            x_r, y_r, d_r = radar_pts[0], radar_pts[1], radar_pts[2]            
            radar_data = radar_pts.T 
                        
            indices_p_all = [] 
            indices_r_all = []  
            indices_rd_lvl = []  
            idx_pts_start = 0
            for idx_level, pts in enumerate(mlvl_points):            
                indices_p, indices_r = self.sample_radar_projections(pts, x_r, y_r, d_r)
                indices_p = indices_p + idx_pts_start
                indices_p_all.append(indices_p)
                indices_r_all.append(indices_r)
                indices_rd_lvl.append(indices_p.new_full(indices_p.shape, idx_level))
                idx_pts_start += num_points_per_lvl[idx_level]
            
            indices_p_all = torch.cat(indices_p_all, dim=0)  
            indices_r_all = torch.cat(indices_r_all, dim=0) 
            indices_rd_lvl = torch.cat(indices_rd_lvl, dim=0)
            
            if len(indices_p_all)==0:
                det_bboxes = self._get_bboxes_single(  
                cls_score_list, bbox_pred_list, dir_cls_pred_list,
                attr_pred_list, centerness_pred_list, mlvl_points, input_meta,
                cfg, rescale) 
            else:
                radarOffset_pred_list = radarOffset_pred_list[indices_p_all]
                radarDepthOffset_pred_list = radarDepthOffset_pred_list[indices_p_all]
                radarClassScore_pred_list = radarClass_pred_list[indices_p_all]
                radar_pixel_list = concat_points[indices_p_all]  
                radar_depth_list = d_r[indices_r_all][:,None]   
                radar_data = radar_data[indices_r_all,:]  
                
                thres_rd_score = 0.03
                radarClassScore_max_list, radarClassIdx_pred_list = radarClassScore_pred_list.max(dim=1)
                indices1 = torch.arange(radarClassIdx_pred_list.shape[0], device = radarClassIdx_pred_list.device)
                msk_det_rd = radarClassScore_pred_list[indices1, radarClassIdx_pred_list] > thres_rd_score
                
                radarOffset_pred = radarOffset_pred_list[msk_det_rd]            
                radarDepthOffset_pred = radarDepthOffset_pred_list[msk_det_rd]  
                rd_cls_score_max = radarClassScore_max_list[msk_det_rd]       
                rd_cls_score_all = radarClassScore_pred_list[msk_det_rd]      
                rd_cls_idx_pred = radarClassIdx_pred_list[msk_det_rd]           
                radar_pixel_list = radar_pixel_list[msk_det_rd]
                radar_depth_list = radar_depth_list[msk_det_rd]
                indices_rd_lvl = indices_rd_lvl[msk_det_rd]
                radar_data = radar_data[msk_det_rd]   
                
                rd_obj_d_pred = radar_depth_list + radarDepthOffset_pred
                rd_obj_xy_pred = radar_pixel_list - radarOffset_pred
               
                det_bboxes = self._get_bboxes_fusion_single(  
                    cls_score_list, bbox_pred_list, dir_cls_pred_list,
                    attr_pred_list, centerness_pred_list, mlvl_points, 
                    input_meta, rd_obj_xy_pred, rd_obj_d_pred, rd_cls_score_max, 
                    rd_cls_score_all, rd_cls_idx_pred, indices_rd_lvl, radar_data,
                    cfg, rescale, model_mlp) 
            
            result_list.append(det_bboxes) 
        return result_list


    def _get_bboxes_fusion_single(self,
                           cls_scores,
                           bbox_preds,
                           dir_cls_preds,
                           attr_preds,
                           centernesses,
                           mlvl_points,
                           input_meta,
                           rd_obj_xy_pred, 
                           rd_obj_d_pred, 
                           rd_cls_score_max, 
                           rd_cls_score_all,
                           rd_cls_idx_pred,
                           indices_rd_lvl,
                           radar_data,
                           cfg,
                           rescale=False,
                           model_mlp = None):

        view = np.array(input_meta['cam2img']) 
        scale_factor = input_meta['scale_factor'] 
        cfg = self.test_cfg if cfg is None else cfg 
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points) 
        mlvl_centers2d = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_attr_scores = []
        mlvl_centerness = []
        indices_cam_lvl = []

        lvl = 0
        for cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, \
                points in zip(cls_scores, bbox_preds, dir_cls_preds,
                              attr_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape( 
                -1, self.cls_out_channels).sigmoid()
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2) 
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]  
            attr_pred = attr_pred.permute(1, 2, 0).reshape(-1, self.num_attrs)  
            attr_score = torch.max(attr_pred, dim=-1)[1]  
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid() 

            bbox_pred = bbox_pred.permute(1, 2,                
                                          0).reshape(-1,
                                                     sum(self.group_reg_dims))  
            bbox_pred = bbox_pred[:, :self.bbox_code_size]  
            nms_pre = cfg.get('nms_pre', -1)  
            if nms_pre > 0 and scores.shape[0] > nms_pre:  
                max_scores, _ = (scores * centerness[:, None]).max(dim=1) 
                _, topk_inds = max_scores.topk(nms_pre) 
                points = points[topk_inds, :]           
                bbox_pred = bbox_pred[topk_inds, :]     
                scores = scores[topk_inds, :]           
                dir_cls_pred = dir_cls_pred[topk_inds, :]  
                centerness = centerness[topk_inds]         
                dir_cls_score = dir_cls_score[topk_inds]   
                attr_score = attr_score[topk_inds]          

            bbox_pred[:, :2] = points - bbox_pred[:, :2]   
            if rescale: 
                bbox_pred[:, :2] /= bbox_pred[:, :2].new_tensor(scale_factor) 
            pred_center2d = bbox_pred[:, :3].clone()  

            mlvl_centers2d.append(pred_center2d)
            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_attr_scores.append(attr_score)
            mlvl_centerness.append(centerness)
            indices_cam_lvl.append(centerness.new_full(centerness.shape, lvl))
            lvl += 1

        mlvl_centers2d = torch.cat(mlvl_centers2d)   
        mlvl_bboxes = torch.cat(mlvl_bboxes)         
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)      
        indices_cam_lvl = torch.cat(indices_cam_lvl) 
        
        cam_obj_d_pred = mlvl_centers2d[:,2].clone()    
        cam_obj_xy_pred = mlvl_centers2d[:,:2].clone()  
        cam_obj_v_pred = mlvl_bboxes[:,7:9]             
        
        n_cam = mlvl_centers2d.shape[0]
        n_rd = rd_obj_xy_pred.shape[0]
        
        cam_obj_d_pred1 = cam_obj_d_pred[...,None].expand(n_cam, n_rd) 
        cam_obj_xy_pred1 = cam_obj_xy_pred[:,None,:].expand(n_cam, n_rd, 2)
         
        rd_obj_d_pred = rd_obj_d_pred.squeeze(1)
        rd_obj_d_pred1 = rd_obj_d_pred[None,...].expand(n_cam, n_rd)
        rd_obj_xy_pred1 = rd_obj_xy_pred[None,...].expand(n_cam, n_rd, 2)
 
        mlvl_scores1 = torch.cat(mlvl_scores) 
        mlvl_centerness1 = torch.cat(mlvl_centerness) 
        mlvl_nms_scores1 = mlvl_scores1 * mlvl_centerness1[:, None]         
        cam_cls_score_max, cam_cls_idx_pred = mlvl_nms_scores1.max(dim=1) 
        
        rd_cls_idx_pred1 = rd_cls_idx_pred[None,...].expand(n_cam, n_rd)
        cam_cls_idx_pred1 = cam_cls_idx_pred[...,None].expand(n_cam, n_rd)
        cam_cls_score_max1 = cam_cls_score_max[...,None].expand(n_cam, n_rd)
        
        d_diff_M = torch.abs(cam_obj_d_pred1 - rd_obj_d_pred1)
        xy_diff_M = cam_obj_xy_pred1 - rd_obj_xy_pred1
        xy_diff_M = (xy_diff_M[...,0]**2 + xy_diff_M[...,1]**2) ** 0.5
        idx_diff_M = rd_cls_idx_pred1 == cam_cls_idx_pred1
        
        cam_thres_msk = cam_cls_score_max1 > 0.02        
        thres_pixel = 25

        msk_s =  (rd_cls_idx_pred1==6) | (rd_cls_idx_pred1==5) 
        msk_m = (rd_cls_idx_pred1==0)| (rd_cls_idx_pred1==1) | (rd_cls_idx_pred1==2) | (rd_cls_idx_pred1==4)|\
                (rd_cls_idx_pred1==7)| (rd_cls_idx_pred1==8) | (rd_cls_idx_pred1==9) 
        msk_l = rd_cls_idx_pred1==3
        
        thres_diff_d = cam_obj_d_pred1 * 0.1
        
        thres_diff_d[msk_s] = torch.clamp(thres_diff_d[msk_s],1.5,4)
        thres_diff_d[msk_m] = torch.clamp(thres_diff_d[msk_m],3.5,8)
        thres_diff_d[msk_l] = torch.clamp(thres_diff_d[msk_l],3.5,8) 
        
        thres_pxiel_M = rd_cls_idx_pred1.new_full(rd_cls_idx_pred1.shape, thres_pixel)
        
        thres_pxiel_M[(rd_cls_idx_pred1==5) | (rd_cls_idx_pred1==6) | (rd_cls_idx_pred1==7)\
                      | (rd_cls_idx_pred1==8) | (rd_cls_idx_pred1==9)] = 15
        
        thres_pxiel_M[(rd_cls_idx_pred1==0) | (rd_cls_idx_pred1==1) | (rd_cls_idx_pred1==2) \
                      | (rd_cls_idx_pred1==4)] = 25
        
        thres_pxiel_M[ rd_cls_idx_pred1==3 ] = 25
        
        msk_c2r = (d_diff_M < thres_diff_d) & (xy_diff_M < thres_pxiel_M) & idx_diff_M & cam_thres_msk
        
        
        for i in range(n_cam):
            if msk_c2r[i].sum()>0:
                cls_idx_cam = cam_cls_idx_pred[i]  
                level_cam = indices_cam_lvl[i]    
                score_cam = cam_cls_score_max[i]  
                d_from_cam = cam_obj_d_pred[i]       
                xy_from_cam = cam_obj_xy_pred[i]     
                v_from_cam = cam_obj_v_pred[i]     

                msk_r = msk_c2r[i]
                
                level_rd_list = indices_rd_lvl[msk_r]    
                score_rd_list = rd_cls_score_max[msk_r] 
                d_from_rd_list = rd_obj_d_pred[msk_r]     
                xy_from_rd_list = rd_obj_xy_pred[msk_r]   
                v_from_radar_list = radar_data[msk_r, -3:-1] 
                rcs_rd_list = radar_data[msk_r, -1]       
                
                vx_cam, vz_cam = v_from_cam  
                vx_rd, vz_rd = v_from_radar_list[:,0], v_from_radar_list[:,1]  
                x_cam, y_cam = xy_from_cam  
                x_rd, y_rd = xy_from_rd_list[:,0], xy_from_rd_list[:,1]  
                
                v_prj = vx_cam * vx_rd + vz_cam * vz_rd 
                v_mag_rd = vx_rd ** 2 + vz_rd ** 2                       
                diff_pixel = ( (x_cam - x_rd)**2 + (y_cam - y_rd)**2 ) ** 0.5  
                diff_d = torch.abs(d_from_cam - d_from_rd_list) 
                 
                n_radar =len(d_from_rd_list)
                
                cls_idx_cam = cls_idx_cam.expand(n_radar, 1)
                level_cam = level_cam.expand(n_radar, 1)
                level_rd_list = level_rd_list[:,None]
                score_cam = score_cam.expand(n_radar, 1)
                score_rd_list = score_rd_list[:,None]
                d_from_cam = d_from_cam.expand(n_radar, 1)
                d_from_rd_list = d_from_rd_list[:,None]
                diff_d = diff_d[:,None]

                vx_cam = vx_cam.expand(n_radar, 1)
                vz_cam = vz_cam.expand(n_radar, 1)
                vx_rd = vx_rd[:,None]
                vz_rd = vz_rd[:,None]
                v_prj = v_prj[:,None]
                v_mag_rd = v_mag_rd[:,None]
                diff_pixel = diff_pixel[:,None]
                rcs_rd_list = rcs_rd_list[:,None]
                
                
                data_in = torch.cat([cls_idx_cam, level_cam, level_rd_list, score_cam, score_rd_list,\
                                    d_from_cam, d_from_rd_list, diff_d, vx_cam, vz_cam, vx_rd, vz_rd,\
                                    v_prj, v_mag_rd, diff_pixel, rcs_rd_list], dim=1)
                                                  
                out = model_mlp(data_in)                 
                out = out.squeeze(1).sigmoid()  
                
                thres_out = 0.26
                msk_out = out > thres_out                
                if msk_out.sum()>0:
                    d_pred = (out[msk_out] * rd_obj_d_pred[msk_r][msk_out]).sum() / out[msk_out].sum()
                else:
                    d_pred = cam_obj_d_pred[i]
                    
                mlvl_centers2d[i,2] = d_pred
                mlvl_bboxes[i,2] = d_pred
                               
        mlvl_bboxes[:, :3] = self.pts2Dto3D(mlvl_bboxes[:, :3], view) 
                                        
        if mlvl_bboxes.shape[0] > 0:
            dir_rot = limit_period(mlvl_bboxes[..., 6] - self.dir_offset, 0,
                                   np.pi)
            mlvl_bboxes[..., 6] = (                
                dir_rot + self.dir_offset +
                np.pi * mlvl_dir_scores.to(mlvl_bboxes.dtype))

        cam_intrinsic = mlvl_centers2d.new_zeros((4, 4))
        cam_intrinsic[:view.shape[0], :view.shape[1]] = \
            mlvl_centers2d.new_tensor(view)
        mlvl_bboxes[:, 6] = torch.atan2(  
            mlvl_centers2d[:, 0] - cam_intrinsic[0, 2],
            cam_intrinsic[0, 0]) + mlvl_bboxes[:, 6]
        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d']( 
            mlvl_bboxes, box_dim=self.bbox_code_size,
            origin=(0.5, 0.5, 0.5)).bev)

        mlvl_scores = torch.cat(mlvl_scores)  
        
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)  
        mlvl_attr_scores = torch.cat(mlvl_attr_scores) 
        mlvl_centerness = torch.cat(mlvl_centerness) 

        mlvl_nms_scores = mlvl_scores * mlvl_centerness[:, None] 
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms, 
                                       mlvl_nms_scores, cfg.score_thr,  
                                       cfg.max_per_img, cfg, mlvl_dir_scores,  
                                       mlvl_attr_scores)
        bboxes, scores, labels, dir_scores, attrs = results  
        attrs = attrs.to(labels.dtype) 
        bboxes = input_meta['box_type_3d']( 
            bboxes, box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5))
        if not self.pred_attrs:
            attrs = None

        return bboxes, scores, labels, attrs


    def create_mlp_data(self,
                        cls_scores,
                        bbox_preds,
                        dir_cls_preds,
                        attr_preds,
                        centernesses,
                        radarOffsets,
                        radarDepthOffsets,
                        radarClasses,
                        gt_bboxes,
                        gt_labels,
                        gt_bboxes_3d,
                        gt_labels_3d,
                        centers2d,
                        depths,
                        attr_labels,
                        img_metas,
                        radar_pts,    
                        gt_bboxes_ignore=None):
        
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]  
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,  
                                           bbox_preds[0].device) 
        
        num_levels = len(mlvl_points)   
        expanded_regress_ranges = [   
            mlvl_points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                mlvl_points[i]) for i in range(num_levels)
        ]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0) 
        concat_points = torch.cat(mlvl_points, dim=0) 

        num_points = [center.size(0) for center in mlvl_points]  

        _, _, labels_3d_list, bbox_targets_3d_list, centerness_targets_list, \
            attr_targets_list, min_dist_inds_list = multi_apply(                                                                  
                self._get_target_single,
                gt_bboxes,  
                gt_labels,
                gt_bboxes_3d,
                gt_labels_3d, 
                centers2d,
                depths,
                attr_labels,
                points=concat_points,  
                regress_ranges=concat_regress_ranges,
                num_points_per_lvl=num_points)
        
        bbox_targets_3d = bbox_targets_3d_list[0]   
          
        gt_cls = labels_3d_list[0].detach()          
        gt_obj_idx_cam = min_dist_inds_list[0].detach()   
        gt_depths = bbox_targets_3d[:,2].detach()   
        gt_offset_xy = bbox_targets_3d[:,:2].detach()   
        gt_xy = concat_points - gt_offset_xy
       
        labels_list, delta_offsets_list, delta_depths_list, radar_depths_list, \
        indices_p_all_list, msk_positive_list, inds_gt_boxes_list = multi_apply(   
                self._get_radar_target_single,
                gt_bboxes,  
                gt_labels,
                gt_bboxes_3d,
                centers2d,
                depths,
                radar_pts,
                points=concat_points, 
                regress_ranges=concat_regress_ranges,
                num_points_per_lvl=num_points)
               
        gt_obj_idx_rd = inds_gt_boxes_list[0].detach()  
        msk_positive = msk_positive_list[0].detach()      
        
        assert len(cls_scores) == len(bbox_preds) == len(dir_cls_preds) == \
            len(centernesses) == len(attr_preds) 
        num_levels = len(cls_scores) 
                
        if self.norm_on_bbox:
            for i in range(num_levels):
                bbox_preds[i][:, :2] *= self.strides[i]
                radarOffsets[i] *= self.strides[i]
        
        img_id = 0            
        cls_score_list = [  
            cls_scores[i][img_id].detach() for i in range(num_levels)
        ]
        bbox_pred_list = [   
            bbox_preds[i][img_id].detach() for i in range(num_levels)
        ]
        if self.use_direction_classifier: 
            dir_cls_pred_list = [   
                dir_cls_preds[i][img_id].detach()
                for i in range(num_levels)
            ]
        else:
            dir_cls_pred_list = [
                cls_scores[i][img_id].new_full(
                    [2, *cls_scores[i][img_id].shape[1:]], 0).detach()
                for i in range(num_levels)
            ]
        if self.pred_attrs: 
            attr_pred_list = [  
                attr_preds[i][img_id].detach() for i in range(num_levels)
            ]
        else:
            attr_pred_list = [ 
                cls_scores[i][img_id].new_full(
                    [self.num_attrs, *cls_scores[i][img_id].shape[1:]],
                    self.attr_background_label).detach()
                for i in range(num_levels)
            ]
        centerness_pred_list = [ 
            centernesses[i][img_id].detach() for i in range(num_levels)
        ]
        input_meta = img_metas[img_id]
        
        radarOffset_pred_list = [radarOffsets[i][img_id].detach() for i in range(num_levels)]   
        radarDepthOffset_pred_list = [radarDepthOffsets[i][img_id].detach() for i in range(num_levels)]  
        radarClass_pred_list = [radarClasses[i][img_id].detach() for i in range(num_levels)]  
                
        radarOffset_pred_list = torch.cat([prd.permute(1, 2, 0).reshape(-1, 2) for prd in radarOffset_pred_list], dim=0)
        radarDepthOffset_pred_list = torch.cat([prd.permute(1, 2, 0).reshape(-1, 1) for prd in radarDepthOffset_pred_list], dim=0)
        n_classes = radarClass_pred_list[0].shape[0]
        radarClass_pred_list = torch.cat([prd.permute(1, 2, 0).reshape(-1, n_classes) for prd in radarClass_pred_list], dim=0)
        radarClass_pred_list = radarClass_pred_list.sigmoid()
        
        num_points_per_lvl = [center.size(0) for center in mlvl_points] 
        concat_points = torch.cat(mlvl_points, dim=0)  
        x_r, y_r, d_r = radar_pts[img_id][0], radar_pts[img_id][1], radar_pts[img_id][2]        
        radar_data = radar_pts[img_id].T   
               
        indices_p_all = [] 
        indices_r_all = []  
        indices_rd_lvl = [] 
        idx_pts_start = 0
        for idx_level, pts in enumerate(mlvl_points):            
            indices_p, indices_r = self.sample_radar_projections(pts, x_r, y_r, d_r)
            indices_p = indices_p + idx_pts_start
            indices_p_all.append(indices_p)
            indices_r_all.append(indices_r)
            indices_rd_lvl.append(indices_p.new_full(indices_p.shape, idx_level))
            idx_pts_start += num_points_per_lvl[idx_level]
        
        indices_p_all = torch.cat(indices_p_all, dim=0)  
        indices_r_all = torch.cat(indices_r_all, dim=0) 
        indices_rd_lvl = torch.cat(indices_rd_lvl, dim=0)
               
        if len(indices_p_all)==0: 
            return []
            
        radarOffset_pred_list = radarOffset_pred_list[indices_p_all]
        radarDepthOffset_pred_list = radarDepthOffset_pred_list[indices_p_all]
        radarClassScore_pred_list = radarClass_pred_list[indices_p_all]
        
        radar_pixel_list = concat_points[indices_p_all]    
        radar_depth_list = d_r[indices_r_all][:,None]              
        radar_data = radar_data[indices_r_all,:] 
                
        gt_obj_idx_rd = inds_gt_boxes_list[0].detach()     
        msk_positive = msk_positive_list[0].detach()       
        n_rd_samples = radarDepthOffset_pred_list.shape[0]
        
        gt_obj_idx_rd_tmp = gt_obj_idx_rd.new_full((n_rd_samples,), -1)
        gt_obj_idx_rd_tmp[msk_positive] = gt_obj_idx_rd    
        gt_obj_idx_rd = gt_obj_idx_rd_tmp
        
        thres_rd_score = 0.03       
        radarClassScore_max_list, radarClassIdx_pred_list = radarClassScore_pred_list.max(dim=1)
        indices1 = torch.arange(radarClassIdx_pred_list.shape[0], device = radarClassIdx_pred_list.device)
        msk_det_rd = radarClassScore_pred_list[indices1, radarClassIdx_pred_list] > thres_rd_score
        
        radarOffset_pred = radarOffset_pred_list[msk_det_rd]            
        radarDepthOffset_pred = radarDepthOffset_pred_list[msk_det_rd]  
        rd_cls_score_pred = radarClassScore_max_list[msk_det_rd]        
        rd_cls_idx_pred = radarClassIdx_pred_list[msk_det_rd]          
        radar_pixel_list = radar_pixel_list[msk_det_rd]                 
        radar_depth_list = radar_depth_list[msk_det_rd]                 
        indices_rd_lvl = indices_rd_lvl[msk_det_rd]                     
                
        radar_data = radar_data[msk_det_rd]       
        gt_obj_idx_rd = gt_obj_idx_rd[msk_det_rd]                              
        rd_obj_d_pred = radar_depth_list + radarDepthOffset_pred
        rd_obj_xy_pred = radar_pixel_list - radarOffset_pred
        
        view = np.array(input_meta['cam2img'])  
        f = view[0,0]
        scale_factor = 1.0  
        cfg = None
        cfg = self.test_cfg if cfg is None else cfg  
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)  
        mlvl_centers2d = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_attr_scores = []
        mlvl_centerness = []
        
        mlvl_gt_cls = []
        mlvl_gt_obj_idx_cam = []
        mlvl_gt_depths = []
        mlvl_gt_xy = []
        indices_cam_lvl = []
        
        
        gt_cls_list = gt_cls.split(num_points, 0)     
        gt_obj_idx_cam_list = gt_obj_idx_cam.split(num_points, 0) 
        gt_depths_list = gt_depths.split(num_points, 0) 
        gt_xy_list = gt_xy.split(num_points,0)
        
        lvl = 0
        for cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, \
                points, gt_cls, gt_obj_idx_cam, gt_depths, gt_xy \
                        in zip(cls_score_list, bbox_pred_list, dir_cls_pred_list,
                              attr_pred_list, centerness_pred_list, mlvl_points, 
                              gt_cls_list, gt_obj_idx_cam_list, gt_depths_list, gt_xy_list):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape( 
                -1, self.cls_out_channels).sigmoid()
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)  
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]  
            attr_pred = attr_pred.permute(1, 2, 0).reshape(-1, self.num_attrs)  
            attr_score = torch.max(attr_pred, dim=-1)[1]  
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid() 

            bbox_pred = bbox_pred.permute(1, 2,                 
                                          0).reshape(-1,
                                                     sum(self.group_reg_dims))  
            bbox_pred = bbox_pred[:, :self.bbox_code_size] 
            nms_pre = cfg.get('nms_pre', -1)  
            if nms_pre > 0 and scores.shape[0] > nms_pre: 
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)  
                _, topk_inds = max_scores.topk(nms_pre) 
                points = points[topk_inds, :]           
                bbox_pred = bbox_pred[topk_inds, :]    
                scores = scores[topk_inds, :]          
                dir_cls_pred = dir_cls_pred[topk_inds, :]  
                centerness = centerness[topk_inds]         
                dir_cls_score = dir_cls_score[topk_inds]   
                attr_score = attr_score[topk_inds]         
                
                gt_cls = gt_cls[topk_inds]
                gt_obj_idx_cam = gt_obj_idx_cam[topk_inds]
                gt_depths = gt_depths[topk_inds]
                gt_xy = gt_xy[topk_inds]
                
            bbox_pred[:, :2] = points - bbox_pred[:, :2]   
            rescale = True
            if rescale: 
                bbox_pred[:, :2] /= bbox_pred[:, :2].new_tensor(scale_factor)  
            pred_center2d = bbox_pred[:, :3].clone()  
           
            mlvl_centers2d.append(pred_center2d)
            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_attr_scores.append(attr_score)
            mlvl_centerness.append(centerness)
            
            mlvl_gt_cls.append(gt_cls)  
            mlvl_gt_obj_idx_cam.append(gt_obj_idx_cam)  
            mlvl_gt_depths.append(gt_depths) 
            mlvl_gt_xy.append(gt_xy)            
            indices_cam_lvl.append(centerness.new_full(centerness.shape, lvl))            
            lvl += 1

        mlvl_centers2d = torch.cat(mlvl_centers2d)  
        mlvl_bboxes = torch.cat(mlvl_bboxes)         
        mlvl_dir_scores = torch.cat(mlvl_dir_scores) 
               
        mlvl_gt_cls = torch.cat(mlvl_gt_cls)    
        mlvl_gt_obj_idx_cam = torch.cat(mlvl_gt_obj_idx_cam)  
        mlvl_gt_depths = torch.cat(mlvl_gt_depths)    
        mlvl_gt_xy = torch.cat(mlvl_gt_xy)       
        indices_cam_lvl = torch.cat(indices_cam_lvl) 
        
        cam_obj_d_pred = mlvl_centers2d[:,2].clone()    
        cam_obj_xy_pred = mlvl_centers2d[:,:2].clone() 
        cam_obj_v_pred = mlvl_bboxes[:,7:9]            
        
        n_cam = mlvl_centers2d.shape[0]
        n_rd = rd_obj_xy_pred.shape[0]
        
        cam_obj_d_pred1 = cam_obj_d_pred[...,None].expand(n_cam, n_rd) 
        cam_obj_xy_pred1 = cam_obj_xy_pred[:,None,:].expand(n_cam, n_rd, 2)
        
        rd_obj_d_pred = rd_obj_d_pred.squeeze(1)
        rd_obj_d_pred1 = rd_obj_d_pred[None,...].expand(n_cam, n_rd)
        rd_obj_xy_pred1 = rd_obj_xy_pred[None,...].expand(n_cam, n_rd, 2)
       
        mlvl_scores1 = torch.cat(mlvl_scores) 
        mlvl_centerness1 = torch.cat(mlvl_centerness)  
        mlvl_nms_scores1 = mlvl_scores1 * mlvl_centerness1[:, None]          
        cam_cls_score_max, cam_cls_idx_pred = mlvl_nms_scores1.max(dim=1)  
        
        rd_cls_idx_pred1 = rd_cls_idx_pred[None,...].expand(n_cam, n_rd)
        cam_cls_idx_pred1 = cam_cls_idx_pred[...,None].expand(n_cam, n_rd)       
        cam_cls_score_max1 = cam_cls_score_max[...,None].expand(n_cam, n_rd)
        
        d_diff_M = torch.abs(cam_obj_d_pred1 - rd_obj_d_pred1)
        xy_diff_M = cam_obj_xy_pred1 - rd_obj_xy_pred1
        xy_diff_M = (xy_diff_M[...,0]**2 + xy_diff_M[...,1]**2) ** 0.5
        idx_diff_M = rd_cls_idx_pred1 == cam_cls_idx_pred1
        
        thres_diff_d = torch.clamp(cam_obj_d_pred1 * 0.1, 1.5, 8)        
        cam_thres_msk = cam_cls_score_max1 > 0.02   
        thres_pixel = 25
        
        msk_s =  (rd_cls_idx_pred1==6) | (rd_cls_idx_pred1==5) 
        msk_m = (rd_cls_idx_pred1==0)| (rd_cls_idx_pred1==1) | (rd_cls_idx_pred1==2) | (rd_cls_idx_pred1==4)|\
                (rd_cls_idx_pred1==7)| (rd_cls_idx_pred1==8) | (rd_cls_idx_pred1==9) 
        msk_l = rd_cls_idx_pred1==3
        
        thres_diff_d = cam_obj_d_pred1 * 0.1
        
        thres_diff_d[msk_s] = torch.clamp(thres_diff_d[msk_s],1.5,4)
        thres_diff_d[msk_m] = torch.clamp(thres_diff_d[msk_m],3.5,8)
        thres_diff_d[msk_l] = torch.clamp(thres_diff_d[msk_l],3.5,8)
        
        thres_pxiel_M = rd_cls_idx_pred1.new_full(rd_cls_idx_pred1.shape, thres_pixel)
        
        thres_pxiel_M[(rd_cls_idx_pred1==5) | (rd_cls_idx_pred1==6) | (rd_cls_idx_pred1==7)\
                      | (rd_cls_idx_pred1==8) | (rd_cls_idx_pred1==9)] = 15
        
        thres_pxiel_M[(rd_cls_idx_pred1==0) | (rd_cls_idx_pred1==1) | (rd_cls_idx_pred1==2) \
                      | (rd_cls_idx_pred1==4)] = 25
        
        thres_pxiel_M[ rd_cls_idx_pred1==3 ] = 25
        
        msk_c2r = (d_diff_M < thres_diff_d) & (xy_diff_M < thres_pxiel_M) & idx_diff_M & cam_thres_msk
                
        data_label = []   
        
        for i in range(n_cam):
            if msk_c2r[i].sum()>0 and mlvl_gt_cls[i]!=self.background_label:
               msk_r = msk_c2r[i]
               
               focal = f
               d_gt = mlvl_gt_depths[i].item()
               cls_idx_cam = cam_cls_idx_pred[i].item()  
               
               level_cam = indices_cam_lvl[i].item()   
               score_cam = cam_cls_score_max[i].item() 
               d_from_cam = cam_obj_d_pred[i].item()   
               xy_from_cam = cam_obj_xy_pred[i].cpu().numpy().tolist()
               v_from_cam = cam_obj_v_pred[i].cpu().numpy().tolist()   
               obj_idx_cam_gt = mlvl_gt_obj_idx_cam[i].item()
               
               level_rd_list = indices_rd_lvl[msk_r].cpu().numpy().tolist()   
               score_rd_list = rd_cls_score_pred[msk_r].cpu().numpy().tolist() 
               d_from_rd_list = rd_obj_d_pred[msk_r].cpu().numpy().tolist()    
               xy_from_rd_list = rd_obj_xy_pred[msk_r].cpu().numpy().tolist()     
               v_from_radar_list = radar_data[msk_r, -3:-1].cpu().numpy().tolist() 
               obj_idx_rd_gt_list = gt_obj_idx_rd[msk_r].cpu().numpy().tolist()  
               rcs_rd_list = radar_data[msk_r, -1].cpu().numpy().tolist()          
               
               n_mapped_rd = len(score_rd_list)               
               for k in range(n_mapped_rd):
                   data_one = np.array([focal, d_gt, cls_idx_cam, level_cam, score_cam, 
                                        d_from_cam, *xy_from_cam, *v_from_cam, obj_idx_cam_gt,
                                        level_rd_list[k], score_rd_list[k], d_from_rd_list[k], 
                                        *xy_from_rd_list[k], *v_from_radar_list[k], 
                                        obj_idx_rd_gt_list[k], rcs_rd_list[k]])

                   data_label.append(data_one.astype('float32'))
                                  
        return data_label


    @staticmethod
    def pts2Dto3D(points, view):
        """
        Args:
            points (torch.Tensor): points in 2D images, [N, 3], \
                3 corresponds with x, y in the image and depth.
            view (np.ndarray): camera instrinsic, [3, 3]

        Returns:
            torch.Tensor: points in 3D space. [N, 3], \
                3 corresponds with x, y, z in 3D space.
        """
        assert view.shape[0] <= 4
        assert view.shape[1] <= 4
        assert points.shape[1] == 3

        points2D = points[:, :2]
        depths = points[:, 2].view(-1, 1)
        unnorm_points2D = torch.cat([points2D * depths, depths], dim=1)

        viewpad = torch.eye(4, dtype=points2D.dtype, device=points2D.device)
        viewpad[:view.shape[0], :view.shape[1]] = points2D.new_tensor(view)
        inv_viewpad = torch.inverse(viewpad).transpose(0, 1)

        nbr_points = unnorm_points2D.shape[0]
        homo_points2D = torch.cat(
            [unnorm_points2D,
             points2D.new_ones((nbr_points, 1))], dim=1)
        points3D = torch.mm(homo_points2D, inv_viewpad)[:, :3]

        return points3D

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""   
        y, x = super()._get_points_single(featmap_size, stride, dtype, device) 
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride), 
                             dim=-1) + stride // 2
        return points
    
    
    def get_radar_targets(self, points, gt_bboxes_list, gt_labels_list,
                      gt_bboxes_3d_list, centers2d_list, depths_list, radar_pts):

          assert len(points) == len(self.regress_ranges)  
          num_levels = len(points)  
          
          expanded_regress_ranges = [   
              points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                  points[i]) for i in range(num_levels)
          ]
          
          concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)  
          concat_points = torch.cat(points, dim=0)        
          num_points = [center.size(0) for center in points]  
               
          labels_list, delta_offsets_list, delta_depths_list, radar_depths_list, \
          indices_p_all_list, msk_positive_list, _ = multi_apply(   
                  self._get_radar_target_single,
                  gt_bboxes_list,  
                  gt_labels_list,
                  gt_bboxes_3d_list,
                  centers2d_list,
                  depths_list,
                  radar_pts,
                  points=concat_points,  
                  regress_ranges=concat_regress_ranges,
                  num_points_per_lvl=num_points)
                    
          num_imgs = len(labels_list)
          
          labels = torch.cat(labels_list, dim=0)                 
          delta_offsets = torch.cat(delta_offsets_list, dim=0)  
          delta_depths = torch.cat(delta_depths_list, dim=0)    
          radar_depths = torch.cat(radar_depths_list, dim=0)    
          msk_positive = torch.cat(msk_positive_list, dim=0)            
        
          indices_p_all_list_tmp = []
          idx_start = 0
          for indices_p in indices_p_all_list:
              indices_p_all_list_tmp.append(indices_p + idx_start)
              idx_start += np.sum(num_points)
            
          indices_p_all = torch.cat(indices_p_all_list_tmp, dim=0)
            
          strides_list = []
          for _ in range(num_imgs):
              for stride, num_pts in zip(self.strides, num_points):
                  strides_list.append(delta_offsets.new_ones(num_pts)*stride)
                     
          strides = torch.cat(strides_list)     
          strides = strides[indices_p_all[msk_positive]]
          strides = strides[...,None]
           
          delta_offsets = delta_offsets / strides 
        
          return labels, delta_offsets, delta_depths, radar_depths, msk_positive, indices_p_all
                      
    
    def sample_radar_projections(self, pts, x_r, y_r, d_r):
               
        assert len(x_r)==len(y_r)==len(d_r)        
        if len(x_r) == 0:
            return pts.new_tensor([], dtype=torch.int64), pts.new_tensor([], dtype=torch.int64)
        
        pts = pts.clone()
        x_r, y_r = x_r.clone(), y_r.clone()
        n_p = len(pts)
        x_p, y_p = pts[:,0], pts[:,1]
        n_r = len(x_r)
        x_p, y_p = x_p[None,...].expand(n_r,n_p), y_p[None,...].expand(n_r,n_p)
        x_r, y_r = x_r[...,None].expand(n_r,n_p), y_r[...,None].expand(n_r,n_p)
        dis_r2p = ((x_p - x_r)**2 + (y_p - y_r)**2) ** 0.5 
        
        min_dists, indices_p = dis_r2p.min(dim=1) 
        
        indices_p1 = torch.unique(indices_p)
        n_sample = len(indices_p1)
        if len(indices_p1)<len(indices_p):  
            indices_r = indices_p1.new_zeros(indices_p1.shape)                
            for i in range(n_sample):
                if torch.sum(indices_p==indices_p1[i])==1: 
                    indices_r[i] = torch.nonzero(indices_p==indices_p1[i], as_tuple=False).squeeze()
                else:
                    overlap_radar_indices = torch.nonzero(indices_p==indices_p1[i], as_tuple=False).squeeze()                  
                    overlap_radar_depth = d_r[overlap_radar_indices]
                    min_depth, idx_min_rd = overlap_radar_depth.min(dim=0)
                    indices_r[i] = overlap_radar_indices[idx_min_rd]             
            indices_pixel = indices_p1   
        elif len(indices_p1) == len(indices_p):
            indices_r = torch.arange(n_sample, dtype=indices_p.dtype, device=indices_p.device) 
            indices_pixel = indices_p
                 
        return indices_pixel, indices_r        
    
    
    def _get_radar_target_single(self, gt_bboxes, gt_labels, gt_bboxes_3d,
                               centers2d, depths, radar_pts, points, regress_ranges, num_points_per_lvl):

        num_points = points.size(0) 
        num_gts = gt_labels.size(0) 
        
        if not isinstance(gt_bboxes_3d, torch.Tensor): 
            gt_bboxes_3d = gt_bboxes_3d.tensor.to(gt_bboxes.device) 
        
        pts0, pts1, pts2, pts3, pts4 = points.split(num_points_per_lvl, dim=0)         
        x_r, y_r, d_r, msk_r, idx_box_r = radar_pts[0], radar_pts[1], radar_pts[2], radar_pts[3], radar_pts[4]
        idx_box_r = idx_box_r.type(torch.int64)
        msk_r = msk_r.type(torch.bool)
        
        indices_p_all = []  
        indices_r_all = [] 
        idx_pts_start = 0
        for idx_level, pts in enumerate([pts0, pts1, pts2, pts3, pts4]):            
            indices_p, indices_r = self.sample_radar_projections(pts, x_r, y_r, d_r)
            indices_p = indices_p + idx_pts_start
            indices_p_all.append(indices_p)
            indices_r_all.append(indices_r)
            idx_pts_start += num_points_per_lvl[idx_level]
             
        indices_p_all = torch.cat(indices_p_all, dim=0)  
        indices_r_all = torch.cat(indices_r_all, dim=0)          
        n_pts_associated_with_rd = indices_p_all.shape[0]
        
        if num_gts == 0: 
            return gt_labels.new_full((n_pts_associated_with_rd,), self.background_label), \
                   centers2d.new_zeros((0, 2)), \
                   depths.new_zeros((0,)), \
                   d_r[indices_r_all], \
                   indices_p_all, \
                   gt_labels.new_zeros((n_pts_associated_with_rd,), dtype=bool), \
                   idx_box_r.new_zeros((0,))    
                       
        
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)  
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4) 
        centers2d = centers2d[None].expand(num_points, num_gts, 2) 
        gt_bboxes_3d = gt_bboxes_3d[None].expand(num_points, num_gts,  
                                                 self.bbox_code_size)
        depths = depths[None, :, None].expand(num_points, num_gts, 1)  
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)  
        ys = ys[:, None].expand(num_points, num_gts) 
        
                
        delta_xs = (xs - centers2d[..., 0])[..., None]  
        delta_ys = (ys - centers2d[..., 1])[..., None]
        bbox_targets_3d = torch.cat((delta_xs, delta_ys, depths), dim=-1)  

        left = xs - gt_bboxes[..., 0] 
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)  
  
        association_msk = msk_r[indices_r_all]            
        indices_gt_box = idx_box_r[indices_r_all]
        radar_depths = d_r[indices_r_all]
        
        max_regress_distance = bbox_targets.max(-1)[0]  
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))  
                
        msk_in_regress_range = inside_regress_range[indices_p_all[association_msk], indices_gt_box[association_msk]]   
        msk_positive = association_msk.clone() 
               
        msk_positive[msk_positive==True] = msk_positive[msk_positive==True] * msk_in_regress_range  
                                
        labels = gt_labels.new_zeros(indices_p_all.shape) 
        labels[msk_positive] = gt_labels[indices_gt_box[msk_positive]]
        labels[torch.logical_not(msk_positive)] = self.background_label
              
        bbox_targets_3d = bbox_targets_3d[ indices_p_all[msk_positive], indices_gt_box[msk_positive] ]          
        delta_depths = bbox_targets_3d[:,2] - radar_depths[msk_positive]  
        
        delta_offsets = bbox_targets_3d[:,0:2] 
        inds_gt_box_pos = indices_gt_box[msk_positive]
            
        return labels, delta_offsets, delta_depths, radar_depths, indices_p_all, msk_positive, inds_gt_box_pos

        
   
    def get_targets(self, points, gt_bboxes_list, gt_labels_list,
                    gt_bboxes_3d_list, gt_labels_3d_list, centers2d_list,
                    depths_list, attr_labels_list):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            gt_bboxes_3d_list (list[Tensor]): 3D Ground truth bboxes of each
                image, each has shape (num_gt, bbox_code_size).
            gt_labels_3d_list (list[Tensor]): 3D Ground truth labels of each
                box, each has shape (num_gt,).
            centers2d_list (list[Tensor]): Projected 3D centers onto 2D image,
                each has shape (num_gt, 2).
            depths_list (list[Tensor]): Depth of projected 3D centers onto 2D
                image, each has shape (num_gt, 1).
            attr_labels_list (list[Tensor]): Attribute labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)  
        num_levels = len(points)          
        expanded_regress_ranges = [   
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]       
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)  
        concat_points = torch.cat(points, dim=0)  

        num_points = [center.size(0) for center in points]  

        if attr_labels_list is None: 
            attr_labels_list = [
                gt_labels.new_full(gt_labels.shape, self.attr_background_label)
                for gt_labels in gt_labels_list
            ]
            
        _, _, labels_3d_list, bbox_targets_3d_list, centerness_targets_list, \
            attr_targets_list, _ = multi_apply(   
                self._get_target_single,
                gt_bboxes_list, 
                gt_labels_list,
                gt_bboxes_3d_list,
                gt_labels_3d_list,  
                centers2d_list,
                depths_list,
                attr_labels_list,
                points=concat_points, 
                regress_ranges=concat_regress_ranges,
                num_points_per_lvl=num_points)

        labels_3d_list = [
            labels_3d.split(num_points, 0) for labels_3d in labels_3d_list   
        ]   
        bbox_targets_3d_list = [
            bbox_targets_3d.split(num_points, 0)
            for bbox_targets_3d in bbox_targets_3d_list
        ]
        centerness_targets_list = [
            centerness_targets.split(num_points, 0)
            for centerness_targets in centerness_targets_list
        ]
        attr_targets_list = [
            attr_targets.split(num_points, 0)
            for attr_targets in attr_targets_list
        ]
        
        concat_lvl_labels_3d = []
        concat_lvl_bbox_targets_3d = []
        concat_lvl_centerness_targets = []
        concat_lvl_attr_targets = []
        
        for i in range(num_levels):
            concat_lvl_labels_3d.append(      
                torch.cat([labels[i] for labels in labels_3d_list]))  
            concat_lvl_centerness_targets.append(
                torch.cat([
                    centerness_targets[i]
                    for centerness_targets in centerness_targets_list
                ]))
            bbox_targets_3d = torch.cat([
                bbox_targets_3d[i] for bbox_targets_3d in bbox_targets_3d_list
            ])
            concat_lvl_attr_targets.append(
                torch.cat(
                    [attr_targets[i] for attr_targets in attr_targets_list]))
            if self.norm_on_bbox: 
                bbox_targets_3d[:, :
                                2] = bbox_targets_3d[:, :2] / self.strides[i]  
            concat_lvl_bbox_targets_3d.append(bbox_targets_3d)
            
        return concat_lvl_labels_3d, concat_lvl_bbox_targets_3d, \
            concat_lvl_centerness_targets, concat_lvl_attr_targets

    def _get_target_single(self, gt_bboxes, gt_labels, gt_bboxes_3d,
                           gt_labels_3d, centers2d, depths, attr_labels,
                           points, regress_ranges, num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)  
        num_gts = gt_labels.size(0) 
        if not isinstance(gt_bboxes_3d, torch.Tensor): 
            gt_bboxes_3d = gt_bboxes_3d.tensor.to(gt_bboxes.device) 
        if num_gts == 0:  
            return gt_labels.new_full((num_points,), self.background_label), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_labels_3d.new_full(
                       (num_points,), self.background_label), \
                   gt_bboxes_3d.new_zeros((num_points, self.bbox_code_size)), \
                   gt_bboxes_3d.new_zeros((num_points,)), \
                   attr_labels.new_full(
                       (num_points,), self.attr_background_label), \
                   gt_bboxes_3d.new_zeros((num_points,), dtype=torch.long)

        gt_bboxes_3d[..., 6] = -torch.atan2(
            gt_bboxes_3d[..., 0], gt_bboxes_3d[..., 2]) + gt_bboxes_3d[..., 6] 

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])           
        areas = areas[None].repeat(num_points, 1)   
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2) 
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)  
        centers2d = centers2d[None].expand(num_points, num_gts, 2)  
        gt_bboxes_3d = gt_bboxes_3d[None].expand(num_points, num_gts, 
                                                 self.bbox_code_size)
        depths = depths[None, :, None].expand(num_points, num_gts, 1)  
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)  
        ys = ys[:, None].expand(num_points, num_gts)  

        delta_xs = (xs - centers2d[..., 0])[..., None]  
        delta_ys = (ys - centers2d[..., 1])[..., None]
        bbox_targets_3d = torch.cat(
            (delta_xs, delta_ys, depths, gt_bboxes_3d[..., 3:]), dim=-1)  

        left = xs - gt_bboxes[..., 0]  
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1) 

        assert self.center_sampling is True, 'Setting center_sampling to '\
            'False has not been implemented for FCOS3D.'
        radius = self.center_sample_radius  
        center_xs = centers2d[..., 0]  
        center_ys = centers2d[..., 1]
        center_gts = torch.zeros_like(gt_bboxes)       
        stride = center_xs.new_zeros(center_xs.shape) 

        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius  
            lvl_begin = lvl_end

        center_gts[..., 0] = center_xs - stride   
        center_gts[..., 1] = center_ys - stride
        center_gts[..., 2] = center_xs + stride
        center_gts[..., 3] = center_ys + stride

        cb_dist_left = xs - center_gts[..., 0] 
        cb_dist_right = center_gts[..., 2] - xs
        cb_dist_top = ys - center_gts[..., 1]
        cb_dist_bottom = center_gts[..., 3] - ys
        center_bbox = torch.stack(
            (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)  
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0  

        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1])) 

        dists = torch.sqrt(torch.sum(bbox_targets_3d[..., :2]**2, dim=-1)) 
        dists[inside_gt_bbox_mask == 0] = INF 
        dists[inside_regress_range == 0] = INF
        min_dist, min_dist_inds = dists.min(dim=1) 

        labels = gt_labels[min_dist_inds]  
        labels_3d = gt_labels_3d[min_dist_inds]  
        attr_labels = attr_labels[min_dist_inds] 
        labels[min_dist == INF] = self.background_label 
        labels_3d[min_dist == INF] = self.background_label  
        attr_labels[min_dist == INF] = self.attr_background_label  

        bbox_targets = bbox_targets[range(num_points), min_dist_inds] 
        bbox_targets_3d = bbox_targets_3d[range(num_points), min_dist_inds]  
        relative_dists = torch.sqrt(
            torch.sum(bbox_targets_3d[..., :2]**2,
                      dim=-1)) / (1.414 * stride[:, 0]) 
        centerness_targets = torch.exp(-self.centerness_alpha * relative_dists) 

        return labels, bbox_targets, labels_3d, bbox_targets_3d, \
            centerness_targets, attr_labels, min_dist_inds    
    
