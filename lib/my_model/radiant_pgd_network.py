import numpy as np
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import torch
from torch import nn as nn
from torch.nn import functional as F
from mmcv.runner import auto_fp16, force_fp32
from mmcv.cnn import bias_init_with_prob, normal_init, ConvModule, Scale, build_conv_layer, build_norm_layer 
from mmdet.core import multi_apply, bbox2result, distance2bbox
from mmdet.models.losses import GIoULoss
from mmdet3d.core import bbox3d2result, box3d_multiclass_nms, limit_period, xywhr2xyxyr
from lib.my_utils import points_cam2img, points_img2cam
from lib.my_model.base_module import BaseModule
from lib.my_model.resnet import ResNet
from lib.my_model.fpn import FPN
from lib.my_model.focal_loss import FocalLoss
from lib.my_model.smooth_l1_loss import SmoothL1Loss
from lib.my_model.cross_entropy_loss import CrossEntropyLoss
from lib.my_model.bbox_coder import PGDBBoxCoder

loss_registry = dict(FocalLoss=FocalLoss, SmoothL1Loss=SmoothL1Loss, 
                     CrossEntropyLoss=CrossEntropyLoss,
                     GIoULoss = GIoULoss)
bbox_coder_registry = dict(PGDBBoxCoder = PGDBBoxCoder)
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
        
        self.bbox_head = PGDFusionHead(**bbox_head)
        
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
                                              gt_labels, gt_bboxes_3d,
                                              gt_labels_3d, centers2d, depths,
                                              attr_labels, gt_bboxes_ignore,
                                              radar_pts)
        return losses
    
    

    def simple_test(self, img, img_metas, model_mlp, radar_map, radar_pts, rescale=False): 
 
        x_img, x_cat = self.extract_feat(img, radar_map[0])       
        outs = self.bbox_head(x_img, x_cat)  
        
        if self.eval_mono:
            bbox_outputs = self.bbox_head.get_bboxes(
                *outs[:-3], img_metas, rescale=rescale)      
        else:
            bbox_outputs = self.bbox_head.get_bboxes_fusion(
                *outs, img_metas, radar_pts[0][0], rescale=rescale, model_mlp=model_mlp)     

        if self.bbox_head.pred_bbox2d:
            bbox2d_img = [
                bbox2result(bboxes2d, labels, self.bbox_head.num_classes)
                for bboxes, scores, labels, attrs, bboxes2d in bbox_outputs
            ]
            bbox_outputs = [bbox_outputs[0][:-1]]

        bbox_img = [
            bbox3d2result(bboxes, scores, labels, attrs)
            for bboxes, scores, labels, attrs in bbox_outputs
        ]

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox
        if self.bbox_head.pred_bbox2d:
            for result_dict, img_bbox2d in zip(bbox_list, bbox2d_img):
                result_dict['img_bbox2d'] = img_bbox2d

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


class PGDFusion3D(SingleStageMono3DDetector):
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
        super(PGDFusion3D, self).__init__(backbone_img, 
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
    
    def build_bbox_coder(self, bbox_coder_cfg):
        cfg_ = bbox_coder_cfg.copy()
        coder_type = cfg_.pop('type')
        coder_cls = bbox_coder_registry.get(coder_type)
        coder = coder_cls(**cfg_)
        return coder   
          
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


class FCOSMono3DHead2(AnchorFreeMono3DHead2):

    def __init__(self,
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
                 bbox_coder=dict(type='FCOS3DBBoxCoder', code_size=9),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 centerness_branch=(64, ),
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
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_dir=loss_dir,
            loss_attr=loss_attr,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = self.build_loss(loss_centerness)
        bbox_coder['code_size'] = self.bbox_code_size
        self.bbox_coder = self.build_bbox_coder(bbox_coder)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness_prev = self._init_branch(
            conv_channels=self.centerness_branch,
            conv_strides=(1, ) * len(self.centerness_branch))
        self.conv_centerness = nn.Conv2d(self.centerness_branch[-1], 1, 1)
        self.scale_dim = 3 
        self.scales = nn.ModuleList([
            nn.ModuleList([Scale(1.0) for _ in range(self.scale_dim)])
            for _ in self.strides
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
                
    def init_weights(self):
        """Initialize weights of the head.
        We currently still use the customized init_weights because the default
        init of DCN triggered by the init_cfg will init conv_offset.weight,
        which mistakenly affects the training stability.
        """
        super().init_weights()
        for m in self.conv_centerness_prev:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        normal_init(self.conv_centerness, std=0.01)
        
        
    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)[:5]


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

        bbox_pred = self.bbox_coder.decode(bbox_pred, scale, stride,
                                           self.training, cls_score)
        
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

        return cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, \
            cls_feat, reg_feat, radarOffset, radarDepthOffset, radarClass


    @staticmethod
    def add_sin_difference(boxes1, boxes2):
        """Convert the rotation difference to difference in sine function.
        Args:
            boxes1 (torch.Tensor): Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2 (torch.Tensor): Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.
        Returns:
            tuple[torch.Tensor]: ``boxes1`` and ``boxes2`` whose 7th
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
                             dir_limit_offset=0.0,
                             num_bins=2,
                             one_hot=True):
        """Encode direction to 0 ~ num_bins-1.
        Args:
            reg_targets (torch.Tensor): Bbox regression targets.
            dir_offset (int, optional): Direction offset. Default to 0.
            dir_limit_offset (float, optional): Offset to set the direction
                range. Default to 0.0.
            num_bins (int, optional): Number of bins to divide 2*PI.
                Default to 2.
            one_hot (bool, optional): Whether to encode as one hot.
                Default to True.
        Returns:
            torch.Tensor: Encoded direction targets.
        """
        rot_gt = reg_targets[..., 6]
        offset_rot = limit_period(rot_gt - dir_offset, dir_limit_offset,
                                  2 * np.pi)
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


    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'dir_cls_preds', 'attr_preds',
                  'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   dir_cls_preds,
                   attr_preds,
                   centernesses,
                   img_metas,
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
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(dir_cls_preds) == \
            len(centernesses) == len(attr_preds)
        num_levels = len(cls_scores)

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
                predictions on a single scale level with shape
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

        cam2img = mlvl_centers2d.new_zeros((4, 4))
        cam2img[:view.shape[0], :view.shape[1]] = \
            mlvl_centers2d.new_tensor(view)
        mlvl_bboxes = self.bbox_coder.decode_yaw(mlvl_bboxes, mlvl_centers2d,
                                                 mlvl_dir_scores,
                                                 self.dir_offset, cam2img)

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

    @staticmethod
    def pts2Dto3D(points, view):
        """
        Args:
            points (torch.Tensor): points in 2D images, [N, 3],
                3 corresponds with x, y in the image and depth.
            view (np.ndarray): camera intrinsic, [3, 3]
        Returns:
            torch.Tensor: points in 3D space. [N, 3],
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
          indices_p_all_list, msk_positive_list, inds_gt_boxes_list = multi_apply(   
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
          inds_gt_boxes = torch.cat(inds_gt_boxes_list, dim=0) 
        
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
        
          return labels, delta_offsets, delta_depths, radar_depths, msk_positive, indices_p_all, inds_gt_boxes
                      
    
    def sample_radar_projections(self, pts, x_r, y_r, d_r):
        
        '''
        pts: sampling points in one level; (n_pts,2)
        xr,yr: radar projections on image
        dr: radar depth
        '''        
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
                concat_lvl_labels (list[Tensor]): Labels of each level.
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each
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



class PGDFusionHead(FCOSMono3DHead2):

    def __init__(self,
                 use_depth_classifier=True,
                 use_onlyreg_proj=False,
                 weight_dim=-1,
                 weight_branch=((256, ), ),
                 depth_branch=(64, ),
                 depth_range=(0, 70),
                 depth_unit=10,
                 division='uniform',
                 depth_bins=8,
                 loss_depth=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_bbox2d=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_consistency=dict(type='GIoULoss', loss_weight=1.0),
                 pred_bbox2d=True,
                 pred_keypoints=False,
                 bbox_coder=dict(
                     type='PGDBBoxCoder',
                     base_depths=((28.01, 16.32), ),
                     base_dims=((0.8, 1.73, 0.6), (1.76, 1.73, 0.6),
                                (3.9, 1.56, 1.6)),
                     code_size=7),
                 **kwargs):
        self.use_depth_classifier = use_depth_classifier  
        self.use_onlyreg_proj = use_onlyreg_proj  
        self.depth_branch = depth_branch  
        self.pred_keypoints = pred_keypoints  
        self.weight_dim = weight_dim 
        self.weight_branch = weight_branch  
        self.weight_out_channels = []
        for weight_branch_channels in weight_branch:
            if len(weight_branch_channels) > 0:
                self.weight_out_channels.append(weight_branch_channels[-1])
            else:
                self.weight_out_channels.append(-1)
        self.depth_range = depth_range  
        self.depth_unit = depth_unit 
        self.division = division 
        if self.division == 'uniform':
            self.num_depth_cls = int(
                (depth_range[1] - depth_range[0]) / depth_unit) + 1  
            if self.num_depth_cls != depth_bins:
                print('Warning: The number of bins computed from ' +
                      'depth_unit is different from given parameter! ' +
                      'Depth_unit will be considered with priority in ' +
                      'Uniform Division.')
        else:
            self.num_depth_cls = depth_bins
        super().__init__(
            pred_bbox2d=pred_bbox2d, bbox_coder=bbox_coder, **kwargs)  
        self.loss_depth = self.build_loss(loss_depth)
        if self.pred_bbox2d:  
            self.loss_bbox2d = self.build_loss(loss_bbox2d)
            self.loss_consistency = self.build_loss(loss_consistency)  
        if self.pred_keypoints:  
            self.kpts_start = 9 if self.pred_velo else 7

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        if self.pred_bbox2d:
            self.scale_dim += 1
        if self.pred_keypoints:
            self.scale_dim += 1
        self.scales = nn.ModuleList([
            nn.ModuleList([Scale(1.0) for _ in range(self.scale_dim)])  
            for _ in self.strides
        ])

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        super()._init_predictor()

        if self.use_depth_classifier:
            self.conv_depth_cls_prev = self._init_branch(
                conv_channels=self.depth_branch,
                conv_strides=(1, ) * len(self.depth_branch)) 
            self.conv_depth_cls = nn.Conv2d(self.depth_branch[-1],
                                            self.num_depth_cls, 1)
            self.fuse_lambda = nn.Parameter(torch.tensor(10e-5))

        if self.weight_dim != -1:
            self.conv_weight_prevs = nn.ModuleList()
            self.conv_weights = nn.ModuleList()
            for i in range(self.weight_dim):
                weight_branch_channels = self.weight_branch[i]
                weight_out_channel = self.weight_out_channels[i]
                if len(weight_branch_channels) > 0:
                    self.conv_weight_prevs.append(
                        self._init_branch(
                            conv_channels=weight_branch_channels,
                            conv_strides=(1, ) * len(weight_branch_channels)))
                    self.conv_weights.append(
                        nn.Conv2d(weight_out_channel, 1, 1))
                else:
                    self.conv_weight_prevs.append(None)
                    self.conv_weights.append(
                        nn.Conv2d(self.feat_channels, 1, 1))

    def init_weights(self):
        """Initialize weights of the head.
        We currently still use the customized defined init_weights because the
        default init of DCN triggered by the init_cfg will init
        conv_offset.weight, which mistakenly affects the training stability.
        """
        super().init_weights()

        bias_cls = bias_init_with_prob(0.01)
        if self.use_depth_classifier:
            for m in self.conv_depth_cls_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
            normal_init(self.conv_depth_cls, std=0.01, bias=bias_cls)

        if self.weight_dim != -1:
            for conv_weight_prev in self.conv_weight_prevs:
                if conv_weight_prev is None:
                    continue
                for m in conv_weight_prev:
                    if isinstance(m.conv, nn.Conv2d):
                        normal_init(m.conv, std=0.01)
            for conv_weight in self.conv_weights:
                normal_init(conv_weight, std=0.01)

    def forward(self, feats_img, feats_cat):

        return multi_apply(self.forward_single, feats_img, feats_cat,
                           self.scales, self.radar_scales, self.strides)


    def forward_single(self, x_img, x_cat, scale, radar_scale, stride):
        
        cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, \
        cls_feat, reg_feat, radarOffset, radarDepthOffset, radarClass \
            = super().forward_single(x_img, x_cat, scale, radar_scale, stride)


        max_regress_range = stride * self.regress_ranges[0][1] / \
            self.strides[0]                                           
        bbox_pred = self.bbox_coder.decode_2d(bbox_pred, scale, stride,
                                              max_regress_range, self.training,
                                              self.pred_keypoints,
                                              self.pred_bbox2d)

        depth_cls_pred = None
        if self.use_depth_classifier:
            clone_reg_feat = reg_feat.clone()
            for conv_depth_cls_prev_layer in self.conv_depth_cls_prev:
                clone_reg_feat = conv_depth_cls_prev_layer(clone_reg_feat)
            depth_cls_pred = self.conv_depth_cls(clone_reg_feat)

        weight = None
        if self.weight_dim != -1: 
            weight = []
            for i in range(self.weight_dim):
                clone_reg_feat = reg_feat.clone()
                if len(self.weight_branch[i]) > 0:
                    for conv_weight_prev_layer in self.conv_weight_prevs[i]:
                        clone_reg_feat = conv_weight_prev_layer(clone_reg_feat)
                weight.append(self.conv_weights[i](clone_reg_feat))
            weight = torch.cat(weight, dim=1)

        return cls_score, bbox_pred, dir_cls_pred, depth_cls_pred, weight, \
            attr_pred, centerness, radarOffset, radarDepthOffset, radarClass

    def get_proj_bbox2d(self,
                        bbox_preds,
                        pos_dir_cls_preds,
                        labels_3d,
                        bbox_targets_3d,
                        pos_points,
                        pos_inds,
                        img_metas,
                        pos_depth_cls_preds=None,
                        pos_weights=None,
                        pos_cls_scores=None,
                        with_kpts=False):
        """Decode box predictions and get projected 2D attributes.
        Args:
            bbox_preds (list[Tensor]): Box predictions for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * bbox_code_size.
            pos_dir_cls_preds (Tensor): Box scores for direction class
                predictions of positive boxes on all the scale levels in shape
                (num_pos_points, 2).
            labels_3d (list[Tensor]): 3D box category labels for each scale
                level, each is a 4D-tensor.
            bbox_targets_3d (list[Tensor]): 3D box targets for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * bbox_code_size.
            pos_points (Tensor): Foreground points.
            pos_inds (Tensor): Index of foreground points from flattened
                tensors.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            pos_depth_cls_preds (Tensor, optional): Probabilistic depth map of
                positive boxes on all the scale levels in shape
                (num_pos_points, self.num_depth_cls). Defaults to None.
            pos_weights (Tensor, optional): Location-aware weights of positive
                boxes in shape (num_pos_points, self.weight_dim). Defaults to
                None.
            pos_cls_scores (Tensor, optional): Classification scores of
                positive boxes in shape (num_pos_points, self.num_classes).
                Defaults to None.
            with_kpts (bool, optional): Whether to output keypoints targets.
                Defaults to False.
        Returns:
            tuple[Tensor]: Exterior 2D boxes from projected 3D boxes,
                predicted 2D boxes and keypoint targets (if necessary).
        """
        views = [np.array(img_meta['cam2img']) for img_meta in img_metas]
        num_imgs = len(img_metas)
        img_idx = []
        for label in labels_3d:
            for idx in range(num_imgs):
                img_idx.append(
                    labels_3d[0].new_ones(int(len(label) / num_imgs)) * idx)
        img_idx = torch.cat(img_idx)
        pos_img_idx = img_idx[pos_inds]

        flatten_strided_bbox_preds = []
        flatten_strided_bbox2d_preds = []
        flatten_bbox_targets_3d = []
        flatten_strides = []

        for stride_idx, bbox_pred in enumerate(bbox_preds):
            flatten_bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(
                -1, sum(self.group_reg_dims))
            flatten_bbox_pred[:, :2] *= self.strides[stride_idx]
            flatten_bbox_pred[:, -4:] *= self.strides[stride_idx]
            flatten_strided_bbox_preds.append(
                flatten_bbox_pred[:, :self.bbox_coder.bbox_code_size])
            flatten_strided_bbox2d_preds.append(flatten_bbox_pred[:, -4:])

            bbox_target_3d = bbox_targets_3d[stride_idx].clone()
            bbox_target_3d[:, :2] *= self.strides[stride_idx]
            bbox_target_3d[:, -4:] *= self.strides[stride_idx]
            flatten_bbox_targets_3d.append(bbox_target_3d)

            flatten_stride = flatten_bbox_pred.new_ones(
                *flatten_bbox_pred.shape[:-1], 1) * self.strides[stride_idx]
            flatten_strides.append(flatten_stride)

        flatten_strided_bbox_preds = torch.cat(flatten_strided_bbox_preds)
        flatten_strided_bbox2d_preds = torch.cat(flatten_strided_bbox2d_preds)
        flatten_bbox_targets_3d = torch.cat(flatten_bbox_targets_3d)
        flatten_strides = torch.cat(flatten_strides)
        pos_strided_bbox_preds = flatten_strided_bbox_preds[pos_inds]
        pos_strided_bbox2d_preds = flatten_strided_bbox2d_preds[pos_inds]
        pos_bbox_targets_3d = flatten_bbox_targets_3d[pos_inds]
        pos_strides = flatten_strides[pos_inds]

        pos_decoded_bbox2d_preds = distance2bbox(pos_points,
                                                 pos_strided_bbox2d_preds)

        pos_strided_bbox_preds[:, :2] = \
            pos_points - pos_strided_bbox_preds[:, :2]
        pos_bbox_targets_3d[:, :2] = \
            pos_points - pos_bbox_targets_3d[:, :2]

        if self.use_depth_classifier and (not self.use_onlyreg_proj):
            pos_prob_depth_preds = self.bbox_coder.decode_prob_depth(
                pos_depth_cls_preds, self.depth_range, self.depth_unit,
                self.division, self.num_depth_cls)
            sig_alpha = torch.sigmoid(self.fuse_lambda)
            pos_strided_bbox_preds[:, 2] = \
                sig_alpha * pos_strided_bbox_preds.clone()[:, 2] + \
                (1 - sig_alpha) * pos_prob_depth_preds

        box_corners_in_image = pos_strided_bbox_preds.new_zeros(
            (*pos_strided_bbox_preds.shape[:-1], 8, 2))
        box_corners_in_image_gt = pos_strided_bbox_preds.new_zeros(
            (*pos_strided_bbox_preds.shape[:-1], 8, 2))

        for idx in range(num_imgs):
            mask = (pos_img_idx == idx)
            if pos_strided_bbox_preds[mask].shape[0] == 0:
                continue
            cam2img = torch.eye(
                4,
                dtype=pos_strided_bbox_preds.dtype,
                device=pos_strided_bbox_preds.device)
            view_shape = views[idx].shape
            cam2img[:view_shape[0], :view_shape[1]] = \
                pos_strided_bbox_preds.new_tensor(views[idx])

            centers2d_preds = pos_strided_bbox_preds.clone()[mask, :2]
            centers2d_targets = pos_bbox_targets_3d.clone()[mask, :2]
            centers3d_targets = points_img2cam(pos_bbox_targets_3d[mask, :3],
                                               views[idx])

            pos_strided_bbox_preds[mask, :3] = points_img2cam(
                pos_strided_bbox_preds[mask, :3], views[idx])
            pos_bbox_targets_3d[mask, :3] = centers3d_targets

            pos_strided_bbox_preds[mask, 2] = \
                pos_bbox_targets_3d.clone()[mask, 2]

            if self.use_direction_classifier:
                pos_dir_cls_scores = torch.max(
                    pos_dir_cls_preds[mask], dim=-1)[1]
                pos_strided_bbox_preds[mask] = self.bbox_coder.decode_yaw(
                    pos_strided_bbox_preds[mask], centers2d_preds,
                    pos_dir_cls_scores, self.dir_offset, cam2img)
            pos_bbox_targets_3d[mask, 6] = torch.atan2(
                centers2d_targets[:, 0] - cam2img[0, 2],
                cam2img[0, 0]) + pos_bbox_targets_3d[mask, 6]

            corners = img_metas[0]['box_type_3d'](
                pos_strided_bbox_preds[mask],
                box_dim=self.bbox_coder.bbox_code_size,
                origin=(0.5, 0.5, 0.5)).corners
            box_corners_in_image[mask] = points_cam2img(corners, cam2img)

            corners_gt = img_metas[0]['box_type_3d'](
                pos_bbox_targets_3d[mask, :self.bbox_code_size],
                box_dim=self.bbox_coder.bbox_code_size,
                origin=(0.5, 0.5, 0.5)).corners
            box_corners_in_image_gt[mask] = points_cam2img(corners_gt, cam2img)

        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        proj_bbox2d_preds = torch.cat([minxy, maxxy], dim=1)

        outputs = (proj_bbox2d_preds, pos_decoded_bbox2d_preds)

        if with_kpts:
            norm_strides = pos_strides * self.regress_ranges[0][1] / \
                self.strides[0]
            kpts_targets = box_corners_in_image_gt - pos_points[..., None, :]
            kpts_targets = kpts_targets.view(
                (*pos_strided_bbox_preds.shape[:-1], 16))
            kpts_targets /= norm_strides

            outputs += (kpts_targets, )

        return outputs

    def get_pos_predictions(self, bbox_preds, dir_cls_preds, depth_cls_preds,
                            weights, attr_preds, centernesses, pos_inds,
                            img_metas):
        """Flatten predictions and get positive ones.
        Args:
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * bbox_code_size.
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * 2. (bin = 2)
            depth_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * self.num_depth_cls.
            attr_preds (list[Tensor]): Attribute scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_attrs.
            centernesses (list[Tensor]): Centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            pos_inds (Tensor): Index of foreground points from flattened
                tensors.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
        Returns:
            tuple[Tensor]: Box predictions, direction classes, probabilistic
                depth maps, location-aware weight maps, attributes and
                centerness predictions.
        """
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
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_dir_cls_preds = torch.cat(flatten_dir_cls_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_dir_cls_preds = flatten_dir_cls_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        pos_depth_cls_preds = None
        if self.use_depth_classifier:
            flatten_depth_cls_preds = [
                depth_cls_pred.permute(0, 2, 3,
                                       1).reshape(-1, self.num_depth_cls)
                for depth_cls_pred in depth_cls_preds
            ]
            flatten_depth_cls_preds = torch.cat(flatten_depth_cls_preds)
            pos_depth_cls_preds = flatten_depth_cls_preds[pos_inds]

        pos_weights = None
        if self.weight_dim != -1:
            flatten_weights = [
                weight.permute(0, 2, 3, 1).reshape(-1, self.weight_dim)
                for weight in weights
            ]
            flatten_weights = torch.cat(flatten_weights)
            pos_weights = flatten_weights[pos_inds]

        pos_attr_preds = None
        if self.pred_attrs:
            flatten_attr_preds = [
                attr_pred.permute(0, 2, 3, 1).reshape(-1, self.num_attrs)
                for attr_pred in attr_preds
            ]
            flatten_attr_preds = torch.cat(flatten_attr_preds)
            pos_attr_preds = flatten_attr_preds[pos_inds]

        return pos_bbox_preds, pos_dir_cls_preds, pos_depth_cls_preds, \
            pos_weights, pos_attr_preds, pos_centerness


    def loss(self,
             cls_scores,
             bbox_preds,
             dir_cls_preds,
             depth_cls_preds,
             weights,
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
        

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        
        radar_labels, radar_delta_offsets_tgt, radar_delta_depths_tgt, radar_depths_tgt, msk_positive, indices_pts, inds_gt_boxes_radar = \
        self.get_radar_targets(
            all_level_points, gt_bboxes, gt_labels, gt_bboxes_3d, centers2d, depths, radar_pts)
               
        loss_dict = dict()
        num_imgs = cls_scores[0].size(0)
                
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
        apply_to=('cls_scores', 'bbox_preds', 'dir_cls_preds',
                  'depth_cls_preds', 'weights', 'attr_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   dir_cls_preds,
                   depth_cls_preds,
                   weights,
                   attr_preds,
                   centernesses,
                   img_metas,
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
            depth_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * self.num_depth_cls.
            weights (list[Tensor]): Location-aware weights for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * self.weight_dim.
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config, optional): Test / postprocessing configuration,
                if None, test_cfg would be used. Defaults to None.
            rescale (bool, optional): If True, return boxes in original image
                space. Defaults to None.
        Returns:
            list[tuple[Tensor]]: Each item in result_list is a tuple, which
                consists of predicted 3D boxes, scores, labels, attributes and
                2D boxes (if necessary).
        """
        assert len(cls_scores) == len(bbox_preds) == len(dir_cls_preds) == \
            len(depth_cls_preds) == len(weights) == len(centernesses) == \
            len(attr_preds), 'The length of cls_scores, bbox_preds, ' \
            'dir_cls_preds, depth_cls_preds, weights, centernesses, and' \
            f'attr_preds: {len(cls_scores)}, {len(bbox_preds)}, ' \
            f'{len(dir_cls_preds)}, {len(depth_cls_preds)}, {len(weights)}' \
            f'{len(centernesses)}, {len(attr_preds)} are inconsistent.'
        num_levels = len(cls_scores)
        
        if self.norm_on_bbox:
            for i in range(num_levels):
                bbox_preds[i][:, :2] *= self.strides[i]
                bbox_preds[i][:, -4:] *= self.strides[i]
                
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
            if self.use_depth_classifier:
                depth_cls_pred_list = [
                    depth_cls_preds[i][img_id].detach()
                    for i in range(num_levels)
                ]
            else:
                depth_cls_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [self.num_depth_cls, *cls_scores[i][img_id].shape[1:]],
                        0).detach() for i in range(num_levels)
                ]
            if self.weight_dim != -1:
                weight_list = [
                    weights[i][img_id].detach() for i in range(num_levels)
                ]
            else:
                weight_list = [
                    cls_scores[i][img_id].new_full(
                        [1, *cls_scores[i][img_id].shape[1:]], 0).detach()
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
                depth_cls_pred_list, weight_list, attr_pred_list,
                centerness_pred_list, mlvl_points, input_meta, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           dir_cls_preds,
                           depth_cls_preds,
                           weights,
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
                predictions on a single scale level with shape
                (num_points * 2, H, W)
            depth_cls_preds (list[Tensor]): Box scores for probabilistic depth
                predictions on a single scale level with shape
                (num_points * self.num_depth_cls, H, W)
            weights (list[Tensor]): Location-aware weight maps on a single
                scale level with shape (num_points * self.weight_dim, H, W).
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 2).
            input_meta (dict): Metadata of input image.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool, optional): If True, return boxes in original image
                space. Defaults to False.
        Returns:
            tuples[Tensor]: Predicted 3D boxes, scores, labels, attributes and
                2D boxes (if necessary).
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
        mlvl_depth_cls_scores = []
        mlvl_depth_uncertainty = []
        mlvl_bboxes2d = None
        if self.pred_bbox2d: 
            mlvl_bboxes2d = []

        for cls_score, bbox_pred, dir_cls_pred, depth_cls_pred, weight, \
                attr_pred, centerness, points in zip(
                    cls_scores, bbox_preds, dir_cls_preds, depth_cls_preds,
                    weights, attr_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()     
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
            depth_cls_pred = depth_cls_pred.permute(1, 2, 0).reshape(
                -1, self.num_depth_cls)
            depth_cls_score = F.softmax(
                depth_cls_pred, dim=-1).topk(
                    k=2, dim=-1)[0].mean(dim=-1) 
            if self.weight_dim != -1:
                weight = weight.permute(1, 2, 0).reshape(-1, self.weight_dim)
            else: 
                weight = weight.permute(1, 2, 0).reshape(-1, 1)
            depth_uncertainty = torch.exp(-weight[:, -1])
            attr_pred = attr_pred.permute(1, 2, 0).reshape(-1, self.num_attrs)
            attr_score = torch.max(attr_pred, dim=-1)[1]
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1,
                                                     sum(self.group_reg_dims))
            bbox_pred3d = bbox_pred[:, :self.bbox_coder.bbox_code_size]  
            if self.pred_bbox2d:
                bbox_pred2d = bbox_pred[:, -4:]
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                merged_scores = scores * centerness[:, None]
                if self.use_depth_classifier:
                    merged_scores *= depth_cls_score[:, None]
                    if self.weight_dim != -1:
                        merged_scores *= depth_uncertainty[:, None]
                max_scores, _ = merged_scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred3d = bbox_pred3d[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_pred = dir_cls_pred[topk_inds, :]
                depth_cls_pred = depth_cls_pred[topk_inds, :]
                centerness = centerness[topk_inds]
                dir_cls_score = dir_cls_score[topk_inds]
                depth_cls_score = depth_cls_score[topk_inds]
                depth_uncertainty = depth_uncertainty[topk_inds]
                attr_score = attr_score[topk_inds]
                if self.pred_bbox2d:
                    bbox_pred2d = bbox_pred2d[topk_inds, :]
            
            bbox_pred3d[:, :2] = points - bbox_pred3d[:, :2]
            if rescale:
                bbox_pred3d[:, :2] /= bbox_pred3d[:, :2].new_tensor(
                    scale_factor)
                if self.pred_bbox2d:
                    bbox_pred2d /= bbox_pred2d.new_tensor(scale_factor)
            if self.use_depth_classifier:  
                prob_depth_pred = self.bbox_coder.decode_prob_depth(
                    depth_cls_pred, self.depth_range, self.depth_unit,
                    self.division, self.num_depth_cls)
                sig_alpha = torch.sigmoid(self.fuse_lambda) 
                bbox_pred3d[:, 2] = sig_alpha * bbox_pred3d[:, 2] + \
                    (1 - sig_alpha) * prob_depth_pred
            pred_center2d = bbox_pred3d[:, :3].clone()
            bbox_pred3d[:, :3] = points_img2cam(bbox_pred3d[:, :3], view)
            mlvl_centers2d.append(pred_center2d)
            mlvl_bboxes.append(bbox_pred3d)
            mlvl_scores.append(scores) 
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_depth_cls_scores.append(depth_cls_score)
            mlvl_attr_scores.append(attr_score)
            mlvl_centerness.append(centerness)
            mlvl_depth_uncertainty.append(depth_uncertainty)
            if self.pred_bbox2d:
                bbox_pred2d = distance2bbox(
                    points, bbox_pred2d, max_shape=input_meta['img_shape'])
                mlvl_bboxes2d.append(bbox_pred2d)

        mlvl_centers2d = torch.cat(mlvl_centers2d)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)
        if self.pred_bbox2d:
            mlvl_bboxes2d = torch.cat(mlvl_bboxes2d)

        cam2img = torch.eye(
            4, dtype=mlvl_centers2d.dtype, device=mlvl_centers2d.device)
        cam2img[:view.shape[0], :view.shape[1]] = \
            mlvl_centers2d.new_tensor(view)
        mlvl_bboxes = self.bbox_coder.decode_yaw(mlvl_bboxes, mlvl_centers2d,
                                                 mlvl_dir_scores,
                                                 self.dir_offset, cam2img)

        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes,
            box_dim=self.bbox_coder.bbox_code_size,
            origin=(0.5, 0.5, 0.5)).bev)

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_attr_scores = torch.cat(mlvl_attr_scores)
        mlvl_centerness = torch.cat(mlvl_centerness)

        mlvl_nms_scores = mlvl_scores * mlvl_centerness[:, None]
        if self.use_depth_classifier:  
            mlvl_depth_cls_scores = torch.cat(mlvl_depth_cls_scores)
            mlvl_nms_scores *= mlvl_depth_cls_scores[:, None]
            if self.weight_dim != -1:
                mlvl_depth_uncertainty = torch.cat(mlvl_depth_uncertainty)
                mlvl_nms_scores *= mlvl_depth_uncertainty[:, None]
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_nms_scores, cfg.score_thr,
                                       cfg.max_per_img, cfg, mlvl_dir_scores,
                                       mlvl_attr_scores, mlvl_bboxes2d)
        bboxes, scores, labels, dir_scores, attrs = results[0:5]
        attrs = attrs.to(labels.dtype)  
        bboxes = input_meta['box_type_3d'](
            bboxes,
            box_dim=self.bbox_coder.bbox_code_size,
            origin=(0.5, 0.5, 0.5))

        if not self.pred_attrs:
            attrs = None

        outputs = (bboxes, scores, labels, attrs)
        if self.pred_bbox2d:
            bboxes2d = results[-1]
            bboxes2d = torch.cat([bboxes2d, scores[:, None]], dim=1)
            outputs = outputs + (bboxes2d, )

        return outputs
    
    
    def get_bboxes_fusion(self,
                        cls_scores,
                        bbox_preds,
                        dir_cls_preds,
                        depth_cls_preds,
                        weights,
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
            len(depth_cls_preds) == len(weights) == len(centernesses) == \
            len(attr_preds)            
        num_levels = len(cls_scores)

        if self.norm_on_bbox:
            for i in range(num_levels):
                bbox_preds[i][:, :2] *= self.strides[i]
                bbox_preds[i][:, -4:] *= self.strides[i]
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
            if self.use_depth_classifier:
                depth_cls_pred_list = [
                    depth_cls_preds[i][img_id].detach()
                    for i in range(num_levels)
                ]
            else:
                depth_cls_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [self.num_depth_cls, *cls_scores[i][img_id].shape[1:]],
                        0).detach() for i in range(num_levels)
                ]
            if self.weight_dim != -1:
                weight_list = [
                    weights[i][img_id].detach() for i in range(num_levels)
                ]
            else:
                weight_list = [
                    cls_scores[i][img_id].new_full(
                        [1, *cls_scores[i][img_id].shape[1:]], 0).detach()
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
                depth_cls_pred_list, weight_list, attr_pred_list,
                centerness_pred_list, mlvl_points, input_meta, cfg, rescale) 
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
                    depth_cls_pred_list, weight_list, attr_pred_list,
                    centerness_pred_list, mlvl_points, input_meta, 
                    rd_obj_xy_pred, rd_obj_d_pred, rd_cls_score_max, 
                    rd_cls_score_all, rd_cls_idx_pred, indices_rd_lvl, radar_data,
                    cfg, rescale, model_mlp) 
            
            result_list.append(det_bboxes)
            
        return result_list
    
    
    def _get_bboxes_fusion_single(self,
                               cls_scores,  
                               bbox_preds,
                               dir_cls_preds,
                               depth_cls_preds,
                               weights,
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
        mlvl_depth_cls_scores = []
        mlvl_depth_uncertainty = []
        mlvl_bboxes2d = None
        if self.pred_bbox2d: 
            mlvl_bboxes2d = []
               
        indices_cam_lvl = []
        
        lvl = 0
        for cls_score, bbox_pred, dir_cls_pred, depth_cls_pred, weight, \
                attr_pred, centerness, points in zip(
                    cls_scores, bbox_preds, dir_cls_preds, depth_cls_preds,
                    weights, attr_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()     
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
            depth_cls_pred = depth_cls_pred.permute(1, 2, 0).reshape(
                -1, self.num_depth_cls)
            depth_cls_score = F.softmax(
                depth_cls_pred, dim=-1).topk(
                    k=2, dim=-1)[0].mean(dim=-1)  
            if self.weight_dim != -1:
                weight = weight.permute(1, 2, 0).reshape(-1, self.weight_dim)
            else:
                weight = weight.permute(1, 2, 0).reshape(-1, 1)
            depth_uncertainty = torch.exp(-weight[:, -1])
            attr_pred = attr_pred.permute(1, 2, 0).reshape(-1, self.num_attrs)
            attr_score = torch.max(attr_pred, dim=-1)[1]
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1,
                                                     sum(self.group_reg_dims))
            bbox_pred3d = bbox_pred[:, :self.bbox_coder.bbox_code_size]  
            if self.pred_bbox2d:
                bbox_pred2d = bbox_pred[:, -4:]
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                merged_scores = scores * centerness[:, None]
                if self.use_depth_classifier:
                    merged_scores *= depth_cls_score[:, None]
                    if self.weight_dim != -1:
                        merged_scores *= depth_uncertainty[:, None]
                max_scores, _ = merged_scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                
                points = points[topk_inds, :]
                bbox_pred3d = bbox_pred3d[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_pred = dir_cls_pred[topk_inds, :]
                depth_cls_pred = depth_cls_pred[topk_inds, :]
                centerness = centerness[topk_inds]
                dir_cls_score = dir_cls_score[topk_inds]
                depth_cls_score = depth_cls_score[topk_inds]
                depth_uncertainty = depth_uncertainty[topk_inds]
                attr_score = attr_score[topk_inds]
                if self.pred_bbox2d:
                    bbox_pred2d = bbox_pred2d[topk_inds, :]
                    
            bbox_pred3d[:, :2] = points - bbox_pred3d[:, :2]
            if rescale:
                bbox_pred3d[:, :2] /= bbox_pred3d[:, :2].new_tensor(
                    scale_factor)
                if self.pred_bbox2d:
                    bbox_pred2d /= bbox_pred2d.new_tensor(scale_factor)
            if self.use_depth_classifier: 
                prob_depth_pred = self.bbox_coder.decode_prob_depth(
                    depth_cls_pred, self.depth_range, self.depth_unit,
                    self.division, self.num_depth_cls)
                sig_alpha = torch.sigmoid(self.fuse_lambda)  
                bbox_pred3d[:, 2] = sig_alpha * bbox_pred3d[:, 2] + \
                    (1 - sig_alpha) * prob_depth_pred
           
            pred_center2d = bbox_pred3d[:, :3].clone()
            
            mlvl_centers2d.append(pred_center2d)
            mlvl_bboxes.append(bbox_pred3d)
            mlvl_scores.append(scores) 
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_depth_cls_scores.append(depth_cls_score)
            mlvl_attr_scores.append(attr_score)
            mlvl_centerness.append(centerness)
            mlvl_depth_uncertainty.append(depth_uncertainty)
            if self.pred_bbox2d:
                bbox_pred2d = distance2bbox(
                    points, bbox_pred2d, max_shape=input_meta['img_shape'])
                mlvl_bboxes2d.append(bbox_pred2d)
                
            indices_cam_lvl.append(centerness.new_full(centerness.shape, lvl))
            lvl += 1

        mlvl_centers2d = torch.cat(mlvl_centers2d)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)
        if self.pred_bbox2d:
            mlvl_bboxes2d = torch.cat(mlvl_bboxes2d)
            
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
        if self.use_depth_classifier:  
           mlvl_depth_cls_scores1 = torch.cat(mlvl_depth_cls_scores)  
           mlvl_nms_scores1 *= mlvl_depth_cls_scores1[:, None]
           if self.weight_dim != -1:
               mlvl_depth_uncertainty1 = torch.cat(mlvl_depth_uncertainty) 
               mlvl_nms_scores1 *= mlvl_depth_uncertainty1[:, None]

        cam_cls_score_max, cam_cls_idx_pred = mlvl_nms_scores1.max(dim=1) 
        rd_cls_idx_pred1 = rd_cls_idx_pred[None,...].expand(n_cam, n_rd)
        cam_cls_idx_pred1 = cam_cls_idx_pred[...,None].expand(n_cam, n_rd)
        cam_cls_score_max1 = cam_cls_score_max[...,None].expand(n_cam, n_rd)
        
        d_diff_M = torch.abs(cam_obj_d_pred1 - rd_obj_d_pred1)
        xy_diff_M = cam_obj_xy_pred1 - rd_obj_xy_pred1
        xy_diff_M = (xy_diff_M[...,0]**2 + xy_diff_M[...,1]**2) ** 0.5
        idx_diff_M = rd_cls_idx_pred1 == cam_cls_idx_pred1
        
        cam_thres_msk = cam_cls_score_max1 > 0.01      

        msk_s = (rd_cls_idx_pred1==5) | (rd_cls_idx_pred1==6) 
        msk_m = (rd_cls_idx_pred1==0) | (rd_cls_idx_pred1==1) | (rd_cls_idx_pred1==2) | (rd_cls_idx_pred1==4)|\
                (rd_cls_idx_pred1==7) | (rd_cls_idx_pred1==8) | (rd_cls_idx_pred1==9) 
        msk_l =  rd_cls_idx_pred1==3
        
        thres_diff_d = cam_obj_d_pred1 * 0.1
        thres_diff_d[msk_s] = torch.clamp(thres_diff_d[msk_s], 1.5, 4)
        thres_diff_d[msk_m] = torch.clamp(thres_diff_d[msk_m], 3.5, 8)
        thres_diff_d[msk_l] = torch.clamp(thres_diff_d[msk_l],3.5,8) 
        
        thres_pixel = 25
        thres_pxiel_M = rd_cls_idx_pred1.new_full(rd_cls_idx_pred1.shape, thres_pixel)
        
        thres_pxiel_M[  (rd_cls_idx_pred1==5) | (rd_cls_idx_pred1==6) | (rd_cls_idx_pred1==7)\
                      | (rd_cls_idx_pred1==8) | (rd_cls_idx_pred1==9) ] = 15
        
        thres_pxiel_M[  (rd_cls_idx_pred1==0) | (rd_cls_idx_pred1==1) | (rd_cls_idx_pred1==2) \
                      | (rd_cls_idx_pred1==4) ] = 25
         
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
                
                thres_out = 0.2   
                msk_out = out > thres_out
                
                if msk_out.sum()>0:
                    d_pred = (out[msk_out] * rd_obj_d_pred[msk_r][msk_out]).sum() / out[msk_out].sum()
                else:
                    d_pred = cam_obj_d_pred[i]
                    
                mlvl_centers2d[i,2] = d_pred
                mlvl_bboxes[i,2] = d_pred
      
        mlvl_bboxes[:, :3] = self.pts2Dto3D(mlvl_bboxes[:, :3], view) 
                        
        cam2img = torch.eye(
            4, dtype=mlvl_centers2d.dtype, device=mlvl_centers2d.device)
        cam2img[:view.shape[0], :view.shape[1]] = \
            mlvl_centers2d.new_tensor(view)
        mlvl_bboxes = self.bbox_coder.decode_yaw(mlvl_bboxes, mlvl_centers2d,
                                                 mlvl_dir_scores,
                                                 self.dir_offset, cam2img)

        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes,
            box_dim=self.bbox_coder.bbox_code_size,
            origin=(0.5, 0.5, 0.5)).bev)

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)

        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_attr_scores = torch.cat(mlvl_attr_scores)
        mlvl_centerness = torch.cat(mlvl_centerness)

        mlvl_nms_scores = mlvl_scores * mlvl_centerness[:, None]
        if self.use_depth_classifier:  
            mlvl_depth_cls_scores = torch.cat(mlvl_depth_cls_scores)
            mlvl_nms_scores *= mlvl_depth_cls_scores[:, None]
            if self.weight_dim != -1:
                mlvl_depth_uncertainty = torch.cat(mlvl_depth_uncertainty)
                mlvl_nms_scores *= mlvl_depth_uncertainty[:, None]
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_nms_scores, cfg.score_thr,
                                       cfg.max_per_img, cfg, mlvl_dir_scores,
                                       mlvl_attr_scores, mlvl_bboxes2d)
        bboxes, scores, labels, dir_scores, attrs = results[0:5]
        attrs = attrs.to(labels.dtype)  
        bboxes = input_meta['box_type_3d'](
            bboxes,
            box_dim=self.bbox_coder.bbox_code_size,
            origin=(0.5, 0.5, 0.5))

        if not self.pred_attrs:
            attrs = None

        outputs = (bboxes, scores, labels, attrs)
        if self.pred_bbox2d:
            bboxes2d = results[-1]
            bboxes2d = torch.cat([bboxes2d, scores[:, None]], dim=1)
            outputs = outputs + (bboxes2d, )

        return outputs 
    

    def create_mlp_data(self,
                        cls_scores,
                        bbox_preds,
                        dir_cls_preds,
                        depth_cls_preds,
                        weights,
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
                bbox_preds[i][:, -4:] *= self.strides[i]
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
            
        if self.use_depth_classifier:
            depth_cls_pred_list = [
                depth_cls_preds[i][img_id].detach()
                for i in range(num_levels)
            ]
        else:
            depth_cls_pred_list = [
                cls_scores[i][img_id].new_full(
                    [self.num_depth_cls, *cls_scores[i][img_id].shape[1:]],
                    0).detach() for i in range(num_levels)
            ]
        if self.weight_dim != -1:
            weight_list = [
                weights[i][img_id].detach() for i in range(num_levels)
            ]
        else:
            weight_list = [
                cls_scores[i][img_id].new_full(
                    [1, *cls_scores[i][img_id].shape[1:]], 0).detach()
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
        
        mlvl_depth_cls_scores = []
        mlvl_depth_uncertainty = []
        mlvl_bboxes2d = None
        if self.pred_bbox2d: 
            mlvl_bboxes2d = []
        
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
        for cls_score, bbox_pred, dir_cls_pred, depth_cls_pred, weight, attr_pred, centerness, \
                points, gt_cls, gt_obj_idx_cam, gt_depths, gt_xy \
                        in zip(cls_score_list, bbox_pred_list, dir_cls_pred_list,
                               depth_cls_pred_list, weight_list,
                              attr_pred_list, centerness_pred_list, mlvl_points, 
                              gt_cls_list, gt_obj_idx_cam_list, gt_depths_list, gt_xy_list):
            
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(  
                -1, self.cls_out_channels).sigmoid()
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)  
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]  
            
            depth_cls_pred = depth_cls_pred.permute(1, 2, 0).reshape(
                -1, self.num_depth_cls)
            depth_cls_score = F.softmax(
                depth_cls_pred, dim=-1).topk(
                    k=2, dim=-1)[0].mean(dim=-1) 
            if self.weight_dim != -1:
                weight = weight.permute(1, 2, 0).reshape(-1, self.weight_dim)
            else:
                weight = weight.permute(1, 2, 0).reshape(-1, 1)
            depth_uncertainty = torch.exp(-weight[:, -1])            
            
            attr_pred = attr_pred.permute(1, 2, 0).reshape(-1, self.num_attrs) 
            attr_score = torch.max(attr_pred, dim=-1)[1] 
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()  

            bbox_pred = bbox_pred.permute(1, 2,                
                                          0).reshape(-1,
                                                     sum(self.group_reg_dims))  
            
            bbox_pred3d = bbox_pred[:, :self.bbox_coder.bbox_code_size]  
            if self.pred_bbox2d:
                bbox_pred2d = bbox_pred[:, -4:]                                                     
                                                                                      
            nms_pre = cfg.get('nms_pre', -1) 
            if nms_pre > 0 and scores.shape[0] > nms_pre: 
                merged_scores = scores * centerness[:, None]
                if self.use_depth_classifier:
                    merged_scores *= depth_cls_score[:, None]
                    if self.weight_dim != -1:
                        merged_scores *= depth_uncertainty[:, None]
                max_scores, _ = merged_scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)                

                points = points[topk_inds, :]          
                bbox_pred3d = bbox_pred3d[topk_inds, :]
                scores = scores[topk_inds, :]           
                dir_cls_pred = dir_cls_pred[topk_inds, :]  
                depth_cls_pred = depth_cls_pred[topk_inds, :]
                centerness = centerness[topk_inds]         
                dir_cls_score = dir_cls_score[topk_inds]  
                depth_cls_score = depth_cls_score[topk_inds]
                depth_uncertainty = depth_uncertainty[topk_inds]                
                attr_score = attr_score[topk_inds]         
                if self.pred_bbox2d:
                    bbox_pred2d = bbox_pred2d[topk_inds, :]                
                
                gt_cls = gt_cls[topk_inds]
                gt_obj_idx_cam = gt_obj_idx_cam[topk_inds]
                gt_depths = gt_depths[topk_inds]
                gt_xy = gt_xy[topk_inds]             

            bbox_pred3d[:, :2] = points - bbox_pred3d[:, :2]
            rescale = True            
            if rescale:
                bbox_pred3d[:, :2] /= bbox_pred3d[:, :2].new_tensor(
                    scale_factor)
                if self.pred_bbox2d:
                    bbox_pred2d /= bbox_pred2d.new_tensor(scale_factor)
            if self.use_depth_classifier:  
                prob_depth_pred = self.bbox_coder.decode_prob_depth(
                    depth_cls_pred, self.depth_range, self.depth_unit,
                    self.division, self.num_depth_cls)
                sig_alpha = torch.sigmoid(self.fuse_lambda) 
                bbox_pred3d[:, 2] = sig_alpha * bbox_pred3d[:, 2] + \
                    (1 - sig_alpha) * prob_depth_pred
           
            pred_center2d = bbox_pred3d[:, :3].clone()
            
            mlvl_centers2d.append(pred_center2d)
            mlvl_bboxes.append(bbox_pred3d)
            mlvl_scores.append(scores)  
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_depth_cls_scores.append(depth_cls_score)
            mlvl_attr_scores.append(attr_score)
            mlvl_centerness.append(centerness)
            mlvl_depth_uncertainty.append(depth_uncertainty)
            if self.pred_bbox2d:
                bbox_pred2d = distance2bbox(
                    points, bbox_pred2d, max_shape=input_meta['img_shape'])
                mlvl_bboxes2d.append(bbox_pred2d)
            
            mlvl_gt_cls.append(gt_cls)  
            mlvl_gt_obj_idx_cam.append(gt_obj_idx_cam)  
            mlvl_gt_depths.append(gt_depths) 
            mlvl_gt_xy.append(gt_xy)                
            
            indices_cam_lvl.append(centerness.new_full(centerness.shape, lvl))
            lvl += 1
            
        mlvl_centers2d = torch.cat(mlvl_centers2d)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)
        if self.pred_bbox2d:
            mlvl_bboxes2d = torch.cat(mlvl_bboxes2d)

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
        if self.use_depth_classifier:  
           mlvl_depth_cls_scores1 = torch.cat(mlvl_depth_cls_scores)  
           mlvl_nms_scores1 *= mlvl_depth_cls_scores1[:, None]
           if self.weight_dim != -1:
               mlvl_depth_uncertainty1 = torch.cat(mlvl_depth_uncertainty)
               mlvl_nms_scores1 *= mlvl_depth_uncertainty1[:, None]

        cam_cls_score_max, cam_cls_idx_pred = mlvl_nms_scores1.max(dim=1)  
        rd_cls_idx_pred1 = rd_cls_idx_pred[None,...].expand(n_cam, n_rd)
        cam_cls_idx_pred1 = cam_cls_idx_pred[...,None].expand(n_cam, n_rd)
        cam_cls_score_max1 = cam_cls_score_max[...,None].expand(n_cam, n_rd)
        
        d_diff_M = torch.abs(cam_obj_d_pred1 - rd_obj_d_pred1)
        xy_diff_M = cam_obj_xy_pred1 - rd_obj_xy_pred1
        xy_diff_M = (xy_diff_M[...,0]**2 + xy_diff_M[...,1]**2) ** 0.5
        idx_diff_M = rd_cls_idx_pred1 == cam_cls_idx_pred1
        cam_thres_msk = cam_cls_score_max1 > 0.01
        
        msk_s = (rd_cls_idx_pred1==5) | (rd_cls_idx_pred1==6) 
        msk_m = (rd_cls_idx_pred1==0) | (rd_cls_idx_pred1==1) | (rd_cls_idx_pred1==2) | (rd_cls_idx_pred1==4)|\
                (rd_cls_idx_pred1==7) | (rd_cls_idx_pred1==8) | (rd_cls_idx_pred1==9) 
        msk_l =  rd_cls_idx_pred1==3
        
        thres_diff_d = cam_obj_d_pred1 * 0.1
        thres_diff_d[msk_s] = torch.clamp(thres_diff_d[msk_s], 1.5, 4)
        thres_diff_d[msk_m] = torch.clamp(thres_diff_d[msk_m], 3.5, 8)
        thres_diff_d[msk_l] = torch.clamp(thres_diff_d[msk_l], 3.5, 8)
        
        thres_pixel = 25
        thres_pxiel_M = rd_cls_idx_pred1.new_full(rd_cls_idx_pred1.shape, thres_pixel)
        
        thres_pxiel_M[  (rd_cls_idx_pred1==5) | (rd_cls_idx_pred1==6) | (rd_cls_idx_pred1==7)\
                      | (rd_cls_idx_pred1==8) | (rd_cls_idx_pred1==9) ] = 15
        
        thres_pxiel_M[  (rd_cls_idx_pred1==0) | (rd_cls_idx_pred1==1) | (rd_cls_idx_pred1==2) \
                      | (rd_cls_idx_pred1==4) ] = 25
        
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

        _, bbox_targets_list, labels_3d_list, bbox_targets_3d_list, \
            centerness_targets_list, attr_targets_list, _ = multi_apply(
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

        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
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
            if self.pred_bbox2d:
                bbox_targets = torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list])
                bbox_targets_3d = torch.cat([bbox_targets_3d, bbox_targets],
                                            dim=1)
            concat_lvl_attr_targets.append(
                torch.cat(
                    [attr_targets[i] for attr_targets in attr_targets_list]))
            if self.norm_on_bbox:
                bbox_targets_3d[:, :2] = \
                    bbox_targets_3d[:, :2] / self.strides[i]
                if self.pred_bbox2d:
                    bbox_targets_3d[:, -4:] = \
                        bbox_targets_3d[:, -4:] / self.strides[i]
            concat_lvl_bbox_targets_3d.append(bbox_targets_3d)
        return concat_lvl_labels_3d, concat_lvl_bbox_targets_3d, \
            concat_lvl_centerness_targets, concat_lvl_attr_targets


