
optimizer_cfg = dict(
                    type='SGD',
                    lr=0.001,
                    momentum=0.9,
                    weight_decay=0.0001,
                    paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0))  
        
lr_config = dict(
            step=[6, 9],      
            gamma=0.1,
            warmup='linear',
            warmup_iters=500,
            warmup_ratio=1.0 / 3)  

model_args = dict(    
    pretrained_img='open-mmlab://detectron2/resnet101_caffe',  
    pretrained_other = None,
    backbone_img=dict(
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),  
        frozen_stages=1,            
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),       
    backbone_other=dict(
        depth=18,
        in_channels=10,  
        num_stages=4,
        out_indices=(0, 1, 2, 3),   
        frozen_stages=-1,          
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='caffe'),
    neck_img=dict(  
        in_channels=[256, 512, 1024, 2048],   
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,                           
        relu_before_extra_convs=True),
    neck_fusion=dict(  
        in_channels=[256, 512, 1024, 2048],    
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,                           
        relu_before_extra_convs=True),    
    bbox_head=dict(  
        num_classes=10,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        cls_feat_channels=256, 
        use_direction_classifier=True,
        diff_rad_by_sin=True,
        pred_attrs=True,
        pred_velo=True,
        pred_bbox2d=True,
        pred_keypoints=False,
        dir_offset=0.7854,
        strides=[8, 16, 32, 64, 128],
        group_reg_dims=(2, 1, 3, 1, 2, 4),
        cls_branch=(256, ),
        reg_branch=((256, ), (256, ), (256, ), (256, ), (), (256, )),
        dir_branch=(256, ),
        attr_branch=(256, ),
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
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        norm_on_bbox=True,
        centerness_on_reg=True,
        center_sampling=True,
        conv_bias=True,
        dcn_on_last_conv=True,
        use_depth_classifier=True,
        depth_branch=(256, ),
        depth_range=(0, 50),
        
        radarOffset_branch = (256,),
        radarDepthOffset_branch = (256,),
        radarClass_branch = (256,),
        
        depth_unit=10,
        division='uniform',
        depth_bins=6,
        bbox_coder=dict(
            type='PGDBBoxCoder',
            code_size=9,
            base_depths=((31.99, 21.12), (37.15, 24.63), (39.69, 23.97),  
                         (40.91, 26.34), (34.16, 20.11), (22.35, 13.7),
                         (24.28, 16.05), (27.26, 15.5), (20.61, 13.68),
                         (22.74, 15.01)),
            base_dims=((4.62, 1.73, 1.96), (6.93, 2.83, 2.51),      
                       (12.56, 3.89, 2.94), (11.22, 3.5, 2.95),
                       (6.68, 3.21, 2.85), (6.68, 3.21, 2.85),
                       (2.11, 1.46, 0.78), (0.73, 1.77, 0.67),
                       (0.41, 1.08, 0.41), (0.5, 0.99, 2.52))),
        loss_depth=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    train_cfg=dict(
        allowed_border=0,
        code_weight=[1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.8,
        score_thr=0.01,
        min_bbox_size=0,
        max_per_img=200))

