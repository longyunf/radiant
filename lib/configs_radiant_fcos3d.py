
optimizer_cfg = dict(
                    type='SGD',
                    lr=0.001,
                    momentum=0.9,
                    weight_decay=0.0001,
                    paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0))  
        
lr_config = dict(
            step=[8, 11],     
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
        pred_attrs=False,   
        pred_velo=True,
        dir_offset=0.7854, 
        strides=[8, 16, 32, 64, 128],
        group_reg_dims=(2, 1, 3, 1, 2), 
        cls_branch=(256, ),
        reg_branch=(
            (256, ),  
            (256, ), 
            (256, ),  
            (256, ),
            ()  
        ),
        dir_branch=(256, ),
        attr_branch=(256, ),
        centerness_branch=(64, ),
        radarOffset_branch = (256,),
        radarDepthOffset_branch = (256,),
        radarClass_branch = (256,),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
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
        dcn_on_last_conv=True),
    train_cfg=dict(
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.8,
        score_thr=0.05,
        min_bbox_size=0,
        max_per_img=200))
