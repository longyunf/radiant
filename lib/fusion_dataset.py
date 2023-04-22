from typing import Callable
import copy
import json
from collections import defaultdict
import time
import tqdm
import itertools
import numpy as np
import tempfile
from typing import Tuple
import os
from os import path as osp
import pyquaternion
import torch
from torch.utils.data import Dataset
import mmcv
from mmdet3d.core import bbox3d2result, box3d_multiclass_nms, xywhr2xyxyr
from mmdet3d.core.bbox import CameraInstance3DBoxes, get_box_type
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.eval.detection.evaluate import NuScenesEval       
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, DetectionMetricDataList, DetectionMetricData
from nuscenes.eval.detection.algo import calc_ap, calc_tp
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.common.loaders import load_prediction, add_center_dist, filter_eval_boxes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
from lib.my_pipelines import LoadImageFromFileMono3D, LoadAnnotations3D, Normalize, \
    Pad, DefaultFormatBundle, Collect3D, label_radar_points, load_multi_radar_to_cam  


class NuScenesFusionDataset(Dataset):
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    
    def __init__(self,
                 ann_file,                  
                 data_root,  
                 classes=None,              
                 modality=None,             
                 mode='train',              
                 box_type_3d='Camera',      
                 eval_version='detection_cvpr_2019',
                 version='v1.0-trainval',
                 filter_empty_gt=True):
        
        self.ann_file = ann_file
        self.data_root = data_root  
        self.img_prefix = self.data_root   
        self.CLASSES = self.get_classes(classes) 
        self.modality = modality 
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d) 
        self.eval_version = eval_version  
        self.mode = mode       
        self.version = version  
        self.filter_empty_gt = filter_empty_gt  
        self.nusc = NuScenes(self.version, dataroot = data_root, verbose=False)                
        self.bbox_code_size = 9 
        self.cam2radar_mappings = {'CAM_FRONT_LEFT': ['RADAR_FRONT_LEFT', 'RADAR_FRONT'],
                                   'CAM_FRONT': ['RADAR_FRONT'],
                                   'CAM_FRONT_RIGHT': ['RADAR_FRONT_RIGHT', 'RADAR_FRONT'],
                                   'CAM_BACK_LEFT': ['RADAR_FRONT_LEFT', 'RADAR_BACK_LEFT'],
                                   'CAM_BACK': ['RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'],
                                   'CAM_BACK_RIGHT': ['RADAR_FRONT_RIGHT', 'RADAR_BACK_RIGHT'] }
                             
        if self.eval_version is not None:
            from nuscenes.eval.detection.config import config_factory
            self.eval_detection_configs = config_factory(self.eval_version)
        
        if self.modality is None:
            self.modality = dict(
                use_camera=True,
                use_lidar=False,
                use_radar=True,
                use_map=False,
                use_external=False)
        
        self.data_infos = self.load_annotations(self.ann_file)

        if mode != 'test':
            valid_inds = self._filter_imgs()  
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            self._set_group_flag()

        self.load_im = LoadImageFromFileMono3D()
        
        self.normalize = Normalize(mean=[103.530, 116.280, 123.675],
                                   std=[1.0, 1.0, 1.0],
                                   to_rgb=False)
        
        self.pad = Pad(size_divisor=32)       
        self.format = DefaultFormatBundle()
                
        if mode != 'test':
            self.load_ann = LoadAnnotations3D()
            self.collect_3d = Collect3D(
                keys=['img', 'gt_bboxes', 'gt_labels', 'attr_labels', 
                      'gt_bboxes_3d', 'gt_labels_3d', 'centers2d', 'depths', 
                      'radar_map', 'radar_pts'])
        else:
            self.collect_3d = Collect3D(keys=['img', 'radar_map', 'radar_pts'])
                
      
    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)
        
        
    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.mode == 'test':  
            return self.prepare_test_img(idx)    
        while True:
            data = self.prepare_train_img(idx) 
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
     
     
    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)
     
     
     
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:  
                self.flag[i] = 1
      
                   
    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        
        Args:
            idx (int): Index of data.
        
        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)        
        self.pre_pipeline(results)        
        results = self.load_im(results)
        results = load_multi_radar_to_cam(self.nusc, self.cam2radar_mappings, results)
        
        results = self.load_ann(results)    
        results = label_radar_points(results)
        
        results = self.normalize(results)
        results = self.pad(results)
        results = self.format(results)
        results = self.collect_3d(results)
        
        return results
    
       
    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        results = self.load_im(results)
        results = load_multi_radar_to_cam(self.nusc, self.cam2radar_mappings, results)
        
        results['radar_pts'] = results['radar_pts'][[2,3,1,4,5,8],:]
        results = self.normalize(results)
        results = self.pad(results)
        results = self.format(results)
        results['scale_factor'] = 1.0
        results = self.collect_3d(results) 

        for key, val in results.items():
            results[key]=[val]
        
        return results
    
    
    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d
    
     
    def get_ann_info(self, idx):
        """Get COCO annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])       
        ann_info = self.coco.loadAnns(ann_ids)
        
        return self._parse_ann_info(self.data_infos[idx], ann_info)
    
    
    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths for training."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):  
            ids_in_cat |= set(self.coco.catToImgs[class_id])  
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds
    
        
    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox annotation.

        Args:
            img_info (list[dict]): Image info.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, labels, \
                gt_bboxes_3d, gt_labels_3d, attr_labels, centers2d, \
                depths, bboxes_ignore, masks, seg_map
        """
        gt_bboxes = []
        gt_labels = []
        attr_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_bboxes_cam3d = []
        centers2d = []
        depths = []
                     
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):  
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))  
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0)) 
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:  
                continue
            bbox = [x1, y1, x1 + w, y1 + h]  
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox) 
                gt_labels.append(self.cat2label[ann['category_id']])  
                attr_labels.append(ann['attribute_id'])
                gt_masks_ann.append(ann.get('segmentation', None))  
                bbox_cam3d = np.array(ann['bbox_cam3d']).reshape(1, -1) 
                velo_cam3d = np.array(ann['velo_cam3d']).reshape(1, 2)  
                nan_mask = np.isnan(velo_cam3d[:, 0])
                velo_cam3d[nan_mask] = [0.0, 0.0]
                
                bbox_cam3d = np.concatenate([bbox_cam3d, velo_cam3d], axis=-1) 
                gt_bboxes_cam3d.append(bbox_cam3d.squeeze()) 
               
                center2d = ann['center2d'][:2]
                depth = ann['center2d'][2]
                centers2d.append(center2d)
                depths.append(depth)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)    
            gt_labels = np.array(gt_labels, dtype=np.int64)     
            attr_labels = np.array(attr_labels, dtype=np.int64)  
        else: 
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            attr_labels = np.array([], dtype=np.int64)

        if gt_bboxes_cam3d:
            gt_bboxes_cam3d = np.array(gt_bboxes_cam3d, dtype=np.float32)  
            centers2d = np.array(centers2d, dtype=np.float32)              
            depths = np.array(depths, dtype=np.float32)                   
        else: 
            gt_bboxes_cam3d = np.zeros((0, self.bbox_code_size),
                                       dtype=np.float32)
            centers2d = np.zeros((0, 2), dtype=np.float32)
            depths = np.zeros((0), dtype=np.float32)
            
        gt_bboxes_cam3d = CameraInstance3DBoxes(
            gt_bboxes_cam3d,
            box_dim=gt_bboxes_cam3d.shape[-1],
            origin=(0.5, 0.5, 0.5))  
        
        gt_labels_3d = copy.deepcopy(gt_labels)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,                   
            labels=gt_labels,                   
            gt_bboxes_3d=gt_bboxes_cam3d,       
            gt_labels_3d=gt_labels_3d,          
            attr_labels=attr_labels,           
            centers2d=centers2d,                
            depths=depths,                     
            bboxes_ignore=gt_bboxes_ignore,    
            masks=gt_masks_ann,                 
            seg_map=seg_map)                     

        return ann
    
    
    def load_annotations(self, ann_file):   
        """Load annotation from COCO style annotation file.
        Args:
            ann_file (str): Path of annotation file.
        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)   
        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)   
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}  
        self.img_ids = self.coco.getImgIds() 
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.getAnnIds(imgIds=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos
    
    
    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.
        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names


    def evaluate(self,
                 results,
                 jsonfile_prefix=None,
                 result_names=['img_bbox']):
        
        result_files, tmp_dir = self.format_results_all_cams(results, jsonfile_prefix)
        
        if self.version == 'v1.0-test':
            return ''
        
        results_dict = dict()
        for name in result_names:
            print('Evaluating bboxes of {}'.format(name))
            ret_dict = self._evaluate_single_nus(result_files[name])  
        results_dict.update(ret_dict)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return results_dict
    
    
    def format_results_all_cams(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        result_files = dict()
        for name in results[0]:
            if '2d' in name:
                continue
            print(f'\nFormating bboxes of {name}')
            results_ = [out[name] for out in results]
            tmp_file_ = osp.join(jsonfile_prefix, name)
            result_files.update(
                {name: self._format_bbox_all_cams(results_, tmp_file_)})

        return result_files, tmp_dir
    
    
    def _format_bbox_all_cams(self, results, jsonfile_prefix=None):

        nusc_annos = {}
        mapped_class_names = self.CLASSES
        print('Start to convert detection format...')
        CAM_NUM = 6

        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):

            if sample_id % CAM_NUM == 0:
                boxes_per_frame = []

            annos = []
            boxes, attrs = output_to_nusc_box(det) 
            sample_token = self.data_infos[sample_id]['token']
            
            boxes = cam_nusc_box_to_global2(self.data_infos[sample_id],
                                            boxes,
                                            mapped_class_names,
                                            self.eval_detection_configs,
                                            self.eval_version)            
            boxes_per_frame.extend(boxes)

            if (sample_id + 1) % CAM_NUM != 0:
                continue
            boxes = global_nusc_box_to_cam(
                self.data_infos[sample_id + 1 - CAM_NUM], boxes_per_frame,
                mapped_class_names, self.eval_detection_configs,
                self.eval_version)
            cam_boxes3d, scores, labels = nusc_box_to_cam_box3d(boxes)

            nms_cfg = dict(
                use_rotate_nms=True,
                nms_across_levels=False,
                nms_pre=4096,
                nms_thr=0.05,
                score_thr=0.01,
                min_bbox_size=0,
                max_per_frame=500)
            from mmcv import Config
            nms_cfg = Config(nms_cfg)
            cam_boxes3d_for_nms = xywhr2xyxyr(cam_boxes3d.bev)
            boxes3d = cam_boxes3d.tensor

            boxes3d, scores, labels = box3d_multiclass_nms(
                boxes3d,
                cam_boxes3d_for_nms,
                scores,
                nms_cfg.score_thr,
                nms_cfg.max_per_frame,
                nms_cfg,
                mlvl_attr_scores=None)
            cam_boxes3d = CameraInstance3DBoxes(boxes3d, box_dim=9)
            det = bbox3d2result(cam_boxes3d, scores, labels, attrs=None)
            boxes, attrs = output_to_nusc_box(det)
            boxes = cam_nusc_box_to_global2(
                self.data_infos[sample_id + 1 - CAM_NUM], boxes,
                mapped_class_names, self.eval_detection_configs,
                self.eval_version)

            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name='')
                annos.append(nusc_anno)
            if sample_token in nusc_annos:
                nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos

        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path
    

    def _evaluate_single_nus(self,
                         result_path,
                         result_name='img_bbox'):

        output_dir = osp.join(*osp.split(result_path)[:-1])  
        nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {'v1.0-mini': 'mini_val',
                        'v1.0-trainval': 'val'}
                
        nusc_eval = NuScenesEval2(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)
        
        nusc_eval.main(render_curves=True)

        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail
    
    

class NuScenesEval2(NuScenesEval):
    
    def __init__(self,
         nusc: NuScenes,
         config: DetectionConfig,
         result_path: str,
         eval_set: str,
         output_dir: str = None,
         verbose: bool = True):
        
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)
                       
        self.gt_boxes = load_gt_all(self.nusc, self.eval_set, self.pred_boxes.sample_tokens, DetectionBox, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)  
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens
        
    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:  
                md = accumulate2(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list


      
def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.
            - attrs_3d (torch.Tensor, optional): Predicted attributes.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    attrs = None
    if 'attrs_3d' in detection:
        attrs = detection['attrs_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    box_dims[:, [0, 1, 2]] = box_dims[:, [2, 0, 1]]
    box_yaw = -box_yaw

    box_list = []
    for i in range(len(box3d)):
        q1 = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        q2 = pyquaternion.Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
        quat = q2 * q1
        velocity = (box3d.tensor[i, 7], 0.0, box3d.tensor[i, 8])
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list, attrs   
    

def cam_nusc_box_to_global2(info,
                           boxes,
                           classes,
                           eval_configs,
                           eval_version='detection_cvpr_2019'):
    """Convert the box from camera to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        box.rotate(pyquaternion.Quaternion(info['cam2ego_rotation']))
        box.translate(np.array(info['cam2ego_translation']))
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
        
    return box_list


def global_nusc_box_to_cam(info,
                           boxes,
                           classes,
                           eval_configs,
                           eval_version='detection_cvpr_2019'):
    """Convert the box from global to camera coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        box.translate(-np.array(info['ego2global_translation']))
        box.rotate(
            pyquaternion.Quaternion(info['ego2global_rotation']).inverse)
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        box.translate(-np.array(info['cam2ego_translation']))
        box.rotate(pyquaternion.Quaternion(info['cam2ego_rotation']).inverse)
        box_list.append(box)
    return box_list


def nusc_box_to_cam_box3d(boxes):
    """Convert boxes from :obj:`NuScenesBox` to :obj:`CameraInstance3DBoxes`.

    Args:
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.

    Returns:
        tuple (:obj:`CameraInstance3DBoxes` | torch.Tensor | torch.Tensor): \
            Converted 3D bounding boxes, scores and labels.
    """
    locs = torch.Tensor([b.center for b in boxes]).view(-1, 3)
    dims = torch.Tensor([b.wlh for b in boxes]).view(-1, 3)
    rots = torch.Tensor([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).view(-1, 1)
    velocity = torch.Tensor([b.velocity[:2] for b in boxes]).view(-1, 2)

    dims[:, [0, 1, 2]] = dims[:, [1, 2, 0]]
    rots = -rots

    boxes_3d = torch.cat([locs, dims, rots, velocity], dim=1).cuda()
    cam_boxes3d = CameraInstance3DBoxes(
        boxes_3d, box_dim=9, origin=(0.5, 0.5, 0.5))
    scores = torch.Tensor([b.score for b in boxes]).cuda()
    labels = torch.LongTensor([b.label for b in boxes]).cuda()
    nms_scores = scores.new_zeros(scores.shape[0], 10 + 1)
    indices = labels.new_tensor(list(range(scores.shape[0])))
    nms_scores[indices, labels] = scores
    return cam_boxes3d, nms_scores, labels


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def load_gt_all(nusc: NuScenes, eval_split: str, sample_tokens_eval, box_cls, verbose: bool = False) -> EvalBoxes:

    if box_cls == DetectionBox:
        attribute_map = {a['token']: a['name'] for a in nusc.attribute}

    if verbose:
        print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))
    sample_tokens_all = [s['token'] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    splits = create_splits_scenes()

    version = nusc.version
    if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split == 'test':
        assert version.endswith('test'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    else:
        raise ValueError('Error: Requested split {} which this function cannot map to the correct NuScenes version.'
                         .format(eval_split))

    if eval_split == 'test':
        assert len(nusc.sample_annotation) > 0, \
            'Error: You are trying to evaluate on the test set but you do not have the annotations!'

    sample_tokens = []
    for sample_token in sample_tokens_eval:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_record = nusc.get('scene', scene_token)
        if scene_record['name'] in splits[eval_split]:
            sample_tokens.append(sample_token)

    all_annotations = EvalBoxes()

    for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):
        
        sample = nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']
        
        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:

            sample_annotation = nusc.get('sample_annotation', sample_annotation_token)            
            if box_cls == DetectionBox:
                detection_name = category_to_detection_name(sample_annotation['category_name'])
                if detection_name is None:
                    continue

                attr_tokens = sample_annotation['attribute_tokens']
                attr_count = len(attr_tokens)
                if attr_count == 0:
                    attribute_name = ''
                elif attr_count == 1:
                    attribute_name = attribute_map[attr_tokens[0]]
                else:
                    raise Exception('Error: GT annotations must not have more than one attribute!')

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        detection_name=detection_name,
                        detection_score=-1.0,
                        attribute_name=attribute_name
                    )
                )
                
            else:
                raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

        all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations


def accumulate2(gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn: Callable,
               dist_th: float,
               verbose: bool = False) -> DetectionMetricData:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """

    used_sample_tokens = ([box.sample_token for box in pred_boxes.all])
    
    npos = len([1 for gt_box in gt_boxes.all if (gt_box.detection_name == class_name) and (gt_box.sample_token in used_sample_tokens) ])
    if verbose:
        print("Found {} GT of class {} out of {} total across {} samples.".
              format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    if npos == 0:
        return DetectionMetricData.no_predictions()

    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    tp = [] 
    fp = []
    conf = []

    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': []}

    taken = set()  
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))

            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]

            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box.detection_score)

        else:
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions()

    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem) 
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp


    for key in match_data.keys():
        if key == "conf":
            continue  
        else:
            tmp = cummean(np.array(match_data[key]))
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    return DetectionMetricData(recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err'])




class COCO():
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:  # True
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset: 
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann                  

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img                

        if 'categories' in self.dataset: 
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat              

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id']) 

        print('index created!')

        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
        
        
    def getCatIds(self, catNms=[], supNms=[], catIds=[]): 
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]
        return ids
    
    
    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)
    
    
    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]
        
    
    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]
         
        
    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists)) 
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

        