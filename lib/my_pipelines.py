import os.path as osp
import numpy as np
from functools import reduce
from collections.abc import Sequence
from pyquaternion import Quaternion
from shapely.geometry import Point, MultiPoint
import torch
import mmcv
from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmcv.parallel import DataContainer as DC
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix


def label_radar_points(results):
        
    thres_dist = 1
    thres_v = 2
    thres_v_error = 0.2
    radar_pts = results['radar_pts']
    gt_bboxes_cam3d = results['gt_bboxes_3d']
       
    if len(gt_bboxes_cam3d.tensor) != 0:           
        vxf_list, vzf_list = gt_bboxes_cam3d.tensor[:,-2].numpy(), gt_bboxes_cam3d.tensor[:,-1].numpy()
        
        corners = gt_bboxes_cam3d.corners  
        n_box = corners.shape[0]
        corners2d = corners[:,[0,1,4,5],0::2]  
        
        polygon_list = []
        for i in range(n_box):
            x_corners, z_corners = corners2d[i,:,0], corners2d[i,:,1]      
            polygon = MultiPoint( [(x,z) for (x,z) in zip(x_corners, z_corners)] ).convex_hull
            polygon_list.append(polygon)
            
        x_list, z_list, vx_list, vz_list = radar_pts[0], radar_pts[1], radar_pts[4], radar_pts[5]
        
        gt_indices = []  
        obj_msk = []    
               
        n_pts = len(x_list)
        dist_mat = np.zeros((n_pts, n_box))
        v_error_mat = np.full((n_pts, n_box), -1.0)
        msk_moving = np.zeros(n_pts, dtype=bool)
        
        for i in range(n_pts):
            x,z,vx,vz = x_list[i], z_list[i], vx_list[i], vz_list[i]
            p1 = Point(x,z) 
            v_radar = (vx**2 + vz**2)**0.5   
            if v_radar >= thres_v:
                msk_moving[i]=True                
            for j in range(n_box):
                poly = polygon_list[j]
                dist_mat[i,j] = p1.distance(poly)
                if v_radar >= thres_v:                    
                    vxf, vzf = vxf_list[j], vzf_list[j]
                    v_error_mat[i,j] = abs( (vxf*vx + vzf*vz)/(vx**2 + vz**2) - 1 )
                    
        msk_v_match = np.logical_and(v_error_mat>=0, v_error_mat < thres_v_error)          
                       
        for i in range(n_pts):
            if msk_moving[i]:
                msk_valid = np.logical_and( dist_mat[i] < thres_dist, msk_v_match[i] )                
            else:
                msk_valid = dist_mat[i] < thres_dist
                    
            if np.any(msk_valid):
                    box_indices = np.arange(n_box)
                    idx_min = np.argmin(dist_mat[i][msk_valid])
                    idx_box = box_indices[msk_valid][idx_min]
                    gt_indices.append(idx_box)
                    obj_msk.append(True)
            else:
                gt_indices.append(-100)
                obj_msk.append(False)    
         
        gt_indices, obj_msk = np.array(gt_indices), np.array(obj_msk)
    else:  
        gt_indices=-100*np.ones(radar_pts.shape[1])
        obj_msk = np.zeros(radar_pts.shape[1], bool)
        
    radar_pts2 = np.row_stack( (radar_pts[[2,3,1],:], obj_msk, gt_indices) ) 
    radar_pts2 = np.row_stack((radar_pts2, radar_pts[[4,5,8],:]))  
    
    results['radar_pts'] = radar_pts2.astype('float32')  
    results['radar_pts_bev'] = np.row_stack( (radar_pts[[0,1,4,5,6,7],:], obj_msk, gt_indices) ) 
    
    return results


def proj2im(nusc, pc_cam, cam_token, min_z = 2):            
    cam_data = nusc.get('sample_data', cam_token) 
    cs_rec = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])         
    depth = pc_cam.points[2]    
    msk = pc_cam.points[2] >= min_z       
    points = view_points(pc_cam.points[:3, :], np.array(cs_rec['camera_intrinsic']), normalize=True)        
    x, y = points[0], points[1]
    msk =  reduce(np.logical_and, [x>0, x<1600, y>0, y<900, msk])        
    return x, y, depth, msk 


def cal_matrix_refSensor_from_car(nusc, sensor_token):    
    sensor_data = nusc.get('sample_data', sensor_token)    
    ref_cs_rec = nusc.get('calibrated_sensor', sensor_data['calibrated_sensor_token'])    
    ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)        
    return ref_from_car


def cal_matrix_refSensor_from_global(nusc, sensor_token):    
    sensor_data = nusc.get('sample_data', sensor_token)    
    ref_pose_rec = nusc.get('ego_pose', sensor_data['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', sensor_data['calibrated_sensor_token'])    
    ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)    
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)        
    M_ref_from_global = reduce(np.dot, [ref_from_car, car_from_global])    
    return M_ref_from_global


def cal_matrix_refSensor_to_global(nusc, sensor_token):    
    sensor_data = nusc.get('sample_data', sensor_token)       
    current_pose_rec = nusc.get('ego_pose', sensor_data['ego_pose_token'])
    global_from_car = transform_matrix(current_pose_rec['translation'],
                                       Quaternion(current_pose_rec['rotation']), inverse=False)
    current_cs_rec = nusc.get('calibrated_sensor', sensor_data['calibrated_sensor_token'])
    car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']), inverse=False)    
    M_ref_to_global = reduce(np.dot, [global_from_car, car_from_current])    
    return M_ref_to_global


def cal_trans_matrix(nusc, sensor1_token, sensor2_token):          
    M_ref_to_global = cal_matrix_refSensor_to_global(nusc, sensor1_token)    
    M_ref_from_global = cal_matrix_refSensor_from_global(nusc, sensor2_token)
    trans_matrix = reduce(np.dot, [M_ref_from_global, M_ref_to_global])   
    return trans_matrix


def load_multi_radar_to_cam(nusc, cam2radar_mappings, results):
    sample_token = results['img_info']['token']
    cam_token =  results['img_info']['id']   
    cam_channel = nusc.get('sample_data', cam_token)['channel']          
    radar_channels = cam2radar_mappings[cam_channel]   
    sample = nusc.get('sample', sample_token)         
    RadarPointCloud.disable_filters()
    n_dims = RadarPointCloud.nbr_dims()
    all_pc = RadarPointCloud(np.zeros((n_dims, 0)))
       
    for radar_channel in radar_channels:
        radar_token = sample['data'][radar_channel]
        radar_path = nusc.get_sample_data_path(radar_token)            
        pc = RadarPointCloud.from_file(radar_path)
        
        T_r2c = cal_trans_matrix(nusc, radar_token, cam_token)
        pc.transform(T_r2c)      
        R_r2c = T_r2c[:3,:3] 
        v0 = np.vstack(( pc.points[[6,7],:], np.zeros(pc.nbr_points()) ))  
        v0_comp = np.vstack(( pc.points[[8,9],:], np.zeros(pc.nbr_points()) )) 
        v1 = R_r2c.dot(v0)
        v1_comp = R_r2c.dot(v0_comp)
        
        pc.points[[6,7],:] = v1[[0,2],:]         
        pc.points[[8,9],:] = v1_comp[[0,2],:]                  
        all_pc.points = np.hstack((all_pc.points, pc.points))
            
    xz_cam = all_pc.points[[0,2],:]   
    v_raw = all_pc.points[[6,7],:]    
    v_comp = all_pc.points[[8,9],:]   
    rcs = all_pc.points[5,:]
              
    x_i, y_i, depth, msk = proj2im(nusc, all_pc, cam_token)
    x_i, y_i, depth = x_i[msk], y_i[msk], depth[msk]
     
    xz_cam, v_raw, v_comp = xz_cam[:,msk], v_raw[:,msk], v_comp[:,msk]  
    rcs = rcs[msk]
    xy_im = np.stack([x_i, y_i])     
    radar_pts = np.concatenate([xz_cam, xy_im, v_comp, v_raw], axis=0)  
       
    radar_pts = np.concatenate([radar_pts, rcs[None,:]], axis=0)  
           
    h_im, w_im = 900, 1600
    radar_map = np.zeros( (h_im, w_im, 10) , dtype=float) 
    
    x_i = np.clip(x_i, 0, w_im - 1)
    y_i = np.clip(y_i, 0, h_im - 1)
    
    x = xz_cam[0,:]
    assert np.array_equal(xz_cam[1,:], depth)
    
    vx, vz = v_raw[0,:], v_raw[1,:]
    v_amplitude = (vx**2 + vz**2)**0.5
    vx_comp, vz_comp = v_comp[0,:], v_comp[1,:]
    v_comp_amplitude = (vx_comp**2 + vz_comp**2)**0.5

    for i in range(len(x_i)):
        x_one, y_one = int(round( x_i[i] )), int(round( y_i[i] )) 
              
        if radar_map[y_one,x_one,0] == 0 or radar_map[y_one,x_one,0] > depth[i]:
            radar_map[y_one,x_one,:] = [x[i], depth[i], 1, vx[i], vz[i], v_amplitude[i], vx_comp[i], vz_comp[i], v_comp_amplitude[i], rcs[i]]  
            
    results['radar_map'] = radar_map.astype('float32')
    results['radar_pts'] = radar_pts.astype('float32')  
    
    return results


class LoadImageFromFile:
    """Load an image from file.
    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).
    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:  
            self.file_client = mmcv.FileClient(**self.file_client_args) 

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename) 
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type) 
        if self.to_float32:  
            img = img.astype(np.float32)

        results['filename'] = filename  
        results['ori_filename'] = results['img_info']['filename']  
        results['img'] = img  
        results['img_shape'] = img.shape   
        results['ori_shape'] = img.shape   
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in \
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam2img'] = results['img_info']['cam_intrinsic'] 
        return results
    

class LoadAnnotations:
    """Load multiple types of annotations.
    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
    """

    def __init__(self,
                 with_bbox=True,     
                 with_label=True):   
        self.with_bbox = with_bbox
        self.with_label = with_label

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)  
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        """Private function to load label annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:  
            results = self._load_bboxes(results)
            if results is None:  
                return None
        if self.with_label: 
            results = self._load_labels(results)
            
        return results

 
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.        
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.        
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
    """

    def __init__(self,
                 with_bbox_3d=True,         
                 with_label_3d=True,        
                 with_attr_label=True,     
                 with_bbox=True,           
                 with_label=True,          
                 with_bbox_depth=True):    
        super().__init__(
            with_bbox,
            with_label)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results
   
    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:  
            results = self._load_bboxes_3d(results)
            if results is None:  
                return None
        if self.with_bbox_depth: 
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d: 
            results = self._load_labels_3d(results)
        if self.with_attr_label: 
            results = self._load_attr_labels(results)

        return results


class Normalize:
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)   
        self.std = np.array(std, dtype=np.float32)     
        self.to_rgb = to_rgb  

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img']): 
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                            self.to_rgb)  
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str
    
    
class Pad:
    """Pad the image & masks & segmentation map.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (dict, optional): A dict for padding value, the default
            value is `dict(img=0, masks=0, seg=255)`.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=dict(img=0, masks=0, seg=255)):
        self.size = size 
        self.size_divisor = size_divisor  
        assert isinstance(pad_val, dict)
        self.pad_val = pad_val  

        assert size is not None or size_divisor is not None, \
            'only one of size and size_divisor should be valid'
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get('img', 0) 
        for key in results.get('img_fields', ['img']):  
            if self.size is not None:  
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=pad_val)
            elif self.size_divisor is not None:  
                padded_img = mmcv.impad_to_multiple(    
                    results[key], self.size_divisor, pad_val=pad_val)
            results[key] = padded_img   
        results['pad_shape'] = padded_img.shape   
        results['pad_fixed_size'] = self.size  
        results['pad_size_divisor'] = self.size_divisor  
        
        padded_radar_map = mmcv.impad_to_multiple(   
                    results['radar_map'], self.size_divisor, pad_val=pad_val)
        results['radar_map'] = padded_radar_map
        

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.
    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


class Collect3D(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
        data = {}  
        img_metas = {}  
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data
    
        
class DefaultFormatBundle(object):
    def __init__(self, ):
        return

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'img' in results:  
            if isinstance(results['img'], list):  
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['img'] = DC(to_tensor(imgs), stack=True)
            else: 
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))  
                results['img'] = DC(to_tensor(img), stack=True)
        
        if 'radar_map' in results:
            radar_map = results['radar_map'].transpose(2, 0, 1)   
            results['radar_map'] = DC(to_tensor(radar_map), stack=True)
        
        for key in ['gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                    'gt_labels_3d', 'attr_labels', 'centers2d', 'depths']:
            if key not in results: 
                continue
            if isinstance(results[key], list): 
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))
        if 'gt_bboxes_3d' in results:
            if isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):  
                results['gt_bboxes_3d'] = DC(
                    results['gt_bboxes_3d'], cpu_only=True)
            else:
                results['gt_bboxes_3d'] = DC(
                    to_tensor(results['gt_bboxes_3d']))
                
        if 'radar_pts' in results:
            results['radar_pts'] = DC(to_tensor(results['radar_pts']))
        
        return results

