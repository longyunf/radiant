import argparse
import mmcv
import numpy as np
import os
from os import path as osp
from typing import List, Tuple, Union
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from mmdet3d.core.bbox.box_np_ops import points_cam2img


nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

NameMapping = {
                'movable_object.barrier': 'barrier',
                'vehicle.bicycle': 'bicycle',
                'vehicle.bus.bendy': 'bus',
                'vehicle.bus.rigid': 'bus',
                'vehicle.car': 'car',
                'vehicle.construction': 'construction_vehicle',
                'vehicle.motorcycle': 'motorcycle',
                'human.pedestrian.adult': 'pedestrian',
                'human.pedestrian.child': 'pedestrian',
                'human.pedestrian.construction_worker': 'pedestrian',
                'human.pedestrian.police_officer': 'pedestrian',
                'movable_object.trafficcone': 'traffic_cone',
                'vehicle.trailer': 'trailer',
                'vehicle.truck': 'truck'
            }


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various informations on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): flie name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict() 
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in NameMapping:
        return None
    cat_name = NameMapping[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def get_2d_boxes(nusc,
                 sample_data_token: str,
                 visibilities: List[str],
                 mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera \
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        box = nusc.get_box(ann_rec['token'])

        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        final_coords = post_process_coords(corner_coords)  

        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()    

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()

            if repro_rec['center2d'][2] <= 0:
                continue

            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name']
            attr_id = nus_attributes.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def _fill_test_infos_all_cams(nusc,
                              test_scenes,
                              mini=False):
   
    test_nusc_infos = []
    
    n_samples = len(nusc.sample)
    sample_indices = np.arange(n_samples)
    if mini:
        num_train_sample = 150     
        ct_train = 0
        
    for sample_idx in mmcv.track_iter_progress(sample_indices):    
        
        sample = nusc.sample[sample_idx]
        if sample['scene_token'] in test_scenes:
            
            if mini:
                if ct_train > num_train_sample:
                    break                            
                ct_train += 1
     
            info = {'cams': dict()}
            
            camera_types = [
                'CAM_FRONT',
                'CAM_FRONT_RIGHT',
                'CAM_FRONT_LEFT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT',
            ]
            for cam in camera_types:
                cam_token = sample['data'][cam]
                cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
                sd_rec = nusc.get('sample_data', cam_token)
                cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
                pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
                
                mmcv.check_file_exist(cam_path)
                
                cam_info = {
                    'data_path': cam_path,
                    'type': cam,
                    'sample_token': sample['token'],
                    'sample_data_token': sd_rec['token'],
                    'sensor2ego_translation': cs_record['translation'],
                    'sensor2ego_rotation': cs_record['rotation'],
                    'ego2global_translation': pose_record['translation'],
                    'ego2global_rotation': pose_record['rotation'],
                    'timestamp': sd_rec['timestamp'],
                    'cam_intrinsic': cam_intrinsic
                }
                
                info['cams'].update({cam: cam_info})
        
            test_nusc_infos.append(info)


    return test_nusc_infos


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
            if not mmcv.is_filepath(lidar_path):  
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def create_nuscenes_infos_all_cams(root_path,
                                   out_dir,
                                   info_prefix,
                                   version='v1.0-trainval'):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
    """
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits
    
    if version == 'v1.0-test':
        test_scenes = splits.test
    else:
        raise ValueError('unknown')

    available_scenes = get_available_scenes(nusc)   
       
    available_scene_names = [s['name'] for s in available_scenes]
    test_scenes = list(filter(lambda x: x in available_scene_names, test_scenes))    
    
    test_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']  
        for s in test_scenes
    ])
    
    print('test scene: {}'.format(len(test_scenes)))

    test_nusc_infos = _fill_test_infos_all_cams(nusc, test_scenes)
    test_nusc_infos_mini = _fill_test_infos_all_cams(nusc, test_scenes, mini=True)

    metadata = dict(version=version)
     
    print('test sample: {}'.format(len(test_nusc_infos)))
    data = dict(infos=test_nusc_infos, metadata=metadata)
    info_path = osp.join(out_dir, '{}_infos_test.pkl'.format(info_prefix))
    mmcv.dump(data, info_path)
    
    print('mini test sample: {}'.format(len(test_nusc_infos_mini)))
    data['infos'] = test_nusc_infos_mini
    info_path_mini = osp.join(out_dir, '{}_infos_test_mini.pkl'.format(info_prefix))
    mmcv.dump(data, info_path_mini)
    

def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool): Whether to export mono3d annotation. Default: True.
    """
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
      
    nusc_infos = mmcv.load(info_path)['infos']
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)  
        for cat_name in nus_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(nusc_infos):
         for cam in camera_types:
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(          
                                nusc,
                                cam_info['sample_data_token'],
                                visibilities=['', '1', '2', '3', '4'],
                                mono3d=mono3d)
            (height, width, _) = mmcv.imread(cam_info['data_path']).shape
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['data_path'].split('data/nuscenes/')
                    [-1],  
                    id=cam_info['sample_data_token'],
                    token=cam_info['sample_token'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=cam_info['ego2global_rotation'],
                    ego2global_translation=cam_info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    width=width,
                    height=height))
                  
            for coco_info in coco_infos:
                if coco_info is None:
                    continue

                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1

    json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       out_dir):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
    """    
    mmcv.mkdir_or_exist(out_dir)    
    create_nuscenes_infos_all_cams(root_path, out_dir, info_prefix, version=version)
    
    assert version == 'v1.0-test'
    info_test_mini_path = osp.join(out_dir, f'{info_prefix}_infos_test_mini.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
 
    export_2d_annotation(root_path, info_test_mini_path, version=version)  
    export_2d_annotation(root_path, info_test_path, version=version)       
        

parser = argparse.ArgumentParser()
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/nuscenes',
    help='specify the root path of dataset')  
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/nuscenes/fusion_data',
    required=False,
    help='name of info pkl')  
parser.add_argument('--extra-tag', type=str, default='nus')  
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()


if __name__ == '__main__':
    test_version = f'{args.version}-test'
    nuscenes_data_prep(
        root_path=args.root_path,    
        info_prefix=args.extra_tag,  
        version=test_version,       
        out_dir=args.out_dir)       

