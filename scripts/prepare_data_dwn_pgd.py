import json
import sys
import argparse
import os
from os.path import join
import numpy as np
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from mmcv import Config
import _init_paths
from lib.my_data_parallel import MMDataParallel
from lib.my_dataloader import init_data_loader_from_file
from lib.fusion_dataset import NuScenesFusionDataset
from lib.my_model.radiant_pgd_network import PGDFusion3D

this_dir = os.path.dirname(__file__)
path_config = join(this_dir, '..', 'lib', 'configs_radiant_pgd.py')
cfg = Config.fromfile(path_config)  


def mkdir(dir1):
    if not os.path.exists(dir1): 
        os.makedirs(dir1)
        print('make directory %s' % dir1)
        
              
def init_env():    
    use_cuda = torch.cuda.is_available()    
    device0 = torch.device('cuda:0' if use_cuda else 'cpu')    
    cudnn.benchmark = True if use_cuda else False
    available_gpu_ids = [i for i in range(torch.cuda.device_count())]
    
    return device0, available_gpu_ids

        
def create_data(model, loader, task_name='generate data'):
    model.eval()
    data_all = []
    with torch.no_grad():  
        for data_batch in tqdm(loader, task_name):             
            data = model.create_data_step(data_batch)  
            data_all.extend(data)
    
    data_all = np.stack(data_all)        
    
    return data_all


def gen_train_data(model, ann_file, out_file_name):

    loader = init_data_loader_from_file(args, NuScenesFusionDataset, ann_file)
    data = create_data(model, loader, 'Generating %s' % out_file_name)
    
    data_dict =  {'data': data.tolist()}
    json.dump(data_dict, open(join(args.out_path, '%s.json' % out_file_name), 'w'))
    
        
def main(args):
                
    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = join(this_dir, '..', 'data', 'nuscenes')
    
    train_ann_file_mini = join(args.dir_data, 'fusion_data', 'nus_infos_train_mini.coco.json')
    val_ann_file_mini = join(args.dir_data, 'fusion_data', 'nus_infos_val_mini.coco.json')   
    
    train_ann_file =  join(args.dir_data, 'fusion_data', 'nus_infos_train.coco.json') 
    val_ann_file = join(args.dir_data, 'fusion_data', 'nus_infos_val.coco.json') 
    
    if not args.dir_checkpoint:
        args.dir_checkpoint = join(args.dir_data, 'fusion_data', 'train_result', 'radiant_pgd')
    
    args.out_path = join(args.dir_data, 'fusion_data', 'dwn_radiant_pgd')              
    mkdir(args.out_path) 
              
    device, available_gpu_ids = init_env()
    
    model = PGDFusion3D(**cfg.model_args)
    model.init_weights()   
    model = MMDataParallel(model.to(device), device_ids=available_gpu_ids)
    
    f_checkpoint = join(args.dir_checkpoint, 'checkpoint.tar')        
    if os.path.isfile(f_checkpoint):
        print('load model')
        checkpoint = torch.load(f_checkpoint)                    
        model.load_state_dict(checkpoint['state_dict'])
    else:
        sys.exit('checkpoint not found') 
        
    ann_files = dict(train_mini = train_ann_file_mini,
                     val_mini = val_ann_file_mini,
                     train = train_ann_file,
                     val = val_ann_file)    
    
    if args.select_data is not None:       
        assert args.select_data in ann_files
        gen_train_data(model, ann_files[args.select_data], args.select_data)       
    else:
        for out_file_name, ann_file in ann_files.items():
            gen_train_data(model, ann_file, out_file_name)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--dir_data', type=str)
    parser.add_argument('--dir_checkpoint', type=str)
    parser.add_argument('--select_data', type=str)  
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--samples_per_gpu', type=int, default=1) 
    parser.add_argument('--workers_per_gpu', type=int, default=2)  
   
    args = parser.parse_args()
    main(args)

