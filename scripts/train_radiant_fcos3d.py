import argparse
import os
from os.path import join
from timeit import default_timer as timer
import copy
import torch
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import mmcv
from mmcv import Config
from mmcv.runner import _load_checkpoint, load_state_dict
import re
import _init_paths
from lib.my_optimizer import build_optimizer, StepLrUpdater
from lib.my_data_parallel import MMDataParallel
from lib.my_dataloader import init_data_loader
from lib.fusion_dataset import NuScenesFusionDataset
from lib.my_model.radiant_fcos3d_network import FCOSFusion3D
from scripts.train_dwn import FusionMLP 

this_dir = os.path.dirname(__file__)
path_config = join(this_dir, '..', 'lib', 'configs_radiant_fcos3d.py')
cfg = Config.fromfile(path_config)  


def freeze_subnet(model, net_names): 
    for child_name, child in model.module.named_children():
        if child_name in net_names:
            child.eval()   
            for param_name, param in child.named_parameters():
                param.requires_grad = False  
                print('%s.%s was frozen.' % (child_name, param_name) )
            print('%s was frozen.' % child_name )   


def freeze_cam_heads(model):
    for child_name, child in model.module.named_children():
        if child_name == 'bbox_head':
            for module_name, m in child.named_children():
                if 'radar' not in module_name:
                    m.eval()
                    for param_name, param in m.named_parameters():
                        param.requires_grad = False  
                        print('%s.%s.%s was frozen.' % (child_name, module_name, param_name) )
                else:
                    print('%s.%s to be trained' % (child_name, module_name) )
                    

def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None,
                    revise_keys=[(r'^module\.', '')]):
    """Load checkpoint from a file or URI.
    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Default: strip
            the prefix 'module.' by [(r'^module\\.', '')].
    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location, logger) 

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items()})

    state_dict._metadata = metadata

    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def clip_grads(params, grad_clip=dict(max_norm=35, norm_type=2)):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **grad_clip)


def single_gpu_test(model, model_mlp, data_loader):
    """Test model with single gpu.
    
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        
    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    model_mlp.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))   
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(model_mlp=model_mlp, return_loss=False, rescale=True, **data)

        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results



def train(args, model, train_loader, optimizer, lr_updater, epoch, iter_idx):
    
    model.train() 
    freeze_subnet(model, ['backbone_img', 'neck_img'])
    freeze_cam_heads(model)
        
    ave_loss=0
    
    for batch_idx, data_batch in enumerate(train_loader):  
                
        lr_updater.before_train_iter(optimizer, iter_idx)  
        optimizer.zero_grad() 
        outputs = model.train_step(data_batch, optimizer=None) 
        loss = outputs['loss']                  
        ave_loss += loss.item()
        loss.backward()
        grad_norm = clip_grads(model.parameters())
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_batch['img'].data) * data_batch['img'].data[0].shape[0], len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), flush=True)
            print('grad_norm: {:.4f}'.format(grad_norm), 'iteration: %d' % iter_idx)
            for loss_name, loss_value in outputs['log_vars'].items():
                print('%s:%.3f ' %( loss_name, loss_value), end=' ')
            print('')
        iter_idx += 1
        
    ave_loss/=len(train_loader)
    print('\nTraining set: Average loss: {:.4f}\n'.format(ave_loss))
    
    return dict(loss = ave_loss, iter_idx = iter_idx)


def test(args, model, test_loader, epoch=None):
    model.eval()
    test_loss = 0    
    loss_dict = dict(loss=0, loss_cls=0, loss_offset=0, loss_depth=0, loss_size=0, 
                     loss_rotsin=0, loss_centerness=0, loss_velo=0, loss_dir=0, 
                     loss_radarClass=0, loss_radarOffset=0, loss_radarDepthOffset=0) 

    with torch.no_grad():  
        for data_batch in tqdm(test_loader, 'Validation'): 
            
            outputs = model.val_step(data_batch)
            loss = outputs['loss']                      
            test_loss += loss.item()
            
            log_vars = outputs['log_vars']                    
            for loss_name in loss_dict:               
                loss_dict[loss_name] += log_vars[loss_name]
                                    
    test_loss/= len(test_loader)
    for loss_name in loss_dict:
        loss_dict[loss_name] /= len(test_loader)
        print('%s:%.3f ' % ( loss_name, loss_dict[loss_name]))
   
    print('\nVal set: Average loss: {:.4f}\n'.format(test_loss))
    
    return test_loss


def save_arguments(args):
    f = open(join(args.dir_result,'args.txt'),'w')
    f.write(repr(args)+'\n')
    f.close()

    out_path = join(args.dir_result,'cfg.py')
    cfg.dump(out_path)
    
    
def mkdir(dir1):
    if not os.path.exists(dir1): 
        os.makedirs(dir1)
        print('make directory %s' % dir1)


def filter_state_dict_keys(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v    
    return new_state_dict
    
    
def init_params(args, model, optimizer):
    loss_train=[]
    loss_val=[]
    
    start_epoch = 1
    iter_idx = 0
    state_dict_best = None
    loss_val_min = None
        
    if args.resume == True:
        f_checkpoint = join(args.dir_result, 'checkpoint.tar')        
        if os.path.isfile(f_checkpoint):
            print('Resume training')
            checkpoint = torch.load(f_checkpoint)   
            state_dict = checkpoint['state_dict'] 
            model.load_state_dict(state_dict) 
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1   
            iter_idx = checkpoint['iter_idx']
            loss_train, loss_val = checkpoint['loss']
            loss_val_min = checkpoint['loss_val_min']
            state_dict_best = checkpoint['state_dict_best']
        else:            
            print('No checkpoint file is found.')
                    
    if args.load_pretrained_fcos3d and start_epoch==1:
        print('Load pretrained weights of FCOS3D')
        _ = load_checkpoint(model, args.path_checkpoint_fcos3d, map_location='cpu', \
                            revise_keys=[(r'^module\.', ''), ('backbone', 'backbone_img'), ('neck', 'neck_img')])
                             
    return loss_train, loss_val, start_epoch, iter_idx, state_dict_best, loss_val_min


def save_checkpoint(epoch, iter_idx, model, optimizer, loss_train, loss_val, loss_val_min, state_dict_best, args):
    
    if epoch == 1:
        loss_val_min = loss_val[-1]
        state_dict_best = copy.deepcopy( model.state_dict() )
    elif loss_val[-1] < loss_val_min:
        loss_val_min = loss_val[-1]
        state_dict_best = copy.deepcopy( model.state_dict() )
            
    state = {'epoch': epoch,
             'iter_idx': iter_idx,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'loss': [loss_train, loss_val],
             'loss_val_min': loss_val_min,
             'state_dict_best': state_dict_best}
    
    torch.save(state, join(args.dir_result, 'checkpoint.tar'))
    return state
        
    
def plot_and_save_loss_curve(epoch, loss_train, loss_val):
    plt.close('all')
    plt.figure()  
    t=np.arange(1,epoch+1)
    plt.plot(t,loss_train,'b.-')
    plt.plot(t,loss_val,'r.-')
    plt.grid()
    plt.legend(['training loss','val loss'],loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')   
    plt.title('loss in logscale')
    plt.savefig(join(args.dir_result, 'loss.png'))


def init_env():    
    use_cuda = torch.cuda.is_available()    
    device0 = torch.device('cuda:0' if use_cuda else 'cpu')    
    cudnn.benchmark = True if use_cuda else False
    available_gpu_ids = [i for i in range(torch.cuda.device_count())]
    
    return device0, available_gpu_ids
 
    
def main(args):
                
    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = join(this_dir, '..', 'data', 'nuscenes')
        
    if args.path_checkpoint_fcos3d == None:
        this_dir = os.path.dirname(__file__)
        args.path_checkpoint_fcos3d = join(this_dir, '..', 'checkpoints', 'fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth')        
            
    if args.lr:
        cfg.optimizer_cfg['lr'] = args.lr
    else:
        cfg.optimizer_cfg['lr'] = cfg.optimizer_cfg['lr']
    
    args.train_ann_file = None 
    args.val_ann_file = join(args.dir_data, 'fusion_data', 'nus_infos_val.coco.json')
    args.test_ann_file = join(args.dir_data, 'fusion_data', 'nus_infos_test.coco.json')
    
    if not args.dir_result: 
        args.dir_result = join(args.dir_data, 'fusion_data', 'train_result', 'radiant_fcos3d')            
    mkdir(args.dir_result)  

    save_arguments(args)            
    device, available_gpu_ids = init_env()
    
    if args.do_eval:
        cfg.model_args['eval_mono']= args.eval_mono
        if args.eval_mono:
            print('---------------------------------------------------------------------------------')
            print('Evaluate cam outputs')
            print('---------------------------------------------------------------------------------')
        else:
            print('---------------------------------------------------------------------------------')
            print('Evaluate fusion outputs')
            print('---------------------------------------------------------------------------------')
            
    model = FCOSFusion3D(**cfg.model_args)
    model.init_weights()
    model = MMDataParallel(model.to(device), device_ids=available_gpu_ids)
    
    if not args.do_eval:    
        optimizer = build_optimizer(model, cfg.optimizer_cfg)    
        lr_updater = StepLrUpdater(**cfg.lr_config)

        loss_train, loss_val, start_epoch, iter_idx, state_dict_best, loss_val_min = \
        init_params(args, model, optimizer)
        
        if args.train_mini:
            args.train_ann_file = join(args.dir_data, 'fusion_data', 'nus_infos_train_mini.coco.json')
        else:
            args.train_ann_file = join(args.dir_data, 'fusion_data', 'nus_infos_train.coco.json')
        
        if args.val_mini:
            args.val_ann_file = join(args.dir_data, 'fusion_data', 'nus_infos_val_mini.coco.json')
        else:
            args.val_ann_file = join(args.dir_data, 'fusion_data', 'nus_infos_val.coco.json')
        
        train_loader = init_data_loader(args, NuScenesFusionDataset, 'train')
        val_loader = init_data_loader(args, NuScenesFusionDataset, 'val')
        
        lr_updater.before_run(optimizer)
        
        epoch = start_epoch
        while epoch <= args.epochs:
            start = timer()
            
            lr_updater.before_train_epoch(optimizer, epoch)
            
            result = train(args, model, train_loader, optimizer, lr_updater, epoch, iter_idx)                   
            
            loss_train.append(result['loss'])
            loss_val.append(test(args, model, val_loader, epoch))
            plot_and_save_loss_curve(epoch, loss_train, loss_val)
            
            iter_idx = result['iter_idx']
            state = save_checkpoint(epoch, iter_idx, model, optimizer, loss_train, loss_val, loss_val_min, state_dict_best, args)
            loss_val_min, state_dict_best = state['loss_val_min'], state['state_dict_best']         

            end = timer(); t = (end - start) / 60; print('Time used: %.1f minutes\n' % t)
            print('Training Epoch %d finished.' % (epoch))
            epoch += 1
                

    if args.do_eval: 
        if args.path_checkpoint_dwn == None:        
            f_checkpoint_mlp = join(args.dir_data, 'fusion_data', 'dwn_radiant_fcos3d', 'train_result', 'checkpoint.tar')  
        else:
            f_checkpoint_mlp = args.path_checkpoint_dwn
        model_mlp = FusionMLP()
        checkpoint_mlp = torch.load(f_checkpoint_mlp)                    
        model_mlp.load_state_dict(filter_state_dict_keys(checkpoint_mlp['state_dict']))
        
        data_loader = init_data_loader(args, NuScenesFusionDataset, 'test')
        f_checkpoint = join(args.dir_result, 'checkpoint.tar')        
        if os.path.isfile(f_checkpoint):
            print('load model')
            checkpoint = torch.load(f_checkpoint)                    
            model.load_state_dict(checkpoint['state_dict'])
        else:            
            print('No checkpoint file is found.')
            return
        
        outputs = single_gpu_test(model, model_mlp, data_loader)
        
        eval_file_name = 'eval_' + os.path.basename(data_loader.dataset.ann_file)[:-10]
        if args.eval_mono:
            out_dir = join(args.dir_result, eval_file_name, 'mono')
        else:
            out_dir = join(args.dir_result, eval_file_name, 'fusion')
        print(data_loader.dataset.evaluate(outputs, jsonfile_prefix=out_dir))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--dir_data', type=str)
    parser.add_argument('--dir_result', type=str)
    parser.add_argument('--load_pretrained_fcos3d', action='store_true', default=True)
    parser.add_argument('--path_checkpoint_fcos3d', type=str)
      
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--samples_per_gpu', type=int, default=1) 
    parser.add_argument('--test_samples_per_gpu', type=int, default=1)    
    parser.add_argument('--workers_per_gpu', type=int, default=2)  
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--lr', type=float, default=None, help='Learning rate') 
    parser.add_argument('--train_mini', action='store_true', default=False)
    parser.add_argument('--val_mini', type=bool, default=True)
    
    parser.add_argument('--do_eval', action='store_true', default=False, help='Evaluate')
    parser.add_argument('--eval_set', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--path_checkpoint_dwn', type=str)
    parser.add_argument('--eval_mono', action='store_true', default=False, help='Evaluate monocular component')
    
    args = parser.parse_args()
    main(args)
    