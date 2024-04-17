from __future__ import division
import warnings
from Networks.models import base_patch16_384_token, base_patch16_384_gap, base_patch16_384_attention, \
    base_patch16_384_fgap, base_patch16_384_swin, base_patch16_384_effnet, base_patch16_384_mamba
import torch.nn as nn
from torchvision import transforms
import dataset
import math
from utils import save_checkpoint, setup_seed
import torch
import os
import logging
import nni
from nni.utils import merge_parameter
from config import return_args, args
import numpy as np
from image import load_data

warnings.filterwarnings('ignore')
import time

setup_seed(args.seed)

logger = logging.getLogger('mnist_AutoML')

def main(args):
    if args['dataset'] == 'ShanghaiA':
        train_file = './npydata/ShanghaiA_train.npy'
        test_file = './npydata/ShanghaiA_test.npy'
    elif args['dataset'] == 'ShanghaiB':
        train_file = './npydata/ShanghaiB_train.npy'
        test_file = './npydata/ShanghaiB_test.npy'
    elif args['dataset'] == 'UCF_QNRF':
        train_file = './npydata/qnrf_train.npy'
        test_file = './npydata/qnrf_test.npy'
    elif args['dataset'] == 'JHU':
        train_file = './npydata/jhu_train.npy'
        test_file = './npydata/jhu_val.npy'
    elif args['dataset'] == 'NWPU':
        train_file = './npydata/nwpu_train.npy'
        test_file = './npydata/nwpu_val.npy'

    with open(train_file, 'rb') as outfile:
        train_list = np.load(outfile).tolist()
    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()

    print(len(train_list), len(val_list))

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_id']

    if args['model_type'] == 'token':
        model = base_patch16_384_token(pretrained=True)
    elif args['model_type'] == 'gap':
        model = base_patch16_384_gap(pretrained=True)
    elif args['model_type'] == 'attention':
        model = base_patch16_384_attention(pretrained=True)
    elif args['model_type'] == 'swin':
        model = base_patch16_384_swin(pretrained=True, mode=args['mode'])
    elif args['model_type'] == 'fgap':
        model = base_patch16_384_fgap(pretrained=True)
    elif args['model_type'] == 'effnet':
        model = base_patch16_384_effnet()
    elif args['model_type'] == 'mamba':
        model = base_patch16_384_mamba(pretrained=False,mode = args['mode'])
    else:
        print("Do not have the network: {}".format(args['model_type']))
        exit(0)

    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    criterion = nn.L1Loss(size_average=False).cuda()

    optimizer = torch.optim.Adam(
        [  #
            {'params': model.parameters(), 'lr': args['lr']},
        ], lr=args['lr'], weight_decay=args['weight_decay'])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1, last_epoch=-1)
    print(args['pre'])

    # args['save_path'] = args['save_path'] + str(args['rdt'])
    print(args['save_path'])#./output/ShanghaiA_swim_cb
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if args['pre']:
        if os.path.isfile(args['pre']):
            print("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            print("=> no checkpoint found at '{}'".format(args['pre']))

    torch.set_num_threads(args['workers'])

    print(args['best_pred'], args['start_epoch'])

    test_data = pre_data(val_list, args, train=False)#

    '''inference'''
    prec1 = validate(test_data, model, args)

    print(' * best MAE {mae:.3f} '.format(mae=args['best_pred']))





def pre_data(train_list, args, train):
    print("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        img, gt_count = load_data(Img_path, args, train)

        blob = {}
        blob['img'] = img
        blob['gt_count'] = gt_count
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1

        '''for debug'''
        # if j> 10:
        #     break
    return data_keys


def train(Pre_data, model, criterion, optimizer, epoch, args, scheduler):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),

                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            batch_size=args['batch_size'],
                            num_workers=args['workers'],
                            args=args),
        batch_size=args['batch_size'], drop_last=False)
    args['lr'] = optimizer.param_groups[0]['lr']
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args['lr']))

    model.train()
    end = time.time()

    for i, (fname, img, gt_count) in enumerate(train_loader):

        data_time.update(time.time() - end)
        img = img.cuda()

        out1 = model(img)
        gt_count = gt_count.type(torch.FloatTensor).cuda().unsqueeze(1)

        # print(out1.shape, kpoint.shape)
        loss = criterion(out1, gt_count)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args['print_freq'] == 0:
            print('4_Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    scheduler.step()


def validate(Pre_data, model, args):
    print('begin test')
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),

                            ]),
                            args=args, train=False),
        batch_size=1)

    model.eval()

    mae1 = 0.0
    mae2 = 0.0
    mae3 = 0.0
    mae4 = 0.0
    mse1 = 0.0
    mse2 = 0.0
    mse3 = 0.0
    mse4 = 0.0
    mse = 0.0
    visi = []
    index = 0
    throughput = AverageMeter()
    a = 0
    b = 0
    c = 0
    d = 0
    for i, (fname, img, gt_count) in enumerate(test_loader):

        img = img.cuda()
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        if i>10:
            start_time = time.time()
        with torch.no_grad():
    
            out1 = model(img)
            count = torch.sum(out1).item()
        if i>10:
            during_time = time.time() - start_time
            throughput.update(during_time,1)

        gt_count = torch.sum(gt_count).item()

        if gt_count < 235:
            mae1 += abs(gt_count - count)
            mse1 += abs(gt_count - count) * abs(gt_count - count)
            smae1 = mae1
            smse1 = mse1
            a += 1

        if 234 < gt_count < 425:
            mae2 += abs(gt_count - count)
            mse2 += abs(gt_count - count) * abs(gt_count - count)
            smae2 = mae2
            smse2 = mse2
            b  += 1
        if 424 < gt_count:
            mae3 += abs(gt_count - count)
            mse3 += abs(gt_count - count) * abs(gt_count - count)
            smae3 = mae3
            smse3 = mse3
            c += 1
     
    
    if a != 0:
        mae1 = mae1 * 1.0 / a   
        mse1 = math.sqrt(mse1 / a) 
    if b != 0:
        mae2 = mae2 * 1.0 / b
        mse2 = math.sqrt(mse2 / b) 
    if c != 0:
        mae3 = mae3 * 1.0 / c
        mse3 = math.sqrt(mse3 / c) 
    sum_mae =(smae1+smae2+smae3)/182
    sum_mse=math.sqrt(smse1+smse2+smse3 /182) 
    
    
    print(' \n* MAE1 {mae1:.3f}\n'.format(mae1=mae1),'* MSE1 {mse1:.3f}'.format(mse1=mse1),' \n* MAE2 {mae2:.3f}\n'.format(mae2=mae2),'* MSE2 {mse2:.3f}'.format(mse2=mse2))
    print(' \n* MAE3 {mae3:.3f}\n'.format(mae3=mae3),'* MSE {mse3:.3f}'.format(mse3=mse3),' \n* SUM_MAE {mae4:.3f}\n'.format(mae4=sum_mae),'* SUM_MSE {mse4:.3f}'.format(mse4=sum_mse))

    print(a,b,c,d,sum)

    return sum


class AverageMeter(object):
    
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    print(params)

    main(params)
