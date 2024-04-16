from __future__ import division
import warnings
from Networks.models import base_patch16_384_token, base_patch16_384_gap, base_patch16_384_attention, base_patch16_384_cls, base_patch16_384_swin
from Networks.cross_entropy_w import CrossEntropyLoss_W
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
from torch.nn import functional as F

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
        model = base_patch16_384_swin(pretrained=True,mode = args['mode'])
    elif args['model_type'] == 'cls':
        model = base_patch16_384_cls(pretrained=True)
    else:
        print("Do not have the network: {}".format(args['model_type']))
        exit(0)

    print('Number of parameters: {}'.format(sum(p.numel() for p in model.parameters())))

    gpu_id = args['gpu_id'].split(',')
    model = nn.DataParallel(model, device_ids=range(len(gpu_id)) )
    model = model.cuda()

    criterion = nn.L1Loss(size_average=False).cuda()
    # criterion_cls = nn.BCEWithLogitsLoss().cuda()
    criterion_cls = nn.CrossEntropyLoss().cuda()
    # criterion_cls = CrossEntropyLoss_W()

    optimizer = torch.optim.Adam(
        [  #
            {'params': model.parameters(), 'lr': args['lr']},
        ], lr=args['lr'], weight_decay=args['weight_decay'])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1, last_epoch=-1)
    print(args['pre'])

    # args['save_path'] = args['save_path'] + str(args['rdt'])
    print(args['save_path'])
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
    train_data = pre_data(train_list, args, train=True)
    test_data = pre_data(val_list, args, train=False)

    for epoch in range(args['start_epoch'], args['epochs']):

        start = time.time()
        train(train_data, model, criterion, criterion_cls, optimizer, epoch, args, scheduler)
        end1 = time.time()

        if epoch % 5 == 0 and epoch >= 10:
            prec1 = validate(test_data, model, args)
            end2 = time.time()
            is_best = prec1 < args['best_pred']
            args['best_pred'] = min(prec1, args['best_pred'])

            print(' * best MAE {mae:.3f} '.format(mae=args['best_pred']), args['save_path'], end1 - start, end2 - end1)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args['pre'],
                'state_dict': model.state_dict(),
                'best_prec1': args['best_pred'],
                'optimizer': optimizer.state_dict(),
            }, is_best, args['save_path'])


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


def train(Pre_data, model, criterion, criterion_cls, optimizer, epoch, args, scheduler):
    losses = AverageMeter()
    losses_cls = AverageMeter()
    acc_cls = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=True,
                            transform=transforms.Compose([
                                # transforms.ColorJitter(brightness=0.2),
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
    start_ep = time.time()

    for i, (fname, img, gt_count) in enumerate(train_loader):

        data_time.update(time.time() - end)
        img = img.cuda()

        out1, out2 = model(img)
        gt_count = gt_count.type(torch.FloatTensor).cuda().unsqueeze(1)
        _, cls_predicted = torch.max(out2.data, 1)

        # print(out1.shape, kpoint.shape)
        # loss = criterion(out1, gt_count)
        # slash_step = gt_count * 0.05
        # loss1 = torch.abs(out1 - gt_count) - slash_step
        # loss1 = loss1.sum()
        # loss = torch.maximum(loss1, torch.zeros_like(loss1))

        # 5 cls loss
        # gt_cls = torch.log10(gt_count)
        # gt_cls[gt_cls<=0] = 0
        # gt_cls = torch.ceil(gt_cls).long().flatten(0)

        # 12 cls loss
        gt_cls = gt_count/100.0
        gt_cls[gt_cls<=0] = -1
        gt_cls[gt_cls>10] = 10
        gt_cls = torch.floor(gt_cls).long().flatten(0) + 1

        # 2 cls loss
        # gt_cls = (gt_count > 0).float()
        loss2 = 100 * criterion_cls(out2, gt_cls)

        # w = torch.abs(cls_predicted.clone().detach()-gt_cls)+1
        # loss2 = 100 * torch.mean(F.cross_entropy(out2, gt_cls,reduction='none') * w)

        # for j in range(gt_cls.shape[0]):
        #     w = torch.range(0, 11).cuda()
        #     w = torch.abs(w-gt_cls[j])
        #     w[gt_cls[j]] = 11
        #     loss2 += criterion_cls(out2[j:j+1,:], gt_cls[j:j+1],w)
        # loss2 = loss2/gt_cls.shape[0]


        # fine count
        gt_cls_used = cls_predicted.detach() if epoch >= args["ref_pred_epoch"] else gt_cls
        predicted = gt_cls_used - 1
        predicted[predicted<0] = 0
        predicted = predicted.float()*100
        res_gt_count = gt_count - predicted.unsqueeze(1)
        loss = criterion(out1, res_gt_count)

        loss += loss2

        losses.update(loss.item(), img.size(0))
        losses_cls.update(loss2.item(), img.size(0))
        acc_cls.update((cls_predicted == gt_cls).sum().item()/img.size(0), img.size(0))
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i == len(train_loader) - 1:
            print('4_Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Cls Acc {acc_cls.val:.4f} ({acc_cls.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc_cls=acc_cls))
    end_ep = time.time()-start_ep
    print('4_Epoch: Minutes per Epoch: %.2f'% (end_ep/60.0))

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

    mae = 0.0
    mse = 0.0
    visi = []
    index = 0

    for i, (fname, img, gt_count) in enumerate(test_loader):

        img = img.cuda()
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            out1, out2 = model(img)
            # out2 = torch.sigmoid(out2)
            # mask = (out2 > 0.2).float()
            _, cls_predicted = torch.max(out2.data, 1)
            predicted = cls_predicted - 1
            predicted[predicted<0] = 0
            predicted = predicted.float()*100
            predicted_count = out1 * (cls_predicted>0).float().unsqueeze(1) + predicted.unsqueeze(1)

            count = torch.sum(predicted_count).item()

        gt_count = torch.sum(gt_count).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)

        if i % 15 == 0:
            print('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))


    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)

    nni.report_intermediate_result(mae)
    print(' \n* MAE {mae:.3f}\n'.format(mae=mae), '* MSE {mse:.3f}'.format(mse=mse))

    return mae


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
