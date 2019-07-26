import os
import sys
sys.path.append('../../utils')
sys.path.append('../../backbone')
from dataset_3d_lc import UCF101_3d, Kinetics400_full_3d, SthsthV1_3d, HMDB51_3d
import argparse
import re

import torch
import torch.optim as optim
from torch.utils import data
import torch.nn as nn  
import numpy as np
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
from tqdm import tqdm

from model_3d_lc import *
from resnet_2d3d import neq_load_customized, inflate_imagenet_weights
from augmentation import *
from utils import AverageMeter, ConfusionMeter, save_checkpoint, write_log
from model_utils import calc_accuracy, calc_topk_accuracy
from vis_utils import denorm
import time
from tensorboardX import SummaryWriter
import ipdb


parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--version', default='v1', type=str)
parser.add_argument('--model', default='lc', type=str)
parser.add_argument('--dataset', default='ucf101', type=str)
parser.add_argument('--split', default=1, type=int)
parser.add_argument('--seq_len', default=5, type=int)
parser.add_argument('--num_seq', default=8, type=int)
parser.add_argument('--num_class', default=101, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--ds', default=3, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--pretrain', default='random', type=str)
parser.add_argument('--test', default='', type=str)
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--print_freq', default=5, type=int)
parser.add_argument('--val_freq', default=1, type=int)
parser.add_argument('--lp', action='store_true', help='Loose positive?')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--unit_test', action='store_true', help='Only train on 1 video?')
parser.add_argument('--verbose', '-v', action='store_true', help='show some score')
parser.add_argument('--vis_embed', action='store_true', help='visualize embedding?')
parser.add_argument('--network_type', default='2d3d', type=str)
parser.add_argument('--bn', default='batchnorm', type=str)
parser.add_argument('--train_what', default='last', type=str, help='Train what parameters?',
                    choices=['last','all','half', 'less', 'rnn', 'ft', 'b4', 'b3', 'b2'])
parser.add_argument('--num_fc', default=1, type=int)
parser.add_argument('--norm', action='store_true', help='Normalize context and pred vector?')
parser.add_argument('--prefix', default='tmp', type=str)
parser.add_argument('--use_rnn', action='store_true', help='Train RNN?')
parser.add_argument('--fake_finetune', action='store_true', help='train feature extractor with lower lr # obsolete')
parser.add_argument('--test_type', default='c', help='different types to load testing data')
parser.add_argument('--optim', default='adam', help='adam or sgd')
parser.add_argument('--full_resnet', action='store_true', help='Use full resnet? ')
parser.add_argument('--rnn_kernel', default=1, type=int)
parser.add_argument('--img_dim', default=0, type=int)
parser.add_argument('--final_bn', action='store_true', help='use final bn?')
parser.add_argument('--five_crop', action='store_true')


def main():
    global args
    args = parser.parse_args()
    
    if args.dataset == 'ucf101':
        args.num_class = 101
        if args.img_dim == 0: args.img_dim = 128
    elif args.dataset == 'ucf101-flow':
        args.num_class = 101
        if args.img_dim == 0: args.img_dim = 128
    elif args.dataset == 'kinetics400':
        args.num_class = 40
        if args.img_dim == 0: args.img_dim = 96
    elif args.dataset == 'sth':
        args.num_class = 174
        if args.img_dim == 0: args.img_dim = 96
    elif args.dataset == 'k400-full':
        args.num_class = 400 
        if args.img_dim == 0: args.img_dim = 128
    elif args.dataset == 'hmdb51':
        args.num_class = 51 
        if args.img_dim == 0: args.img_dim = 128

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda
    cuda = torch.device('cuda')

    # model
    if args.model == 'lc':
        model = LC(sample_size=args.img_dim, 
                   num_seq=args.num_seq, 
                   seq_len=args.seq_len, 
                   network=args.net,
                   network_type=args.network_type,
                   num_class=args.num_class,
                   batchnorm=args.bn,
                   dropout=args.dropout,
                   num_fc=args.num_fc,
                   norm=args.norm,
                   use_rnn=args.use_rnn,
                   rnn_kernel=args.rnn_kernel,
                   use_final_bn=args.final_bn,
                   full_resnet=args.full_resnet)
    else:
        raise ValueError('wrong model!')

    model = nn.DataParallel(model)
    model = model.to(cuda)

    # optimizer
    criterion = nn.CrossEntropyLoss()
    params = None

    if args.train_what == 'last':
        for name, param in model.module.named_parameters():
            params = []
            if ('resnet' in name) or ('final_rnn' in name) or ('convrnn' in name):
                param.requires_grad = False # freeze pre-trained resnet + rnn, only train FC
                params.append({'params': param, 'weight_decay': 0.0})
            else:
                params.append({'params': param})
    elif args.train_what == 'rnn':
        params = []
        for name, param in model.module.named_parameters():
            if ('resnet' in name):
                param.requires_grad = False # freeze pre-trained resnet, train rnn + FC
                params.append({'params': param, 'weight_decay': 0.0})
            else:
                params.append({'params': param})
    elif args.train_what == 'ft':
        print('=> finetune backbone with smaller lr')
        params = []
        for name, param in model.module.named_parameters():
            if ('resnet' in name) or ('final_rnn' in name) or ('convrnn' in name):
                params.append({'params': param, 'lr': args.lr/10})
            else:
                params.append({'params': param})
    elif args.train_what == 'b2':
        params = []
        for name, param in model.module.named_parameters():
            if ('resnet' in name) and ('layer4' not in name) \
                and ('layer3' not in name) and ('layer2' not in name):
                param.requires_grad = False # train layer2 and afterwards
                params.append({'params': param, 'weight_decay': 0.0})
            else:
                params.append({'params': param})
    elif args.train_what == 'b3':
        params = []
        for name, param in model.module.named_parameters():
            if ('resnet' in name) and ('layer4' not in name) and ('layer3' not in name):
                param.requires_grad = False # train layer3 and afterwards
                params.append({'params': param, 'weight_decay': 0.0})
            else:
                params.append({'params': param})
    elif args.train_what == 'b4':
        params = []
        for name, param in model.module.named_parameters():
            if ('resnet' in name) and ('layer4' not in name):
                param.requires_grad = False # train layer4 and afterwards
                params.append({'params': param, 'weight_decay': 0.0})
            else:
                params.append({'params': param})
    else: # train all
        pass 
    
    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    if params is None:
        params = model.parameters()

    if args.optim == 'adam': # TODO: need to decide best scheduler
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
        if args.dataset == 'k400-full':
            lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=[300,400], repeat=1)
        elif args.dataset == 'hmdb51':
            lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=[150,250,300], repeat=3)
        elif args.dataset == 'ucf101':
            if args.img_dim == 224:
                lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=[300,400,500], repeat=3) # [60,80,100], [100,150]
            else:
                lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=[60, 80, 100], repeat=3) # [60,80,100], [100,150]
        else:
            lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=[40,60], repeat=3)
        # lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=[20,40,50], repeat=4) # this is called quick decay
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    args.old_lr = None

    best_acc = 0
    global iteration; iteration = 0

    if args.resume:
        if os.path.isfile(args.resume):
            args.old_lr = float(re.search('_lr(.+?)_', args.resume).group(1))
            print("=> loading resumed checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            if not args.reset_lr: # if didn't reset lr
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('==== Change lr from %f to %f ====' % (args.old_lr, args.lr))
            if 'iteration' in checkpoint.keys():
                iteration = checkpoint['iteration']
            else:
                iteration = 0
            print("=> loaded resumed checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.test:
        vis_embed = False # visualize embedding or not.
        print("=> load testing set in mode '%s'" % args.test_type)
        if os.path.isfile(args.test):
            print("=> loading testing checkpoint '{}'".format(args.test))
            checkpoint = torch.load(args.test)
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                print('=> Warning: weight structure is not equal to test model; Use non-equal load ==')
                vis_embed = args.vis_embed
                model = neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded testing checkpoint '{}' (epoch {})"
                  .format(args.test, checkpoint['epoch']))
            global num_epoch
            num_epoch = checkpoint['epoch']
        elif args.test == 'random':
            print("=> [Warning] loaded random weights")
            vis_embed = args.vis_embed
        else: 
            raise ValueError()
        if args.dataset == 'sth':
            transform = transforms.Compose([
                CenterCrop(size=args.img_dim),
                ToTensor(),
                Normalize()
            ])
        else:
            if args.five_crop:
                transform = transforms.Compose([
                    # CenterCrop(consistent=True, size=(192,256)), # remove black bar?? - 11:54AM 
                    # FiveCropSplit(size=180),
                    FiveCropSplit(size=224),
                    transforms.Lambda(lambda crops: 
                        [Scale(size=(args.img_dim,args.img_dim))(crop) for crop in crops]),
                    transforms.Lambda(lambda crops:
                        [Normalize()(ToTensor()(crop)) for crop in crops])
                    ])
            else:
                transform = transforms.Compose([
                    # RandomRotation(consistent=True, degree=5, p=0.05),
                    # CenterCrop(consistent=True, size=(192,256)), # remove black bar?? - 11:54AM 
                    # RandomSizedCrop(consistent=True, size=180, p=0.0), # have zoom-in zoom-out
                    # Scale(size=(args.img_dim,args.img_dim)),
                    RandomSizedCrop(consistent=True, size=224, p=0.0),
                    Scale(size=(args.img_dim,args.img_dim)),
                    # RandomHorizontalFlip(consistent=True),
                    # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.1),
                    ToTensor(),
                    Normalize()
                ])
                # transform = transforms.Compose([
                #     CenterCrop(consistent=True, size=(192,256)), # remove black bar?? - 11:54AM 
                #     FiveCrop(size=180),
                #     transforms.Lambda(lambda crops: 
                #         [Scale(size=(args.img_dim,args.img_dim))(crop) for crop in crops]),
                #     transforms.Lambda(lambda crops:
                #         [Normalize()(ToTensor()(crop)) for crop in crops])
                #     ])
            

        test_loader = get_data(transform, 'test') # note! 
        test_loss, test_acc = test(test_loader, model, criterion, vis_embed)
        sys.exit()
    else: # not test
        torch.backends.cudnn.benchmark = True

    if (not args.resume) and args.pretrain:
        if args.pretrain == 'random':
            print('=> using random weights')
        elif args.pretrain == 'kinetics':
            print('=> using kinetics pretrained weights')
            assert args.net == 'resnet18'
            checkpoint = torch.load('../../backbone/resnet-18-kinetics.pth', map_location=torch.device('cpu'))
            model = neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded kinetics pretrained checkpoint")
        elif args.pretrain == 'imagenet':
            print('=> using (inflated) imagenet pretrained weights')
            model = inflate_imagenet_weights(model, args.net)
        elif os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            # model.load_state_dict(checkpoint['state_dict'], strict=False)
            model = neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})"
                  .format(args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    # load data
    if args.dataset == 'sth': # obsolete
        transform = transforms.Compose([
            # RandomRotation(consistent=True),
            RandomHorizontalFlip(consistent=True),
            # RandomRotation(consistent=False),
            RandomCrop(size=96, consistent=True),
            # RandomGray(consistent=False, p=0.3),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0),
            ToTensor(),
            Normalize()
        ])
    elif (args.dataset == 'ucf101') or (args.dataset == 'hmdb51'): # because 0.7^2*0.8=0.4, there is 40% probability to return CenterCropped raw image
        transform = transforms.Compose([
            # CenterCrop(consistent=True, size=(192,256)), # remove black bar?? - 11:54AM 
            # RandomRotation(consistent=True, degree=5, p=0.1),
            # RandomSizedCrop(consistent=True, size=180, p=0.3), # have zoom-in zoom-out
            # Scale(size=(args.img_dim,args.img_dim)),
            RandomSizedCrop(consistent=True, size=224, p=1.0), # was 0.3
            Scale(size=(args.img_dim,args.img_dim)),
            RandomHorizontalFlip(consistent=True),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.3, consistent=True),
            ToTensor(),
            Normalize()
        ])
        val_transform = transforms.Compose([
            # CenterCrop(consistent=True, size=(192,256)), # remove black bar?? - 11:54AM 
            # RandomSizedCrop(consistent=True, size=180, p=0.0), # have zoom-in zoom-out
            # Scale(size=(args.img_dim,args.img_dim)),
            RandomSizedCrop(consistent=True, size=224, p=0.3),
            Scale(size=(args.img_dim,args.img_dim)),
            RandomHorizontalFlip(consistent=True),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3, consistent=True),
            ToTensor(),
            Normalize()
        ])
    elif args.dataset == 'k400-full': # because 0.7^2*0.8=0.4, there is 40% probability to return CenterCropped raw image
        transform = transforms.Compose([
            RandomRotation(consistent=True, degree=5, p=0.05),
            RandomSizedCrop(consistent=True, size=args.img_dim, p=0.2), # have zoom-in zoom-out
            RandomHorizontalFlip(consistent=True),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.1),
            ToTensor(),
            Normalize()
        ])
    elif (args.dataset == 'ucf101-flow'):
        transform = transforms.Compose([
            RandomRotation(consistent=True, degree=5, p=0.05),
            RandomSizedCrop(consistent=True, size=224, p=0.2), # have zoom-in zoom-out
            Scale(size=(args.img_dim,args.img_dim)),
            RandomHorizontalFlip(consistent=True),
            # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.1),
            ToTensor(),
            Normalize(mean=[0.0, 0.5, 0.5], std=[1.0,1.0,1.0])
        ])

    train_loader = get_data(transform, 'train')
    if (args.dataset == 'ucf101') or (args.dataset == 'hmdb51'):
        val_loader = get_data(val_transform, 'val')
    else:
        val_loader = get_data(transform, 'val')

    # setup tools
    global de_normalize
    de_normalize = denorm()

    global img_path
    img_path, model_path = set_path(args)

    global writer_train
    writer_val = SummaryWriter(log_dir=os.path.join(img_path, 'val'))
    writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))

    val_counter = 0
    # main loop
    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step(epoch)

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)

        gap = args.val_freq
        val_counter += 1
        if val_counter < args.val_freq: 
            continue

        val_counter = 0 # reset
        val_loss, val_acc = validate(val_loader, model, criterion)

        writer_train.add_scalar('global/loss', train_loss, epoch)
        writer_train.add_scalar('global/accuracy', train_acc, epoch)
        writer_val.add_scalar('global/loss', val_loss, epoch)
        writer_val.add_scalar('global/accuracy', val_acc, epoch)

        # save check_point
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'epoch': epoch+1,
            'net': args.net,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'iteration': iteration
        }, is_best, gap, filename=os.path.join(model_path, 'epoch%s.pth.tar' % str(epoch+1)), keep_all=True)
    
    import ipdb; ipdb.set_trace()


def train(data_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    model.train()
    global iteration

    for idx, (input_seq, target) in enumerate(data_loader):
        tic = time.time()
        input_seq = input_seq.to(cuda)
        target = target.to(cuda)
        B = input_seq.size(0)
        h0 = model.module.init_hidden(B)
        output, _ = model(input_seq, h0)

        # visualize
        if (iteration == 0) or (iteration == args.print_freq):
            if B > 2: input_seq = input_seq[0:2,:]
            writer_train.add_image('input_seq', 
                                   de_normalize(
                                       vutils.make_grid(
                                           input_seq.transpose(2,3).contiguous().view(-1,3,args.img_dim,args.img_dim), nrow=args.num_seq*args.seq_len)), 
                                   iteration)
        del input_seq

        [_, N, D] = output.size()
        output = output.view(B*N, D)
        target = target.repeat(1, N).view(-1)

        loss = criterion(output, target)
        acc = calc_accuracy(output, target)

        del target 

        losses.update(loss.item(), B)
        accuracy.update(acc.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % args.print_freq == 0:
            # print(output[0, args.N//2-2::])
            if args.verbose: print(output[0:10])
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.local_avg:.4f})\t'
                  'Acc: {acc.val:.4f} ({acc.local_avg:.4f}) T:{3:.2f}\t'.format(
                   epoch, idx, len(data_loader), time.time()-tic,
                   loss=losses, acc=accuracy))

            total_weight = 0.0
            decay_weight = 0.0
            for m in model.parameters():
                if m.requires_grad: decay_weight += m.norm(2).data
                total_weight += m.norm(2).data
            print('Decay weight / Total weight: %.3f/%.3f' % (decay_weight, total_weight))
            
            writer_train.add_scalar('local/loss', losses.val, iteration)
            writer_train.add_scalar('local/accuracy', accuracy.val, iteration)

            iteration += 1

    return losses.local_avg, accuracy.local_avg

def validate(data_loader, model, criterion):
    losses = AverageMeter()
    accuracy = AverageMeter()
    model.eval()
    with torch.no_grad():
        for idx, (input_seq, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq = input_seq.to(cuda)
            target = target.to(cuda)
            B = input_seq.size(0)
            h0 = model.module.init_hidden(B)
            output, _ = model(input_seq, h0)

            [_, N, D] = output.size()
            output = output.view(B*N, D)
            target = target.repeat(1, N).view(-1)

            loss = criterion(output, target)
            acc = calc_accuracy(output, target)

            losses.update(loss.item(), B)
            accuracy.update(acc.item(), B)
                
    print('Loss {loss.avg:.4f}\t'
          'Acc: {acc.avg:.4f} \t'.format(loss=losses, acc=accuracy))
    return losses.avg, accuracy.avg

def test(data_loader, model, criterion, vis_embed=False):
    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    confusion_mat = ConfusionMeter(args.num_class)
    if vis_embed: 
        embed_list = []; target_list = []
        test_log = os.path.exists(os.path.join(args.test[:-6], 'test'))
        if not os.path.exists(test_log):
            os.makedirs(test_log)
        writer_test = SummaryWriter(log_dir=test_log) # log_dir not working
    model.eval()
    with torch.no_grad():
        if args.five_crop:
            # for crop_idx in [5]:
            crop_idx = 5
            for direction in [None]:
                print('Now testing: ', direction)
                new_transform = transforms.Compose([
                    CenterCrop(consistent=True, size=(192,256)), # remove black bar?? - 11:54AM 
                    FiveCropSplit(size=180, idx=crop_idx),
                    RandomHorizontalFlip(consistent=True, command=direction),
                    Scale(size=(args.img_dim,args.img_dim)),
                    ToTensor(),
                    Normalize()
                    ])

                data_loader.dataset.transform = new_transform
                for idx, (input_seq, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
                    input_seq = input_seq.to(cuda)
                    target = target.to(cuda)
                    B = input_seq.size(0)
                    h0 = model.module.init_hidden(B)
                    if args.test_type == 'c': input_seq = input_seq.squeeze(0) # squeeze the '1' batch dim
                    output, feature = model(input_seq, h0)
                    del input_seq, feature
                    top1, top5 = calc_topk_accuracy(torch.mean(
                                                    torch.mean(
                                                        nn.functional.softmax(output,2),
                                                        0),0, keepdim=True), 
                                                    target, (1,5))

                    output = torch.mean(torch.mean(output, 0), 0, keepdim=True)
                    if vis_embed:
                        raise NotImplementedError()
                        feature = torch.mean(torch.mean(feature, 0), 0, keepdim=True)
                        target_list.append(target.data)
                        embed_list.append(feature.data)
                    loss = criterion(output, target.squeeze(-1))

                    losses.update(loss.item(), B)
                    acc_top1.update(top1.item(), B)
                    acc_top5.update(top5.item(), B)

                    _, pred = torch.max(output, 1)
                    confusion_mat.update(pred, target.view(-1).byte())
        else:
            for idx, (input_seq, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
                input_seq = input_seq.to(cuda)
                target = target.to(cuda)
                B = input_seq.size(0)
                h0 = model.module.init_hidden(B)
                if args.test_type == 'c': input_seq = input_seq.squeeze(0) # squeeze the '1' batch dim
                output, _ = model(input_seq, h0)
                del input_seq
                top1, top5 = calc_topk_accuracy(torch.mean(
                                                torch.mean(
                                                    nn.functional.softmax(output,2),
                                                    0),0, keepdim=True), 
                                                target, (1,5))
                acc_top1.update(top1.item(), B)
                acc_top5.update(top5.item(), B)
                del top1, top5

                output = torch.mean(torch.mean(output, 0), 0, keepdim=True)
                if vis_embed:
                    raise NotImplementedError()
                    # feature = torch.mean(torch.mean(feature, 0), 0, keepdim=True)
                    # target_list.append(target.data)
                    # embed_list.append(feature.data)
                loss = criterion(output, target.squeeze(-1))

                losses.update(loss.item(), B)
                del loss


                _, pred = torch.max(output, 1)
                confusion_mat.update(pred, target.view(-1).byte())

    if vis_embed: 
        raise NotImplementedError() 
        embed_list = torch.cat(embed_list, 0)
        embed_list = embed_list / embed_list.norm(dim=1, keepdim=True)
        target_list = torch.cat(target_list, 0)
        writer_test.add_embedding(embed_list, metadata=[dataset.decode_action(i) for i in target_list.squeeze().tolist()])        
    print('Loss {loss.avg:.4f}\t'
          'Acc top1: {top1.avg:.4f} Acc top5: {top5.avg:.4f} \t'.format(loss=losses, top1=acc_top1, top5=acc_top5))
    confusion_mat.plot_mat(args.test+'.svg')
    write_log(content='test type {args.test_type}\nLoss {loss.avg:.4f}\t Acc top1: {top1.avg:.4f} Acc top5: {top5.avg:.4f} \t'.format(loss=losses, top1=acc_top1, top5=acc_top5, args=args),
              epoch=num_epoch,
              filename=os.path.dirname(args.test)+'/test_log.md')
    import ipdb; ipdb.set_trace()
    return losses.avg, [acc_top1.avg, acc_top5.avg]

def get_data(transform, mode='train'):
    if args.unit_test: print('* Unit Test Mode *')
    print('Loading data for "%s" ...' % mode)
    global dataset
    elif args.dataset == 'ucf101':
        dataset = UCF101_3d(mode=mode, 
                         transform=transform, 
                         seq_len=args.seq_len,
                         num_seq=args.num_seq,
                         downsample=args.ds,
                         unit_test=args.unit_test,
                         test_type=args.test_type,
                         which_split=args.split)
    elif args.dataset == 'hmdb51':
        dataset = HMDB51_3d(mode=mode, 
                         transform=transform, 
                         seq_len=args.seq_len,
                         num_seq=args.num_seq,
                         downsample=1,
                         unit_test=args.unit_test,
                         test_type=args.test_type,
                         which_split=args.split)
    else:
        raise ValueError('dataset not supported')
    my_sampler = data.RandomSampler(dataset)
    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=16,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=16,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'test':
        data_loader = data.DataLoader(dataset,
                                      batch_size=1,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=16,
                                      pin_memory=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader

def set_path(args):
    if args.resume: exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = 'log_{args.prefix}/{0}{args.dataset}-{args.img_dim}-\
sp{args.split}_{1}{2}{args.version}_{args.model}_bs{args.batch_size}_\
lr{3}_wd{args.wd}_ds{args.ds}_seq{args.num_seq}_len{args.seq_len}_fc{args.num_fc}_\
dp{args.dropout}{4}_train-{args.train_what}_optim-{args.optim}{5}{6}'.format(
                    'ut/' if args.unit_test else '', \
                    '3d-r%s' % args.net[6::] if args.network_type=='3d' else 'r%s' % args.net[6::], \
                    'F-' if args.full_resnet else '',
                    args.old_lr if args.old_lr is not None else args.lr, \
                    '_'+args.bn if args.bn else '', \
                    '_pt='+args.pretrain.replace('/','-') if args.pretrain else '', \
                    '_RNN' if args.use_rnn else '', \
                    args=args)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    return img_path, model_path

def MultiStepLR_Restart_Multiplier(epoch, gamma=0.1, step=[10,15,20], repeat=3):
    '''return the multipier for LambdaLR, 
    0  <= ep < 10: gamma^0
    10 <= ep < 15: gamma^1 
    15 <= ep < 20: gamma^2
    20 <= ep < 30: gamma^0 ... repeat 3 cycles and then keep gamma^2'''
    max_step = max(step)
    effective_epoch = epoch % max_step
    if epoch // max_step >= repeat:
        exp = len(step) - 1
    else:
        exp = len([i for i in step if effective_epoch>=i])
    return gamma ** exp

if __name__ == '__main__':
    main()
