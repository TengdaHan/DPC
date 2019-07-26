import os
import sys
sys.path.append('../utils')
from dataset_3d import *
import argparse
import re
import math

import torch
import torch.optim as optim
from torch.utils import data
import numpy as np
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
from tqdm import tqdm

from model_3d import *
from resnet_2d3d import neq_load_customized, inflate_imagenet_weights, neq_load_kinetics
from augmentation import *
from utils import AverageMeter, ConfusionMeter, LabelQueue, save_checkpoint
from model_utils import calc_accuracy_binary, calc_accuracy_with_mask_no_step, \
     calc_accuracy_with_mask_with_step, calc_topk_accuracy
from vis_utils import denorm
import time
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
plt.switch_backend('agg')

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--model', default='dpc-rnn', type=str)
parser.add_argument('--dataset', default='ucf101', type=str)
parser.add_argument('--seq_len', default=5, type=int)
parser.add_argument('--num_seq', default=8, type=int)
parser.add_argument('--num_fc', default=2, type=int)
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--pretrain', default='', type=str)
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--print_freq', default=5, type=int)
parser.add_argument('--val_freq', default=1, type=int)
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--unit_test', action='store_true', help='Only train on 1 video?')
parser.add_argument('--verbose', '-v', action='store_true', help='show some score')
parser.add_argument('--network_type', default='2d3d', type=str)
parser.add_argument('--bn', default='batchnorm', type=str)
parser.add_argument('--prefix', default='tmp', type=str)
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--pooled_size', default=0, type=int)
parser.add_argument('--full_resnet', action='store_true', help='Use full resnet? ')
parser.add_argument('--strict_step', default=True, type=bool, help='Only use last few steps for score?') # TODO check this line
parser.add_argument('--img_dim', default=0, type=int)
parser.add_argument('--test', default='', type=str)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    global args
    args = parser.parse_args()
    if args.img_dim == 0:
        if args.dataset == 'sth':
            args.img_dim = 96
        if args.dataset == 'ucf101':
            args.img_dim = 128
        if args.dataset == 'k400-full':
            args.img_dim = 128

    if args.pooled_size == 0:
        if args.full_resnet:
            args.pooled_size = int(math.ceil(args.img_dim / 32))
        else:
            args.pooled_size = int(math.ceil(args.img_dim / 16))

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda
    cuda = torch.device('cuda')

    # model
    if args.model == 'dpc-rnn':
        model = DPC_RNN(sample_size=args.img_dim,
                         num_seq=args.num_seq,
                         seq_len=args.seq_len,
                         network=args.net,
                         network_type=args.network_type,
                         pred_step=args.pred_step,
                         num_fc=args.num_fc,
                         batchnorm=args.bn,
                         norm=args.norm,
                         pooled_size=args.pooled_size,
                         affine=(not args.no_affine),
                         full_resnet=args.full_resnet,
                         strict_step=args.strict_step,
                         multi_hypo=args.multi_hypo,
                         real_rate=args.real_rate,
                         bi_dir=args.bi_dir)
    else:
        raise ValueError('wrong model!')

    model = nn.DataParallel(model)
    model = model.to(cuda)

    # optimizer
    print('==> Using softmax')
    criterion = nn.CrossEntropyLoss()

    if args.train_what == 'last':
        for name, param in model.module.resnet.named_parameters():
            param.requires_grad = False
    else: # train all
        pass

    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)

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
            if ('MH' not in args.resume) and args.multi_hypo:
                assert args.single2multi, 'Please specify if convert single- to multi-Hypothesis'
                model = single2multi(model, checkpoint['state_dict'])
            else: # normal state
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
        if os.path.isfile(args.test):
            print("=> loading testing checkpoint '{}'".format(args.test))
            checkpoint = torch.load(args.test)
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                print('=> Warning: weight structure is not equal to test model; Use non-equal load ==')
                model = neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded testing checkpoint '{}' (epoch {})"
                  .format(args.test, checkpoint['epoch']))
        elif args.test == 'imagenet':
            print('=> using (inflated) imagenet pretrained weights')
            # model = inflate_imagenet_weights(model, args.net)
        elif args.test == 'kinetics':
            print('=> using kinetics pretrained weights')
            assert args.net == 'resnet18'
            checkpoint = torch.load('../backbone/resnet-18-kinetics.pth', map_location=torch.device('cpu'))
            model = neq_load_kinetics(model, checkpoint['state_dict'])
            print("=> loaded kinetics pretrained checkpoint")
        else: print("=> no checkpoint found at '{}'".format(args.test))
        crop_size = 96 if args.dataset == 'sth' else 224
        if (args.dataset == 'ucf101') or (args.dataset == 'hmdb51'):
            transform = transforms.Compose([
                # RandomHorizontalFlip(consistent=True),
                # RandomRotation(consistent=False),
                CenterCrop(size=180, consistent=True),
                Scale(size=(args.img_dim,args.img_dim)),
                # RandomGray(consistent=False, p=0.5),
                # ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                ToTensor(),
                Normalize()
            ])
        elif args.dataset == 'k400-full':
            transform = transforms.Compose([
                RandomSizedCrop(size=args.img_dim, consistent=True, p=0.0),
                # RandomHorizontalFlip(consistent=True),
                # RandomGray(consistent=False, p=0.5),
                # ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
                ToTensor(),
                Normalize()
            ])
        test_loader = get_data(transform, 'test') # note!
        test_loss, test_acc = test(test_loader, model, criterion, args.vis_embed)

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            # model.load_state_dict(checkpoint['state_dict'])
            if ('MH' not in args.pretrain) and args.multi_hypo:
                assert args.single2multi, 'Please specify if convert single- to multi-Hypothesis'
                model = single2multi(model, checkpoint['state_dict'])
            else: # normal state
                model = neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})"
                  .format(args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    # load data
    if args.dataset == 'ucf101':
        transform = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            # RandomRotation(consistent=False),
            # CenterCrop(size=224),
            # RandomCropWithProb(size=224, p=0.5, consistent=False), # TODO: maybe wrong?
            RandomCrop(size=224, consistent=True),
            Scale(size=(args.img_dim,args.img_dim)),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
    elif args.dataset == 'k400-full': # designed for kinetics, input 150 - crop to 128
        transform = transforms.Compose([
            # RandomRotation(consistent=True, degree=5, p=0.05),
            RandomSizedCrop(size=args.img_dim, consistent=True, p=1.0),
            RandomHorizontalFlip(consistent=True),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])

    train_loader = get_data(transform, 'train')
    val_loader = get_data(transform, 'val')

    # setup tools
    global de_normalize; de_normalize = denorm()
    global img_path; img_path, model_path = set_path(args)
    global writer_train
    writer_val = SummaryWriter(log_dir=os.path.join(img_path, 'val'))
    writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))

    val_counter = 0

    # main loop
    for epoch in range(args.start_epoch, args.epochs):
        # train
        train_loss, train_acc, train_accuracy_list = train(train_loader, model, criterion, optimizer, epoch)
        
        # skip validation controlled by val frequency
        gap = args.val_freq
        val_counter += 1
        if val_counter < args.val_freq: continue

        # validate
        val_counter = 0
        val_loss, val_acc, val_accuracy_list = validate(val_loader, model, criterion, epoch)

        writer_train.add_scalar('global/loss', train_loss, epoch)
        writer_train.add_scalar('global/accuracy', train_acc, epoch)
        writer_val.add_scalar('global/loss', val_loss, epoch)
        writer_val.add_scalar('global/accuracy', val_acc, epoch)
        writer_train.add_scalar('accuracy/top1', train_accuracy_list[0], epoch)
        writer_train.add_scalar('accuracy/top3', train_accuracy_list[1], epoch)
        writer_train.add_scalar('accuracy/top5', train_accuracy_list[2], epoch)
        writer_val.add_scalar('accuracy/top1', val_accuracy_list[0], epoch)
        writer_val.add_scalar('accuracy/top3', val_accuracy_list[1], epoch)
        writer_val.add_scalar('accuracy/top5', val_accuracy_list[2], epoch)

        # save check_point
        is_best = val_acc > best_acc; best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch+1,
                         'net': args.net,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'iteration': iteration}, 
                         is_best, gap, filename=os.path.join(model_path, 'epoch%s.pth.tar' % str(epoch+1)), keep_all=False)

    print('Training form ep %d to ep %d finished' % (args.start_epoch, args.epochs))

def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, *_) = mask.size()
    # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    (_, NP, SQ, B2, NS, _) = mask.size() # SQ = squared pooled size
    return target, (B, B2, NS, NP, SQ)

def train(data_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.train()
    global iteration

    for idx, input_seq in enumerate(data_loader):
        tic = time.time()
        input_seq = input_seq.to(cuda)
        B = input_seq.size(0)
        h0 = model.module.init_hidden(B)
        [score_, mask_] = model(input_seq, h0)
        # visualize
        if (iteration == 0) or (iteration == args.print_freq):
            if B > 2: input_seq = input_seq[0:2,:]
            writer_train.add_image('input_seq',
                                   de_normalize(
                                       vutils.make_grid(
                                           input_seq.transpose(2,3).contiguous().view(-1,3,args.img_dim,args.img_dim).cpu(), 
                                           nrow=args.num_seq*args.seq_len)),
                                   iteration)
        del input_seq
        
        if idx == 0: target_, (_, B2, NS, NP, SQ) = process_output(mask_)

        # score is a 6d tensor: [B, P, SQ, B, N, SQ]
        score_flattened = score_.view(B*NP*SQ, B2*NS*SQ)
        target_flattened = target_.view(B*NP*SQ, B2*NS*SQ)

        target_flattened = target_flattened.argmax(dim=1)
        loss = criterion(score_flattened, target_flattened)
        acc, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1,3,5))

        accuracy_list[0].update(acc.item(),  B)
        accuracy_list[1].update(top3.item(), B)
        accuracy_list[2].update(top5.item(), B)

        losses.update(loss.item(), B)
        accuracy.update(acc.item(), B)

        del score_

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del loss

        if idx % args.print_freq == 0:
            if args.verbose: print(score_flattened.contiguous().view(-1)[0:10])
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f} ({loss.local_avg:.4f})\t'
                  'Acc: top1 {3:.4f}; top3 {4:.4f}; top5 {5:.4f} T:{6:.2f}\t'.format(
                   epoch, idx, len(data_loader), acc, top3, top5, time.time()-tic,
                   loss=losses, acc=accuracy))

            # total_weight = 0.0
            # for m in model.parameters():
            #     total_weight += m.norm(2).data
            # print('Total weight: %f' % total_weight)

            writer_train.add_scalar('local/loss', losses.val, iteration)
            writer_train.add_scalar('local/accuracy', accuracy.val, iteration)

            iteration += 1

    return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]


def validate(data_loader, model, criterion, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    confusion_mat = ConfusionMeter(2)
    model.eval()

    with torch.no_grad():
        for idx, input_seq in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq = input_seq.to(cuda)
            B = input_seq.size(0)
            h0 = model.module.init_hidden(B)
            [score_, mask_] = model(input_seq, h0)
            del input_seq

            if idx == 0: target_, (_, B2, NS, NP, SQ) = process_output(mask_)

            # [B, P, SQ, B, N, SQ]
            score_flattened = score_.view(B*NP*SQ, B2*NS*SQ)
            target_flattened = target_.view(B*NP*SQ, B2*NS*SQ)
            target_flattened = target_flattened.argmax(dim=1)

            loss = criterion(score_flattened, target_flattened)
            acc, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1,3,5))

            losses.update(loss.item(), B)
            accuracy.update(acc.item(), B)

            accuracy_list[0].update(acc.item(),  B)
            accuracy_list[1].update(top3.item(), B)
            accuracy_list[2].update(top5.item(), B)

    print('[{3}/{4}] Loss {loss.local_avg:.4f}\t'
          'Acc: top1 {0:.4f}; top3 {1:.4f}; top5 {2:.4f} \t'.format(
            *[i.avg for i in accuracy_list], epoch, args.epochs, loss=losses))
    return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]


def get_data(transform, mode='train'):
    if args.unit_test: print('* Unit Test Mode *')
    print('Loading data for "%s" ...' % mode)
    if args.use_training_data: print('force to use training set'); mode = 'train'
    if args.dataset == 'k400-full':
        use_big_K400 = args.img_dim > 140
        dataset = Kinetics400_full_3d(mode=mode,
                              transform=transform,
                              seq_len=args.seq_len,
                              num_seq=args.num_seq,
                              downsample=5,
                              unit_test=args.unit_test,
                              big=use_big_K400)
    elif args.dataset == 'ucf101':
        dataset = UCF101_3d(mode=mode,
                         transform=transform,
                         seq_len=args.seq_len,
                         num_seq=args.num_seq,
                         downsample=args.ds,
                         unit_test=args.unit_test)
    else:
        raise ValueError('dataset not supported')

    my_sampler = data.RandomSampler(dataset)
    shuffle = False
    drop_last = True

    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=shuffle,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=drop_last)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=shuffle,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=drop_last)
    elif mode == 'test':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=shuffle,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=drop_last)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader

def set_path(args):
    if args.resume: exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = 'log_{args.prefix}/{0}{args.dataset}-{args.img_dim}_{1}{2}_{args.model}_{args.loss}_\
bs{args.batch_size}_lr{3}_seq{args.num_seq}_pred{args.pred_step}_len{args.seq_len}_ds{args.ds}_fc{args.num_fc}_\
p{args.pooled_size}_train-{args.train_what}{4}{5}'.format(
                    'ut/' if args.unit_test else '', \
                    '3d-r%s' % args.net[6::] if args.network_type=='3d' else 'r%s' % args.net[6::], \
                    'F-' if args.full_resnet else '',
                    args.old_lr if args.old_lr is not None else args.lr, \
                    '_'+args.bn if args.bn else '', \
                    '_pt=%s' % (args.pretrain.split('/')[-3]) if args.pretrain else '', \
                    args=args)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    return img_path, model_path

if __name__ == '__main__':
    main()
