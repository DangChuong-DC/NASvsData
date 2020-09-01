import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import sklearn.metrics as sk
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import warnings

from utils import *
from models.network import Network


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--finetune_lr', type=float, default=0.001, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--epochs', type=int, default=1, help='num of fine-tuning epochs')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--save', type=str, default='./CheckPoints/', help='experiment path')
parser.add_argument('--load_at', type=str, default='./CheckPoints/supernet-try-20200831-191439/supernet_weights.pt', help='Checkpoint path.')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--tmp_data_dir', type=str, default='/home/anhcda/Storage/OoD_NAS/data/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--cifar100', action='store_true', default=False, help='search with cifar100 dataset')
parser.add_argument('--fine_tune', action='store_true', default=False, help='Specify if fine-tuning is done.')

args, unparsed = parser.parse_known_args()

if args.cifar100:
    CIFAR_CLASSES = 100
    data_folder = 'cifar-100-python'
else:
    CIFAR_CLASSES = 10
    data_folder = 'cifar-10-batches-py'


def main():
    if not torch.cuda.is_available():
        print('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    print("args = %s", args)
    print("unparsed args = %s", unparsed)

    # prepare dataset
    if args.cifar100:
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.cifar100:
        train_data = dset.CIFAR100(root=args.tmp_data_dir, train=True, download=False, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.tmp_data_dir, train=False, download=False, transform=valid_transform)
    else:
        train_data = dset.CIFAR10(root=args.tmp_data_dir, train=True, download=False, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.tmp_data_dir, train=False, download=False, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)


    # build Network
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    supernet = Network(
        args.init_channels, CIFAR_CLASSES, args.layers
    )
    supernet.cuda()

    ckpt = torch.load(args.load_at)
    print(args.load_at)
    supernet.load_state_dict(ckpt)
    supernet.generate_share_alphas()

    alphas = supernet.cells[0].ops_alphas
    print(alphas)
    out_dir = './results/{}/eval_out/{}'.format(args.load_at.split('/')[2], args.seed)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    torch.save(alphas, os.path.join(out_dir, 'alphas.pt'))
    with open(os.path.join(out_dir, 'alphas.txt'), 'w') as f:
        for i in alphas.cpu().detach().numpy():
            for j in i:
                f.write('{:d}'.format(int(j)))
            f.write('\n')

    # Getting subnet according to sample alpha
    subnet = supernet.get_sub_net(alphas)

    init_valid_acc, _ = infer(valid_queue, subnet, criterion)
    print('Initial Valid Acc {:.2f}'.format(init_valid_acc))

    if args.fine_tune:
        if args.cifar100:
            weight_decay = 5e-4
        else:
            weight_decay = 3e-4

        # Fine tuning whole network:
        subnet = supernet.get_sub_net(alphas)
        optimizer = torch.optim.SGD(
            subnet.parameters(),
            args.finetune_lr,
            momentum=args.momentum,
            weight_decay=weight_decay,
        )

        for epoch in range(args.epochs):
            # scheduler.step()
            print('epoch {} lr {:.4f}'.format(epoch, args.finetune_lr))

            train_acc, _ = train(train_queue, subnet, criterion, optimizer)
            print('train_acc {:.2f}'.format(train_acc))

            whole_valid_acc, _ = infer(valid_queue, subnet, criterion)
            print('valid_acc after whole fine-tune {:.2f}'.format(whole_valid_acc))

        # Fine-tuning only classifier:
        subnet = supernet.get_sub_net(alphas)
            # Freezing other weights except classifier:
        for name, param in subnet.named_parameters():
            if not 'classifier' in name:
                param.requires_grad_(requires_grad=False)

        optimizer = torch.optim.SGD(
            subnet.classifier.parameters(),
            args.finetune_lr,
            momentum=args.momentum,
            weight_decay=weight_decay,
        )

        for epoch in range(args.epochs):
            # scheduler.step()
            print('epoch {} lr {:.4f}'.format(epoch, args.finetune_lr))

            train_acc, _ = train(train_queue, subnet, criterion, optimizer)
            print('train_acc {:.2f}'.format(train_acc))

            part_valid_acc, _ = infer(valid_queue, subnet, criterion)
            print('valid_acc after fine-tuning classifier {:.2f}'.format(part_valid_acc))

        with open(os.path.join(out_dir, 'results.txt'), 'w') as f:
            f.write('-'.join([str(init_valid_acc), str(whole_valid_acc), str(part_valid_acc)]))

    if not args.fine_tune:
        with open(os.path.join(out_dir, 'results.txt'), 'w') as f:
            f.write(str(init_valid_acc))


def train(train_queue, model, criterion, optimizer):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.train()

    for step, (inp, target) in enumerate(train_queue):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        model.zero_grad()

        logits = model(inp)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
        n = inp.size(0)
        objs.update(loss.clone().item(), n)
        top1.update(prec1.clone().item(), n)

        if ((step + 1) % args.report_freq) == 0:
            print('Train Step: {:3d} Objs: {:.4f} Acc: {:.2f}'.format(step + 1, objs.avg, top1.avg))

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.eval()

    for step, (inp, target) in enumerate(valid_queue):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            logits = model(inp)
            loss = criterion(logits, target)

        prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
        n = inp.size(0)
        objs.update(loss.clone().item(), n)
        top1.update(prec1.clone().item(), n)

    return top1.avg, objs.avg


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    print('Running time: {}s.'.format(duration))
