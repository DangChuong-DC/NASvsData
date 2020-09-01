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
from torch.utils.tensorboard import SummaryWriter


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--grad_clip', type=float, default=-1, help='gradient clipping')
parser.add_argument('--save', type=str, default='./results/', help='experiment path')
parser.add_argument('--load_at', type=str, default='./CheckPoints/supernet-try-20200831-191439/supernet_weights.pt', help='Checkpoint path.')
parser.add_argument('--super_seed', type=int, default=12345, help='random seed for supernet')
parser.add_argument('--folder', type=int, default=0, help='folder for saving')
parser.add_argument('--ckpt_path', type=str, default='./subnet_exp1/', help='path to save subnet weights')
parser.add_argument('--tmp_data_dir', type=str, default='/home/anhcda/Storage/ANAS/data/', help='temp data dir')
parser.add_argument('--is_cifar100', action='store_true', default=False, help='experiment with cifar100 dataset')

args, unparsed = parser.parse_known_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, '{}/eval_out/{}/subnet_log.txt'.format(args.load_at.split('/')[2], args.folder)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

plot_pth = './results/{}/eval_out/{}/plot/tr/'.format(args.load_at.split('/')[2], args.folder)
writer_tr = SummaryWriter(plot_pth, flush_secs=30)
plot_pth = './results/{}/eval_out/{}/plot/va/'.format(args.load_at.split('/')[2], args.folder)
writer_va = SummaryWriter(plot_pth, flush_secs=30)
global_step = 0

if args.is_cifar100:
    CIFAR_CLASSES = 100
else:
    CIFAR_CLASSES = 10


def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.super_seed)
    cudnn.benchmark = True
    torch.manual_seed(args.super_seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.super_seed)
    logging.info("args = %s", args)
    logging.info("unparsed args = %s", unparsed)

    # prepare dataset
    if args.is_cifar100:
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.is_cifar100:
        train_data = dset.CIFAR100(root=args.tmp_data_dir, train=True, download=False, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.tmp_data_dir, train=False, download=False, transform=valid_transform)
    else:
        train_data = dset.CIFAR10(root=args.tmp_data_dir, train=True, download=False, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.tmp_data_dir, train=False, download=False, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers, drop_last=True)


    # build Network
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    supernet = Network(
        args.init_channels, CIFAR_CLASSES, args.layers, is_bequeath=False
    )
    supernet.cuda()
    supernet.generate_share_alphas()   #This is to prevent supernet alpha attribute being `None` type

    alphas_path = './results/{}/eval_out/{}/alphas.pt'.format(args.load_at.split('/')[2], args.folder)
    logging.info('Loading alphas at: %s' % alphas_path)
    alphas = torch.load(alphas_path)

    subnet = supernet.get_sub_net(alphas)
    logging.info(alphas)

    if args.is_cifar100:
        weight_decay = 5e-4
    else:
        weight_decay = 3e-4
    optimizer = torch.optim.SGD(
        subnet.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    for epoch in range(args.epochs):
        logging.info('epoch {} lr {:.4f}'.format(epoch, scheduler.get_last_lr()[0]))

        train_acc, _ = train(train_queue, subnet, criterion, optimizer)
        logging.info('train_acc {:.2f}'.format(train_acc))

        valid_acc, valid_loss = infer(valid_queue, subnet, criterion)
        writer_va.add_scalar('loss', valid_loss, global_step)
        writer_va.add_scalar('acc', valid_acc, global_step)
        logging.info('valid_acc {:.2f}'.format(valid_acc))
        scheduler.step()

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    utils.save(subnet, os.path.join(args.ckpt_path, 'subnet_{}_weights.pt'.format(args.folder)))

    logging.info('Writting results:')
    out_dir = './results/{}/eval_out/{}/'.format(args.load_at.split('/')[2], args.folder)
    with open(os.path.join(out_dir, 'subnet_results.txt'), 'w') as f:
        f.write(str(valid_acc))


def train(train_queue, model, criterion, optimizer):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.train()

    for step, (inp, target) in enumerate(train_queue):
        global global_step
        global_step += 1
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        model.zero_grad()

        logits = model(inp)
        loss = criterion(logits, target)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
        n = inp.size(0)
        objs.update(loss.clone().item(), n)
        top1.update(prec1.clone().item(), n)
        writer_tr.add_scalar('loss', loss.item(), global_step)
        writer_tr.add_scalar('acc', prec1.item(), global_step)

        if (step + 1) % args.report_freq == 0:
            logging.info('Train Step: %03d Objs: %e Acc: %.2f', step + 1, objs.avg, top1.avg)

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
    logging.info('Running time: {}s.'.format(duration))
