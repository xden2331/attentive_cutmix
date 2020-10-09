# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet as RN
import pyramidnet as PYRM
import utils
import numpy as np
from attention_util import AttentiveInputTransform, AttentiveTargetTransform

import warnings

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

best_err1 = 100
best_err5 = 100

net_type = 'resnet'
workers = 4  # number of data loading workers
epochs = 90  # number of total epochs to run
batch_size = 128
lr = 0.1
momentum = 0.9
weight_decay = 1e-4
print_freq = 1  # print frequency
depth = 32
bottleneck = True  # to use basicblock for CIFAR datasets
dataset = 'imagenet'  # [cifar10, cifar100, imagenet]
verbose = True
alpha = 300  # number of new channel increases per depth
expname = 'no_pretrained_cutmix'  # name of experiment
beta = 0  # hyperparameter beta
cutmix_prob = 0  # cutmix probability
train_cutmix = True
k = 6
grid_count = 64
pretrained = 'path/to/pretrained/folder'

best_err1 = 100
best_err5 = 100


def main():
    global best_err1, best_err5, grid_count

    if dataset.startswith('cifar'):
        grid_count = 64
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        ori_dataset = datasets.CIFAR10 if dataset == 'cifar10' else datasets.CIFAR100
        placeholder = {}

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            AttentiveInputTransform('cifar', ori_dataset(
                './data', train=True, download=True), placeholder, k=k),
            transforms.ToTensor(),
            normalize,
        ])

        transform_train_target = transforms.Compose([
            AttentiveTargetTransform(placeholder),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        if dataset == 'cifar100':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('./data', train=True, download=True, transform=transform_train, target_transform=transform_train_target), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('./data', train=False,
                                  transform=transform_test),
                batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
            numberofclass = 100
        elif dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('./data', train=True, download=True, transform=transform_train, target_transform=transform_train_target), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('./data', train=False,
                                 transform=transform_test),
                batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(dataset))
    elif dataset == 'imagenet':
        grid_count = 49
        traindir = os.path.join('/home/data/ILSVRC/train')
        valdir = os.path.join('/home/data/ILSVRC/val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        jittering = utils.ColorJitter(brightness=0.4, contrast=0.4,
                                      saturation=0.4)
        lighting = utils.Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[[-0.5675, 0.7192, 0.4009],
                                          [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                jittering,
                lighting,
                normalize,
            ]))

        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(
                train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)
        numberofclass = 1000
    else:
        raise Exception('unknown dataset: {}'.format(dataset))

    print("=> creating model '{}'".format(net_type))
    if net_type == 'resnet':
        model = RN.ResNet(dataset, depth, numberofclass,
                          bottleneck)  # for ResNet
    elif net_type == 'pyramidnet':
        model = PYRM.PyramidNet(dataset, depth, alpha, numberofclass,
                                bottleneck)
    else:
        raise Exception('unknown network architecture: {}'.format(net_type))

    model = torch.nn.DataParallel(model).cuda()

    # else:
    #     raise Exception(
    #         "=> no checkpoint found at '{}'".format(pretrained))

    print(model)
    print('the number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay, nesterov=True)

    start_epoch = 0
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(pretrained))

        start_epoch = checkpoint['epoch']+1
        net_type = checkpoint['net_type']
        best_err1 = checkpoint['best_err1']
        best_err5 = checkpoint['best_err5']
        optimizer = optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    for epoch in range(start_epoch, epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        err1, err5, val_loss = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5

        print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        save_checkpoint({
            'epoch': epoch,
            'arch': net_type,
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
            'best_err5': best_err5,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()
        if train_cutmix:
            # generate mixed sample
            target_a, target_b = target[:, 0], target[:, 1]
            output = model(input)
            lam = k/grid_count
            target_a_loss = criterion(output, target_a) * lam
            target_b_loss = criterion(output, target_b) * (1-lam)
            loss = target_a_loss + target_b_loss
        else:
            # compute output
            output = model(input)
            loss = criterion(output, target[:, 0])

        # measure accuracy and record loss
        err1, err5 = accuracy(
            output.detach(), target[:, 0].detach(), topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 and verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                      epoch, epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
        epoch, epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.detach(), target.detach(), topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 and verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                      epoch, epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' %
                        (expname) + 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if dataset.startswith('cifar'):
        lr = lr * (0.1 ** (epoch // (epochs * 0.5))) * \
            (0.1 ** (epoch // (epochs * 0.75)))
    elif dataset == ('imagenet'):
        if epochs == 300:
            lr = lr * (0.1 ** (epoch // 75))
        else:
            lr = lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
