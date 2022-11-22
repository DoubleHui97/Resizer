# python run.py
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os, sys, time, random
import argparse

from models import resnet, densenet, googlenet, mobilenet
from resizing_model import classification_model
from utils import progress_bar

@hydra.main(config_name="config_run")
def main(cfg: DictConfig):
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

    # parser = argparse.ArgumentParser(description='CIFAR10 Training')
    # parser.add_argument('--arch', default='resnet50', help='recog model architecture')
    # parser.add_argument('--lr', default=0.05, type=float, help='initial learning rate')
    # parser.add_argument('--wd', default=5e-4, type=float, help='weight decay coefficient')
    # parser.add_argument('--epoch', default=200, type=int, help='total training epochs')
    # parser.add_argument('--scheduler', default='multisteplr', type=str, 
    #                     help='choose from "multisteplr", "cosineannealinglr", "steplr"')
    # parser.add_argument('--milestone', default=[100, 150], help='lr decay epochs')
    # # parser.add_argument('--pretrain', action='store_true', help='using pretrained model')
    # parser.add_argument('--resume_path', '-r', type=str, default='',
    #                     help='ckpt path of resumed model')
    # parser.add_argument('--checkpoint_path', type=str, default='', 
    #                     help='finetune checkpoint directory, default=ckpt/finetune_<modelName><time>/')
    # parser.add_argument('--data_path', type=str, default='/home/xw221/data/')
    # parser.add_argument('--resizer', action='store_true', help='add resizer to the recognition model')
    # parser.add_argument('-seed', type=int, default=42, help='random seed')
    # args = parser.parse_args()

    # for arg in vars(args):
    #     print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using {}'.format(device))

    def seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    seed_torch(cfg.trainer.seed)


    CKPT_ROOT = '/home/xw221/course_projects/ECE588/'
    global best_acc
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=cfg.trainer.data_path, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(
        root=cfg.trainer.data_path, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=8)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = classification_model(cfg)
    if net.resizer_model is not None:
        print('Model size: {:.2f}M'.format(sum(p.numel() for p in net.parameters()) / 1e6))
        print('Resizer model size: {:.2f}M'.format(sum(p.numel() for p in net.resizer_model.parameters()) / 1e6))
        print('Base model size: {:.2f}M'.format(sum(p.numel() for p in net.base_model.parameters()) / 1e6))
    else:
        print('Base model size: {:.2f}M'.format(sum(p.numel() for p in net.base_model.parameters()) / 1e6))

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if cfg.trainer.resume_path != '':
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(cfg.trainer.resume_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(cfg.trainer.resume_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=cfg.trainer.lr,
                        momentum=0.9, weight_decay=cfg.trainer.wd)

    if cfg.trainer.scheduler == 'multisteplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.trainer.milestone)
    elif cfg.trainer.scheduler == 'cosineannealinglr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.trainer.epoch)
    elif cfg.trainer.scheduler == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.94)
    else:
        print('Scheduler {} not supported!'.format(cfg.trainer.scheduler))
        sys.exit(0)

    print('==> Creating ckpt path..')
    if cfg.trainer.checkpoint_path != '':
        if not os.path.exists(cfg.trainer.checkpoint_path):
            os.mkdir(cfg.trainer.checkpoint_path)
        checkpoint_path = cfg.trainer.checkpoint_path
    else:
        checkpoint_path = os.path.join(CKPT_ROOT + 'ckpt/cifar10/' + str(cfg.trainer.arch) + '_'
                                        + time.strftime('%m%d%y_%H%M%S', time.localtime()) + '/')
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
    print('model saved at {}'.format(checkpoint_path))

    # Training
    def train(epoch):
        lr = optimizer.param_groups[-1]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch, cfg.trainer.epoch, lr))
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        print('train loss: {:.3f}, train acc: {:.3f}'.format(train_loss/(batch_idx+1), 100.*correct/total))


    def test(epoch, best_acc):
        # global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        acc = 100.*correct/total
        print('test loss: {:.3f}, test acc: {:.3f}'.format(test_loss/(batch_idx+1), acc))
        # Save checkpoint.
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + 'checkpoint.pth')
            best_acc = acc
        
        return best_acc

    best_acc = 0.0
    for epoch in range(cfg.trainer.epoch):
        train(epoch)
        best_acc = test(epoch, best_acc)
        scheduler.step()

    print('\nBest acc:{}'.format(best_acc))

if __name__ == "__main__":
    main()