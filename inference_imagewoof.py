# python inference_imagewoof.py
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

from models import resnet, densenet, googlenet, mobilenet
from resizing_model import classification_model
from utils import progress_bar
from collections import OrderedDict
import matplotlib.pyplot as plt

@hydra.main(config_name="config_run_imagewoof")
def main(cfg: DictConfig):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'


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


    CKPT_ROOT = '/home/xw221/Resizer/'
    global best_acc
    # Data
    print('==> Preparing data..')

    transform_test = transforms.Compose([
        transforms.Resize((cfg.data.image_size,cfg.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    testset = torchvision.datasets.ImageFolder(
        root=cfg.trainer.data_path+'val', transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=8)



    classes = ('Australian terrier', 'Border terrier', 'Samoyed', 'Beagle', 'Shih-Tzu', 
                'English foxhound', 'Rhodesian ridgeback', 'Dingo', 'Golden retriever', 'Old English sheepdog')

    # Model
    print('==> Building model..')
    net = classification_model(cfg)
    if net.resizer_model is not None:
        print('Model size: {:.2f}M'.format(sum(p.numel() for p in net.parameters()) / 1e6))
        print('Resizer model size: {:.2f}M'.format(sum(p.numel() for p in net.resizer_model.parameters()) / 1e6))
        print('Base model size: {:.2f}M'.format(sum(p.numel() for p in net.base_model.parameters()) / 1e6))
        state_dict = torch.load(CKPT_ROOT + 'ckpt/imagewoof2/resnet50_resizer/checkpoint.pth')
        state_dict = state_dict['net']
        new_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_dict[name] = v
        net.load_state_dict(new_dict)
        # net.load_state_dict(state_dict)
    else:
        print('Base model size: {:.2f}M'.format(sum(p.numel() for p in net.base_model.parameters()) / 1e6))

    net = net.to(device)
    resizer = net.resizer_model

    criterion = nn.CrossEntropyLoss()

    def save_images(samples, sample_dir, sample_name, offset=0, nrows=0):
        if nrows == 0:
            bs = samples.shape[0]
            nrows = int(bs**.5)
        if offset > 0:
            sample_name += '_' + str(offset)
        save_path = os.path.join(sample_dir, sample_name + '.png')
        torchvision.utils.save_image(samples.cpu(), save_path, nrow=nrows, normalize=True) 
        
    
    # Training
    def inference():
        # global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        seen_labels = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                for i in range (0, 100):
                    target0 = targets[i]
                    if classes[target0] not in seen_labels:
                        seen_labels.append(classes[target0])
                    else:
                        continue

                    print(target0)
                    input0 = inputs[i].to(device)
                    input0 = torch.unsqueeze(input0, 0)
                    target0 = targets[i].to(device)
                    target0 = torch.unsqueeze(target0, 0)
                    
                    output0 = net(input0)
                    loss = criterion(output0, target0)

                    test_loss += loss.item()
                    _, predicted = output0.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(target0).sum().item()

                    mid_output0 = resizer(input0)

                    acc = 100.*correct/total
                    print('test loss: {:.3f}, test acc: {:.3f}'.format(test_loss/(batch_idx+1), acc))

                    save_images(input0, CKPT_ROOT, sample_name=classes[target0] + '_imagewoof')
                    save_images(mid_output0, CKPT_ROOT, sample_name=classes[target0] + '_imagewoof_mid_output')
                    if len(seen_labels) == 10:
                        return
                
                return

    
    inference()
    # input0, mid_output0 = inference()
    # print(input0.shape)
    # print(mid_output0.shape)


    

if __name__ == "__main__":
    main()