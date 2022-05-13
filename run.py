import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import os, argparse
import random
import matplotlib.pyplot as plt
import numpy as np
import hypergrad as hg
from itertools import repeat
from torchvision.datasets import CIFAR10
from datasets.gtsrb import GTSRB

from poi_util import poison_dataset,patching_test
import poi_util

from models.wideresnet import WRN16, WRN28

parser = argparse.ArgumentParser(description='PyTorch cifar10 Example')
parser.add_argument('--gpu', default='0')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_lr', '--blr', type=float, default=10, help='learning rate')
parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'gtsrb'])
parser.add_argument('--model_path', required=True)
# attack params:
parser.add_argument('--target', default=0, type=int, help='target class')
parser.add_argument('--trigger_pattern', '--pattern', default='badnet_grid', 
    choices=['badnet_sq', 'badnet_grid', 'trojan_3x3', 'trojan_8x8', 'trojan_wm', 'l0_inv', 'l2_inv', 'blend', 'smooth', 'sig'], 
    help='pattern of trigger'
)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

save_dir = os.path.join('logs', args.dataset, args.trigger_pattern)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
fp = open(os.path.join(save_dir, 'lr%s-blr%s.txt' % (args.lr, args.batch_lr)), 'a+')

device = 'cuda'
def get_results(model, criterion, data_loader, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.long())

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return correct / total

print('==> Preparing data..')
root = '/ssd1/haotao/datasets'
if args.dataset == 'cifar10':
    trainset = CIFAR10(root, train=True, transform=None)
    testset = CIFAR10(root, train=False, transform=None)
    num_classes = 10
elif args.dataset == 'gtsrb':
    trainset = GTSRB(root, train=True, transform=None)
    testset = GTSRB(root, train=False, transform=None)
    num_classes = 43
x_train, y_train = trainset.data, trainset.targets
x_train = x_train.astype('float32')/255
y_train = np.asarray(y_train)
x_test, y_test = testset.data, testset.targets
x_test = x_test.astype('float32')/255
y_test = np.asarray(y_test)

attack_name = args.trigger_pattern
target_lab = args.target
x_poi_test, y_poi_test= patching_test(x_test, y_test, attack_name, target_lab=target_lab)

y_train = torch.Tensor(y_train.reshape((-1,)).astype(np.int))
y_test = torch.Tensor(y_test.reshape((-1,)).astype(np.int))
y_poi_test = torch.Tensor(y_poi_test.reshape((-1,)).astype(np.int))

x_train = torch.Tensor(np.transpose(x_train,(0,3,1,2)))
x_test = torch.Tensor(np.transpose(x_test,(0,3,1,2)))
x_poi_test = torch.Tensor(np.transpose(x_poi_test,(0,3,1,2)))

N_train_selected = int(0.05*len(x_train))

# test_set = TensorDataset(x_test[5000:],y_test[5000:])
# unl_set = TensorDataset(x_test[:5000],y_test[:5000])
# att_val_set = TensorDataset(x_poi_test[:5000],y_poi_test[:5000])
test_set = TensorDataset(x_test,y_test)
unl_set = TensorDataset(x_train[:N_train_selected],y_train[:N_train_selected])
att_val_set = TensorDataset(x_poi_test,y_poi_test)

#data loader for verifying the clean test accuracy
clnloader = torch.utils.data.DataLoader(
    test_set, batch_size=200, shuffle=False, num_workers=2)

#data loader for verifying the attack success rate
poiloader = torch.utils.data.DataLoader(
    att_val_set, batch_size=200, shuffle=False, num_workers=2)

#data loader for the unlearning step
unlloader = torch.utils.data.DataLoader(
    unl_set, batch_size=100, shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

#define the inner loss L2
def loss_inner(perturb, model_params):
    images = images_list[0].cuda()
    labels = labels_list[0].long().cuda()
#     per_img = torch.clamp(images+perturb[0],min=0,max=1)
    per_img = images+perturb[0]
    per_logits = model.forward(per_img)
    loss = F.cross_entropy(per_logits, labels, reduction='none')
    loss_regu = torch.mean(-loss) +0.001*torch.pow(torch.norm(perturb[0]),2)
    return loss_regu

#define the outer loss L1
def loss_outer(perturb, model_params):
    portion = 0.01
    images, labels = images_list[batchnum].cuda(), labels_list[batchnum].long().cuda()
    patching = torch.zeros_like(images, device='cuda')
    number = images.shape[0]
    rand_idx = random.sample(list(np.arange(number)),int(number*portion))
    patching[rand_idx] = perturb[0]
#     unlearn_imgs = torch.clamp(images+patching,min=0,max=1)
    unlearn_imgs = images+patching
    logits = model(unlearn_imgs)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    return loss

images_list, labels_list = [], []
for index, (images, labels) in enumerate(unlloader):
    images_list.append(images)
    labels_list.append(labels)
inner_opt = hg.GradientDescent(loss_inner, 0.1)


# initialize theta
model = WRN16(num_classes=num_classes).to(device).eval()
outer_opt = torch.optim.Adam(params=model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()
model_path = os.path.join('/ssd1/haotao/BackDoorBlocker_results/normal_training', args.dataset, 'WRN16', 
    args.trigger_pattern, args.model_path, 'latest.pth')
model.load_state_dict(torch.load(model_path)['model'])

ACC = get_results(model, criterion, clnloader, device)
ASR = get_results(model, criterion, poiloader, device)

val_str = 'Original ACC: %.4f, Original ASR: %.4f' % (ACC, ASR)
fp.write(val_str + '\n')
fp.flush()

#inner loop and optimization by batch computing
print("Conducting Defence")

model.eval()
ASR_list = [get_results(model, criterion, poiloader, device)]
ACC_list = [get_results(model, criterion, clnloader, device)]

for round in range(3): 
    batch_pert = torch.zeros_like(x_test[:1], requires_grad=True, device='cuda')
    batch_opt = torch.optim.SGD(params=[batch_pert],lr=args.batch_lr)
   
    for images, labels in unlloader:
        images = images.to(device)
        ori_lab = torch.argmax(model.forward(images),axis = 1).long()
#         per_logits = model.forward(torch.clamp(images+batch_pert,min=0,max=1))
        per_logits = model.forward(images+batch_pert)
        loss = F.cross_entropy(per_logits, ori_lab, reduction='mean')
        loss_regu = torch.mean(-loss) +0.001*torch.pow(torch.norm(batch_pert),2)
        batch_opt.zero_grad()
        loss_regu.backward(retain_graph = True)
        batch_opt.step()

    #l2-ball
    pert = batch_pert * min(1, 10 / torch.norm(batch_pert))

    #unlearn step         
    for batchnum in range(len(images_list)): #T
        outer_opt.zero_grad()
        hg.fixed_point(pert, list(model.parameters()), 5, inner_opt, loss_outer) 
        outer_opt.step()
    
    ACC = get_results(model,criterion,clnloader,device)
    ASR = get_results(model,criterion,poiloader,device)

    val_str = 'Round: %d, Original ACC: %.4f, Original ASR: %.4f' % (round, ACC, ASR)
    print(val_str)
    fp.write(val_str + '\n')
    fp.flush()
fp.close()