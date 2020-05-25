

from utils import load_model
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils import progress_bar
import os
import torchvision
import torchvision.transforms as transforms
import numpy as np

best_acc = 0
train_batch=320
test_batch=320
# from model import DLASeg
# transform_train = transforms.Compose([
#     transforms.Resize((64,64)),
#     transforms.RandomCrop(64, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
#
# transform_test = transforms.Compose([
#     transforms.Resize((64,64)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
#
# model = DLASeg('dla{}'.format(34), {'hm': 80, 'wh': 2, 'reg': 2, 'cla': 1},
#                pretrained=True,
#                down_ratio=4,
#                final_kernel=1,
#                last_level=5,
#                head_conv=256,batch_size=train_batch).cuda()
from networks.dlav0 import  DLASeg
transform_train = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
model = DLASeg('dla{}'.format(34), {'hm': 80, 'wh': 2, 'reg': 2, 'cla': 1},
               pretrained=True,
               down_ratio=4,
               head_conv=256,batch_size=train_batch).cuda()


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch, shuffle=True, num_workers=32)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers=32)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# model = load_model(model, '/home/rvlab/PycharmProjects/CenterNet/models/ctdet_coco_dlav0_1x.pth')
# for param in model.parameters():
#     param.requires_grad = False
#     print(param.requires_grad)
# for param in model.cla.parameters():
#     param.requires_grad = True
#     print(param.requires_grad)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.01,
            weight_decay=1e-5,momentum=0.9,nesterov=True)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs=torch.nn.functional.softmax(outputs[0]['cla'])
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = model(inputs)
            outputs = torch.nn.functional.softmax(outputs[0]['cla'])
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
for epoch in range(200):
        train(epoch)
        test(epoch)
