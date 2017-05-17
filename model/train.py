__author__ = 'SherlockLiao'

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import ImageFolder
from model import InceptionResnetV2
import os
import time

# hyperparameters
batch_size = 8
learning_rate = 1e-3
num_epoch = 10

img_transform = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'valid': transforms.Compose([
        transforms.Scale(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

root_path = '/home/sherlock/Documents/distracted_driver_detection'
train_path = os.path.join(root_path, 'data/imgs/train')
valid_path = os.path.join(root_path, 'data/imgs/valid')
dataset = {
    'train': ImageFolder(train_path, transform=img_transform['train']),
    'valid': ImageFolder(valid_path, transform=img_transform['valid'])
}

dataloader = {
    'train': DataLoader(dataset['train'], batch_size=batch_size,
                        shuffle=True, num_workers=4),
    'valid': DataLoader(dataset['valid'], batch_size=batch_size,
                        num_workers=4)
}

data_size = {
    'train': len(dataset['train']),
    'valid': len(dataset['valid'])
}

data_classes = dataset['train'].classes

model = InceptionResnetV2(len(data_classes))
model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    print('{}/{}'.format(epoch+1, num_epoch))
    print('*'*10)
    print('Train')
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    since = time.time()
    for i, data in enumerate(dataloader['train'], 1):
        img, label = data
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        # forward
        out = model(img)
        loss = criterion(out, label)
        _, pred = torch.max(out, 1)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0] * label.size(0)
        num_correct = torch.sum(pred == label)
        running_acc += num_correct.data[0]
        if i % 50 == 0:
            print('Loss: {:.6f}, Acc: {:.4f}'.format(
                                            running_loss / (i * batch_size),
                                            running_acc / (i * batch_size)))
    running_loss /= data_size['train']
    running_acc /= data_size['train']
    elips_time = time.time() - since
    print('Loss: {:.6f}, Acc: {:.4f}, Time: {:.0f}s'.format(
                                                        running_loss,
                                                        running_acc,
                                                        elips_time))
    print('Validation')
    model.eval()
    num_correct = 0.0
    total = 0.0
    eval_loss = 0.0
    for data in dataloader['val']:
        img, label = data
        img = Variable(img, volatile=True).cuda()
        label = Variable(label, volatile=True).cuda()
        out = model(img)
        _, pred = torch.max(out.data, 1)
        loss = criterion(out, label)
        eval_loss += loss.data[0] * label.size(0)
        num_correct += (pred.cpu() == label.data.cpu()).sum()
        total += label.size(0)
    print('Loss: {:.6f} Acc: {:.6f}'.format(eval_loss / total,
                                            num_correct / total))
    print()
print('Finish Training!')
print()

save_path = './model.pth'
torch.save(model.state_dict(), save_path)
