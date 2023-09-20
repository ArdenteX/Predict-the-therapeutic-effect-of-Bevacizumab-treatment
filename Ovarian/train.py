from ResNet18 import ResNet18
from torchvision.datasets import ImageFolder
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 15
pre_epoch = 0
BATCH_SIZE = 48
LR = 0.1

transform_train = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

train = ImageFolder('C:\\dataset\\Ovarian_After_Split\\', transform=transform_train)
trainloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)


net = ResNet18(2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

total_loss = []
total_acc = []

for epoch in range(pre_epoch, EPOCH):
    print('\nEpoch: %d' % (epoch + 1))
    net.train()
    sum_loss = 0.0
    total = 0.0
    correct = 0.0
    for i, data in enumerate(trainloader, 0):

        length = len(trainloader)
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        schedular.step()

        sum_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()

        print('\r[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
              % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total), end='')
    total_loss.append(sum_loss)
    total_acc.append(correct / total)


torch.save(net.state_dict(), 'D:\\Resource\\models\\model_2.pth')

acc_np = np.array(total_acc)
loss_np = np.array(total_loss)

np.save('D:\\Resource\\modelEvaluation\\acc_4', acc_np)
np.save('D:\\Resource\\modelEvaluation\\loss_4', loss_np)
