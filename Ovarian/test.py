import itertools
from matplotlib import pyplot as plt
from ResNet18 import ResNet18
from ResNet18 import mpncovresnet50
from resnet import resnet18
from ResNet18 import ResBlock
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from tqdm.contrib import tenumerate
from ECANET.eca_resnet import eca_resnet50
from ECANET.eca_resnet import eca_resnet18
import random
import math
from Utils.View import visualize_network, init_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = eca_resnet18(num_classes=2)
init_weights(model)
model = torch.nn.DataParallel(model)
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, 2)
model.to(device)
model.load_state_dict(torch.load("D:\Resource\models\\best\model_best_ResNet-Net-18.pth"))

model.eval()

transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_set = ImageFolder("D:\\Dataset\\test\\", transform=transforms)
t = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=64)

error_count = 0
labels = []
predicts = []
with torch.no_grad():
    for i, data in tenumerate(t, 0):
        images, label = data
        images, label = images.to(device), label.to(device)

        output = model(images)

        _, predict = torch.max(output, 1)
        labels.extend(label.cpu().numpy().tolist())
        predicts.extend(predict.cpu().numpy().tolist())

        # print('\rLabel -----> {} Predict -----> {} '.format(label.item(), predict.item()), end='')
        # if label.item() != predict.item():
        #     error_count += 1


print(classification_report(labels, predicts))

for la, p in zip(labels, predicts):
    if la != p:
        error_count += 1

accuracy = (len(labels) - error_count) / len(labels) * 100

c_m = confusion_matrix(labels, predicts)
TP = c_m[0][0]
FP = c_m[1][0]
FN = c_m[0][1]
TN = c_m[1][1]

sensitively = TP / (TP + FN)


def plot_confusion_metrix(cm, classes, normalize=False, title='Confusion Metrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        print("Confusion Metrix without normalize")

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


cm_plot_label =['Effect', 'Invalid']
plot_confusion_metrix(c_m, cm_plot_label, True)

# n = np.array([1, 2])
# b = np.append(n, [1, 3 ])


# target = "C:\\dataset\\Validation\\Efficient\\"
# target_i = "C:\\dataset\\Validation\\Invalid\\"
# target_test_e = "D:\\Dataset\\test\\Efficient\\"
# target_test_i = "D:\\Dataset\\test\\Invalid\\"
# src = "C:\\dataset\\Ovarian_After_Split\\Efficient\\"
# src_i = "C:\\dataset\\Ovarian_After_Split\\Invalid\\"
#
#
# def move(s, t):
#     files = os.listdir(s)
#     val_size = int(len(files) * 0.5)
#     n = random.sample(range(0, len(files)), val_size)
#     for i in n:
#         print(s+files[i])
#         shutil.move(s+files[i], t+files[i])


# move(target, target_test_e)
# move(target_i, target_test_i)
#
#
# k = int(abs((math.log2(256) / 2) + 1 / 2))
# out = k if k % 2 else k + 1
#
# files = os.listdir(target_test_e)
# val_size = int(len(files) * 0.5)
#
# aa = np.array([1, 1, 0, 2, 1])
# b = aa.reshape((aa.size, 1))
#
# g = torch.arange(6)
# g = g.reshape((len(g), 1))
# g.to(dtype=torch.float32)
