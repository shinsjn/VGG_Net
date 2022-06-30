import torch
import torchvision
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.utils.data import Subset

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit


BATCH_SIZE = 16

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
#train
train_set = torchvision.datasets.STL10('./data',split="train", download="True",transform=transform)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE,shuffle="True")

#split_test&val
test_val_set = torchvision.datasets.STL10('./data',split="test", download="True", transform=transform)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

indices = list(range(len(test_val_set)))
y_test0 = [y for _, y in test_val_set]

for test_index, val_index in sss.split(indices,y_test0):
    print(len(val_index), len(test_index))
#val
val_set = Subset(test_val_set,val_index)
val_loader = torch.utils.data.DataLoader(val_set,batch_size=BATCH_SIZE, shuffle="True")

#test
test_set = Subset(test_val_set,test_index)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=BATCH_SIZE, shuffle="False")

VGG_types = {
    'VGG11' : [64, 'M', 128, 'M', 256, 256, 'M', 512,512, 'M',512,512,'M'],
    'VGG13' : [64,64, 'M', 128, 128, 'M', 256, 256, 'M', 512,512, 'M', 512,512,'M'],
    'VGG16' : [64,64, 'M', 128, 128, 'M', 256, 256,256, 'M', 512,512,512, 'M',512,512,512,'M'],
    'VGG19' : [64,64, 'M', 128, 128, 'M', 256, 256,256,256, 'M', 512,512,512,512, 'M',512,512,512,512,'M']
}

class VGGnet(nn.Module):
    def __init__(self,model, in_channels=3, num_classes=10):
        super(VGGnet,self).__init__()
        self.in_channels = in_channels

        self.conv_layers = self.create_conv_laters(VGG_types[model])

        self.FC_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.avg_pool = nn.AvgPool2d(7)
        self.classifier = nn.Linear(512,10)

    def forward(self,x):


        features = self.conv_layers(x)
        x = self.avg_pool(features)
        x = x.view(x.size(0),-1)

        x = self.classifier(x)
        return x, features

    def saveModel(self):
        path = "./VGG_best_model_lr_schedule_ON_01.pth"
        torch.save(self.state_dict(),path)

    def create_conv_laters(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1)),
                    nn.BatchNorm2d(x),
                    nn.ReLU()]
                in_channels = x

            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vggNet = VGGnet('VGG16',in_channels=3, num_classes=10)
vggNet = vggNet.to(device)

summary(vggNet,input_size=(3,224,224),device=device.type)

classes =  ('airplance', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
import torch.optim as optim

criterion = F.cross_entropy
optimizer = optim.Adam(vggNet.parameters(),lr=0.0005)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda epoch: 0.9*epoch,)
best_accuracy=0.0
print("start")
for epoch in range(200):
    running_loss=0.0
    running_val_accuracy=0.0
    total=0

    for i, data in enumerate(train_loader):
        vggNet.train()
        x,y = data
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs , f = vggNet(x)
        loss = criterion(outputs,y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if i % 50 == 49:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0

    print('Finished Training')


    #val data로 좋은 모델 저장
    vggNet.eval()
    val_loss=0.0
    val_correct=0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs, _ = vggNet(images)
            val_loss +=criterion(outputs,labels).item()
            predicted = outputs.max(1,keepdim=True)[1]
            val_correct+=predicted.eq(labels.view_as(predicted)).sum().item()
        val_loss/=len(val_loader.dataset)
        val_accuracy = 100.*val_correct/len(val_loader.dataset)
        print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            val_loss, val_correct, len(val_loader.dataset), val_accuracy))
        print('='*50)

    if best_accuracy < val_accuracy:
        best_accuracy=val_accuracy
        print("SaveModel")
        vggNet.saveModel()



#test
'''
print("*"*50)
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    vggNet.eval()
    for data in test_loader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs, _ = vggNet(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
'''









