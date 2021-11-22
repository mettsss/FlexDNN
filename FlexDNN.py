'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class early_exit_Branch(nn.Module):
    def __init__(self, inp, oup, fc_inp, kernel_size=3, stride=1, relu=False,class_nums=10):
        '''
        inp:输入通道数
        out:输出通道数
        fc_inp: 全连接层输入维度
        '''
        super(early_exit_Branch, self).__init__()
        self.dep_conv = nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )
        self.classifier = nn.Linear(fc_inp, class_nums)
    def forward(self, x):
        x = self.dep_conv(x)
        # print(x.shape)
        out = x.reshape((x.size(0), -1))
        # print(out.shape)
        logits = self.classifier(out)
        return logits

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features1 = nn.Sequential(
            # 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.early_exit1 = early_exit_Branch(128,128,fc_inp=32768,class_nums=10)
        self.features2 = nn.Sequential(
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        # self.early_exit2 = early_exit_Branch(128, 128, fc_inp=32768, class_nums=10)
        self.features3 = nn.Sequential(
            # 7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.features4 = nn.Sequential(
            # 10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=1, stride=1),
        )
        self.classifier = nn.Sequential(
            # 14
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 16
            nn.Linear(4096, num_classes),
        )
        #self.classifier = nn.Linear(512, 10)
    def forward(self, x):
        out = self.features1(x)
        early_out = self.early_exit1(out)
        out = self.features2(out)
        out = self.features3(out)
        out = self.features4(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out,early_out
# 定义当前设备是否支持 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = VGG16().to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)