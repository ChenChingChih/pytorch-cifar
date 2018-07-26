﻿ -*- coding: cp950 -*-
Python 2.7.8 (default, Jun 30 2014, 16:03:49) [MSC v.1500 32 bit (Intel)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> '''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
-*- coding:UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
# tip 程式要由後往前看!順著機器思考的步驟去，先了解大架構(函數的目的)後再進到函數內部的細節

# Chunk 3 定義BasicBlock
class BasicBlock(nn.Module):
    expansion = 1 # expansion意思是什麼?是在Bottleneck時才需要調整

    def __init__(self, in_planes, planes, stride=1): #in_plane為input channel
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) #定義Conv1->in_planes就是input channel代表放入的照片張數，planes就是output channel代表輸出的照片張數
        self.bn1 = nn.BatchNorm2d(planes) # 定義Batch1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) #定義Conv2
        self.bn2 = nn.BatchNorm2d(planes) # 定義Batch2

        self.shortcut = nn.Sequential() # 定義shortcut??想問nn.Sequential()設定函數的方式。是不是因為已縮排就表示下面的code在描述這一行呢?
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False), #self.expansion*planes(16層變成32層)，stride=stride(32層變成16層)
                nn.BatchNorm2d(self.expansion*planes)
            )
# Chunk 3/

# Chunk 4 forward是實際上有"執行"的地方!
    def forward(self, x): # 見basic block的流程圖! ?待確認每個位子代表什麼?
        out = F.relu(self.bn1(self.conv1(x))) # 方框一，先做convolution再做batch最後做ReLu(把self.conv1(x)的值代入self.bn1中，再把self.bn1的值代入F.relu中)
        out = self.bn2(self.conv2(out)) # 方框二，上面的值代入函式中
        out += self.shortcut(x) # shortcut
        out = F.relu(out) # out再做ReLu
        return out
# Chunk 4/

# 刪除Bottleneck

# Chunk 2
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64 # 1_此處要和2,3的一起修改!

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) #64為output channel, padding補0
        self.bn1 = nn.BatchNorm2d(64) # 2_此處要和1一起修改!
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) # 3_此處要和1的一起修改!
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) # 4_此處要和3成倍數關係!
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) # 5_此處要和4成倍數關係!
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) 因為只有RGB，所以只會用到三個
        self.linear = nn.Linear(512*block.expansion, num_classes) # 6_此處要和5成4倍數關係!

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides: # 在strides中抽出的stride要進行下列的動作
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
# Chunk 2/

# Chunk 5 forward是實際上有"執行"的地方!
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out) 因為只有RGB，所以只會用到三個
        out = F.avg_pool2d(out, 8) # 改成8，output map size由64*8*8變成64*1*1 
        out = out.view(out.size(0), -1) # 由64*1*1變成64
        out = self.linear(out) # 由64*1*1變成10
        return out # 最終結果!
# Chunk 5
    
# Chunk 1
def ResNet20():  
    return ResNet(BasicBlock, [3,3,3]) # 因為只有RGB，所以只會用到三個，並調成number blocks為[3,3,3]
#Chunk 1/

def ResNet56():
    return ResNet(BasicBlock, [9,9,9]) # 因為只有RGB，所以只會用到三個，並調成number blocks為[9,9,9]

def ResNet110():
    return ResNet(BasicBlock, [18,18,18]) # 因為只有RGB，所以只會用到三個，並調成number blocks為[18,18,18]


def test():
    net = ResNet20()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
