import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2D(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.2, width=50):
        super(Conv2D, self).__init__()
        self.in_size=in_size
        if width==30:
            self.hidden_out = 64
        if width==50:
            self.hidden_out = 576 
        if width==60:
            self.hidden_out = 1024 
        if width==80:
            self.hidden_out = 3136
        if width==100:
            self.hidden_out = 5184
        

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, 16, 5, 2),
            nn.ReLU()
            
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 2),
            nn.ReLU()
            
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 2),
            nn.ReLU()
            
        )
        self.fc_out=nn.Sequential(
            nn.Linear(self.hidden_out, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, out_size)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x= self.fc_out(x)
        return x

#====================================================================================
# CNN
#==================================================================================
class ReccurrentBlock(torch.nn.Module):
    """[summary]
    
    Arguments:
        torch {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    def __init__(self, eps=10, delta=10):
        super(ReccurrentBlock, self).__init__()
        self.eps         = torch.nn.Parameter(torch.randn(1), requires_grad = True)
        self.delta       = torch.nn.Parameter(torch.randn(1), requires_grad = True)
        torch.nn.init.constant_(self.eps, eps)
        torch.nn.init.constant_(self.delta, delta)
        
    def forward(self, dist):
        dist = torch.floor(dist*self.eps)
        dist[dist>self.delta]=self.delta
        return dist


class Conv2DAdaptiveRecurrence(nn.Module):
    def __init__(self, in_size=1, out_size=12, dropout=0.2, eps=10, delta=10, width=50):
        super(Conv2DAdaptiveRecurrence, self).__init__()
        
        self.in_size  = in_size
        self.rec_block = ReccurrentBlock(eps=eps, delta=delta)
        self.con_layer = Conv2D(in_size, out_size, dropout, width)
        
    def forward(self, x):
        x  = self.rec_block(x)
        prediction = self.con_layer(x)
        return prediction

#===============================================================================================
# ResNet
#================================================================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_size, out_size, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_size != self.expansion*out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, self.expansion*out_size,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_size)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_size, out_size, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.conv3 = nn.Conv2d(out_size, self.expansion *
                               out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_size)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_size != self.expansion*out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, self.expansion*out_size,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_size)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_size, num_classes=10,eps=10, delta=10):
        super(ResNet, self).__init__()
        self.in_size = 64

        self.conv1 = nn.Conv2d(in_size, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.rec_block = ReccurrentBlock(eps=eps, delta=delta)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, out_size, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_size, out_size, stride))
            self.in_size = out_size * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.rec_block(x)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def GetModel(model_name,in_size, out_size=10, dropout=0.2, eps=10, delta=10, width=50):
    if model_name == 'ResNet18':
        return ResNet(BasicBlock, [2, 2, 2, 2], in_size=in_size, num_classes=out_size,eps=eps, delta=delta)
    elif model_name == 'ResNet34':
        return ResNet(BasicBlock, [3, 4, 6, 3], in_size=in_size, num_classes=out_size,eps=eps, delta=delta)
    elif model_name == 'ResNet50':
        return ResNet(Bottleneck, [3, 4, 6, 3], in_size=in_size, num_classes=out_size,eps=eps, delta=delta)
    elif model_name == 'ResNet101':
        return ResNet(Bottleneck, [3, 4, 23, 3], in_size=in_size, num_classes=out_size,eps=eps, delta=delta)
    elif model_name == 'ResNet152':
        return ResNet(Bottleneck, [3, 8, 36, 3], in_size=in_size, num_classes=out_size,eps=eps, delta=delta)
    else:
        return Conv2DAdaptiveRecurrence(in_size=in_size, out_size=out_size, dropout=dropout, eps=eps, delta=delta, width=width)