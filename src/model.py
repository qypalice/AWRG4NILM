import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

#====================================================================================
# CNN
#==================================================================================
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
    def __init__(self, block, num_blocks, in_size, num_classes=10):
        super(ResNet, self).__init__()
        self.in_size = 64

        self.conv1 = nn.Conv2d(in_size, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet_AR(nn.Module):
    def __init__(self, block, num_blocks, in_size=1, num_classes=12, eps=10, delta=10):
        super(ResNet_AR, self).__init__()
        
        self.in_size  = in_size
        self.rec_block = ReccurrentBlock(eps=eps, delta=delta)
        self.con_layer = ResNet(block, num_blocks, in_size, num_classes)
        
    def forward(self, x):
        x  = self.rec_block(x)
        prediction = self.con_layer(x)
        return prediction

#===============================================================================================
# EfficientNet v2 modified from https://github.com/d-li14/efficientnetv2.pytorch.git
#================================================================================================
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

 
class SELayer(nn.Module):
    def __init__(self, in_size, out_size, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(out_size, _make_divisible(in_size // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(in_size // reduction, 8), out_size),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(in_size, out_size, stride):
    return nn.Sequential(
        nn.Conv2d(in_size, out_size, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_size),
        SiLU()
    )


def conv_1x1_bn(in_size, out_size):
    return nn.Sequential(
        nn.Conv2d(in_size, out_size, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_size),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, in_size, out_size, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(in_size * expand_ratio)
        self.identity = stride == 1 and in_size == out_size
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(in_size, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(in_size, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, out_size, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_size),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(in_size, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, out_size, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_size),
            )


    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self, cfgs, in_size, num_classes=10, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(in_size, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def effnetv2_s(in_size, out_size):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, in_size, out_size)


def effnetv2_m(in_size, out_size):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  3, 1, 0],
        [4,  48,  5, 2, 0],
        [4,  80,  5, 2, 0],
        [4, 160,  7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512,  5, 1, 1],
    ]
    return EffNetV2(cfgs, in_size, out_size)


def effnetv2_l(in_size, out_size):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  7, 2, 0],
        [4,  96,  7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640,  7, 1, 1],
    ]
    return EffNetV2(cfgs, in_size, out_size)


def effnetv2_xl(in_size, out_size):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  8, 2, 0],
        [4,  96,  8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640,  8, 1, 1],
    ]
    return EffNetV2(cfgs, in_size, out_size)

class EfficientNet_AR(nn.Module):
    def __init__(self, model_size, in_size=1, out_size=12, eps=10, delta=10):
        super(EfficientNet_AR, self).__init__()
        
        self.in_size  = in_size
        self.rec_block = ReccurrentBlock(eps=eps, delta=delta)
        if model_size == 's':
            self.con_layer = effnetv2_s(in_size, out_size)
        elif model_size == 'm':
            self.con_layer = effnetv2_m(in_size, out_size)
        elif model_size == 'l':
            self.con_layer = effnetv2_l(in_size, out_size)
        elif model_size == 'xl':
            self.con_layer = effnetv2_xl(in_size, out_size)
        
    def forward(self, x):
        x  = self.rec_block(x)
        prediction = self.con_layer(x)
        return prediction


def GetModel(model_name,in_size, out_size=10, dropout=0.2, eps=10, delta=10, width=50):
    if model_name == 'ResNet18':
        return ResNet_AR(BasicBlock, [2, 2, 2, 2], in_size=in_size, num_classes=out_size,eps=eps, delta=delta)
    elif model_name == 'ResNet34':
        return ResNet_AR(BasicBlock, [3, 4, 6, 3], in_size=in_size, num_classes=out_size,eps=eps, delta=delta)
    elif model_name == 'ResNet50':
        return ResNet_AR(Bottleneck, [3, 4, 6, 3], in_size=in_size, num_classes=out_size,eps=eps, delta=delta)
    elif model_name == 'ResNet101':
        return ResNet_AR(Bottleneck, [3, 4, 23, 3], in_size=in_size, num_classes=out_size,eps=eps, delta=delta)
    elif model_name == 'ResNet152':
        return ResNet_AR(Bottleneck, [3, 8, 36, 3], in_size=in_size, num_classes=out_size,eps=eps, delta=delta)
    elif model_name == 'EffiNet_small':
        return EfficientNet_AR('s',in_size=in_size, out_size=out_size, eps=eps, delta=delta)
    elif model_name == 'EffiNet_medium':
        return EfficientNet_AR('m',in_size=in_size, out_size=out_size, eps=eps, delta=delta)
    elif model_name == 'EffiNet_large':
        return EfficientNet_AR('l',in_size=in_size, out_size=out_size, eps=eps, delta=delta)
    elif model_name == 'EffiNet_extra_large':
        return EfficientNet_AR('xl',in_size=in_size, out_size=out_size, eps=eps, delta=delta)
    else:
        return Conv2DAdaptiveRecurrence(in_size=in_size, out_size=out_size, dropout=dropout, eps=eps, delta=delta, width=width)