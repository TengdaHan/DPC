## modified from https://github.com/kenshohara/3D-ResNets-PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'ResNet2d3d_full', 'resnet18_2d3d_full', 'resnet34_2d3d_full', 'resnet50_2d3d_full', 'resnet101_2d3d_full',
    'resnet152_2d3d_full', 'resnet200_2d3d_full',
]

def conv3x3x3(in_planes, out_planes, stride=1, bias=False):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias)

def conv1x3x3(in_planes, out_planes, stride=1, bias=False):
    # 1x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1,3,3),
        stride=(1,stride,stride),
        padding=(0,1,1),
        bias=bias)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock3d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, feature_size=None, batchnorm=False, affine=True, track_running_stats=True, use_final_relu=True):
        super(BasicBlock3d, self).__init__()
        bias = not batchnorm
        self.use_final_relu = use_final_relu
        self.conv1 = conv3x3x3(inplanes, planes, stride, bias=bias)
        if batchnorm == 'batchnorm': 
            self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        elif batchnorm == 'insnorm': 
            self.bn1 = nn.InstanceNorm3d(planes, affine=affine)
        elif batchnorm == 'layernorm':
            feature_size = (math.ceil(feature_size[0]/stride), math.ceil(feature_size[1]/stride), math.ceil(feature_size[2]/stride))
            self.bn1 = nn.LayerNorm((planes, *feature_size))

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, bias=bias)
        if batchnorm == 'batchnorm': 
            self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        elif batchnorm == 'insnorm':
            self.bn2 = nn.InstanceNorm3d(planes, affine=affine)
        elif batchnorm == 'layernorm':
            self.bn2 = nn.LayerNorm((planes, *feature_size))

        self.downsample = downsample
        self.stride = stride
        self.batchnorm = batchnorm 

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batchnorm: out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batchnorm: out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu: out = self.relu(out)

        return out


class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, feature_size=None, batchnorm=False, affine=True, track_running_stats=True, use_final_relu=True):
        super(BasicBlock2d, self).__init__()
        bias = not batchnorm
        self.use_final_relu = use_final_relu
        self.conv1 = conv1x3x3(inplanes, planes, stride, bias=bias)
        if batchnorm == 'batchnorm': 
            self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        elif batchnorm == 'insnorm': 
            self.bn1 = nn.InstanceNorm3d(planes, affine=affine)
        elif batchnorm == 'layernorm':
            feature_size = (feature_size[0]//1, math.ceil(feature_size[1]/stride), math.ceil(feature_size[2]/stride))
            self.bn1 = nn.LayerNorm((planes, *feature_size))

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes, bias=bias)
        if batchnorm == 'batchnorm': 
            self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        elif batchnorm == 'insnorm':
            self.bn2 = nn.InstanceNorm3d(planes, affine=affine)
        elif batchnorm == 'layernorm':
            self.bn2 = nn.LayerNorm((planes, *feature_size))

        self.downsample = downsample
        self.stride = stride
        self.batchnorm = batchnorm 

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batchnorm: out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batchnorm: out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu: out = self.relu(out)

        return out


class Bottleneck3d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, feature_size=None, batchnorm=False, affine=True, track_running_stats=True, use_final_relu=True):
        super(Bottleneck3d, self).__init__()
        bias = not batchnorm
        self.use_final_relu = use_final_relu
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=bias)
        if batchnorm == 'batchnorm': 
            self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        elif batchnorm == 'insnorm': 
            self.bn1 = nn.InstanceNorm3d(planes, affine=affine)
        elif batchnorm == 'layernorm':
            self.bn1 = nn.LayerNorm((planes, *feature_size))

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        if batchnorm == 'batchnorm': 
            self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        elif batchnorm == 'insnorm': 
            self.bn2 = nn.InstanceNorm3d(planes, affine=affine)
        elif batchnorm == 'layernorm':
            feature_size = (math.ceil(feature_size[0]/stride), math.ceil(feature_size[1]/stride), math.ceil(feature_size[2]/stride))
            self.bn2 = nn.LayerNorm((planes, *feature_size))

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=bias)
        if batchnorm == 'batchnorm': 
            self.bn3 = nn.BatchNorm3d(planes * 4, track_running_stats=track_running_stats)
        elif batchnorm == 'insnorm': 
            self.bn3 = nn.InstanceNorm3d(planes * 4, affine=affine)
        elif batchnorm == 'layernorm':
            self.bn3 = nn.LayerNorm((planes * 4, *feature_size))

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.batchnorm = batchnorm

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batchnorm: out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batchnorm: out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.batchnorm: out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu: out = self.relu(out)

        return out


class Bottleneck2d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, feature_size=None, batchnorm=False, affine=True, track_running_stats=True, use_final_relu=True):
        super(Bottleneck2d, self).__init__()
        bias = not batchnorm 
        self.use_final_relu = use_final_relu
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=bias)
        if batchnorm == 'batchnorm': 
            self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        elif batchnorm == 'insnorm': 
            self.bn1 = nn.InstanceNorm3d(planes, affine=affine)
        elif batchnorm == 'layernorm':
            self.bn1 = nn.LayerNorm((planes, *feature_size))

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1,3,3), stride=(1,stride,stride), padding=(0,1,1), bias=bias)
        if batchnorm == 'batchnorm': 
            self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        elif batchnorm == 'insnorm': 
            self.bn2 = nn.InstanceNorm3d(planes, affine=affine)
        elif batchnorm == 'layernorm':
            feature_size = (feature_size[0]//1, math.ceil(feature_size[1]/stride), math.ceil(feature_size[2]/stride))
            self.bn2 = nn.LayerNorm((planes, *feature_size))

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=bias)
        if batchnorm == 'batchnorm': 
            self.bn3 = nn.BatchNorm3d(planes * 4, track_running_stats=track_running_stats)
        elif batchnorm == 'insnorm': 
            self.bn3 = nn.InstanceNorm3d(planes * 4, affine=affine)
        elif batchnorm == 'layernorm':
            self.bn3 = nn.LayerNorm((planes * 4, *feature_size))

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.batchnorm = batchnorm 

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batchnorm: out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batchnorm: out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.batchnorm: out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu: out = self.relu(out)

        return out


class ResNet2d3d_full(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 batchnorm=False, 
                 affine=True,
                 track_running_stats=True,
                 expand_factor=1):
        self.inplanes = 64 * expand_factor
        self.batchnorm = batchnorm 
        self.affine=affine # affine for InsNorm
        self.track_running_stats = track_running_stats
        bias = not batchnorm 
        super(ResNet2d3d_full, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64*expand_factor,
            kernel_size=(1,7,7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=bias)
        if batchnorm == 'batchnorm': 
            print('=> use BatchNorm')
            self.bn1 = nn.BatchNorm3d(64*expand_factor, track_running_stats=track_running_stats)
        elif batchnorm == 'insnorm': 
            print('=> use InstanceNorm with affine=%s' % str(affine))
            self.bn1 = nn.InstanceNorm3d(64*expand_factor, affine=affine)
        elif batchnorm == 'layernorm': 
            print('=> use LayerNorm')
            self.bn1 = nn.LayerNorm((64*expand_factor, sample_duration, math.ceil(sample_size/2), math.ceil(sample_size/2)))
        elif batchnorm != '':
            raise ValueError('BN choice is wrong')
        else:
            print('=> no BatchNorm or InstanceNorm or LayerNorm')
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        if not isinstance(block, list):
            block = [block] * 4

        layer_size = [math.ceil(sample_size/4), math.ceil(sample_size/4), math.ceil(sample_size/8), math.ceil(sample_size/16)]
        layer_duration = [sample_duration]
        for i in block:
            if (i == Bottleneck2d) or (i == BasicBlock2d):
                layer_duration.append(layer_duration[-1])
            else:
                layer_duration.append(math.ceil(layer_duration[-1]/2))
        self.layer_size = layer_size
        self.layer_duration = layer_duration

        self.layer1 = self._make_layer(block[0], 64*expand_factor, layers[0], shortcut_type, feature_size=(layer_duration[0],layer_size[0],layer_size[0]))
        self.layer2 = self._make_layer(
            block[1], 128*expand_factor, layers[1], shortcut_type, stride=2, feature_size=(layer_duration[1],layer_size[1],layer_size[1]))
        self.layer3 = self._make_layer(
            block[2], 256*expand_factor, layers[2], shortcut_type, stride=2, feature_size=(layer_duration[2],layer_size[2],layer_size[2]))
        self.layer4 = self._make_layer(
            block[3], 256*expand_factor, layers[3], shortcut_type, stride=2, feature_size=(layer_duration[3],layer_size[3],layer_size[3]), is_final=True)
        # modify layer4 from exp=512 to exp=256
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: m.bias.data.zero_()
                # nn.init.orthogonal_(m.weight, 1)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, feature_size=(0,0,0), is_final=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if (block == Bottleneck2d) or (block == BasicBlock2d):
                customized_stride = (1, stride, stride) # todo
                ds_feature_size = (math.ceil(feature_size[0]/customized_stride[0]),
                                   math.ceil(feature_size[1]/customized_stride[1]),
                                   math.ceil(feature_size[2]/customized_stride[2]))
            else:
                customized_stride = stride
                ds_feature_size = (math.ceil(feature_size[0]/customized_stride),
                                   math.ceil(feature_size[1]/customized_stride),
                                   math.ceil(feature_size[2]/customized_stride))

            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=customized_stride)
            else:
                if self.batchnorm == 'batchnorm':
                    downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=customized_stride,
                        bias=False), 
                    nn.BatchNorm3d(planes * block.expansion, track_running_stats=self.track_running_stats)
                    )
                elif self.batchnorm == 'insnorm':
                    downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=customized_stride,
                        bias=False), 
                    nn.InstanceNorm3d(planes * block.expansion, affine=self.affine)
                    )
                elif self.batchnorm == 'layernorm':
                    downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=customized_stride,
                        bias=False), 
                    nn.LayerNorm((planes * block.expansion, *ds_feature_size))
                    )
                else:
                    downsample = nn.Sequential(
                        nn.Conv3d(
                            self.inplanes,
                            planes * block.expansion,
                            kernel_size=1,
                            stride=customized_stride,
                            bias=True), 
                        )
        else:
            ds_feature_size = feature_size

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, feature_size=feature_size, batchnorm=self.batchnorm, affine=self.affine, track_running_stats=self.track_running_stats))
        self.inplanes = planes * block.expansion
        if is_final:
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes, feature_size=ds_feature_size, batchnorm=self.batchnorm, affine=self.affine, track_running_stats=self.track_running_stats))
            layers.append(block(self.inplanes, planes, feature_size=ds_feature_size, batchnorm=self.batchnorm, affine=self.affine, track_running_stats=self.track_running_stats, use_final_relu=False))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, feature_size=ds_feature_size, batchnorm=self.batchnorm, affine=self.affine, track_running_stats=self.track_running_stats))
                
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.batchnorm: x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x)

        return x


## full resnet
def resnet18_2d3d_full(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet2d3d_full([BasicBlock2d, BasicBlock2d, BasicBlock3d, BasicBlock3d], 
                   [2, 2, 2, 2], **kwargs)
    return model

def resnet34_2d3d_full(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet2d3d_full([BasicBlock2d, BasicBlock2d, BasicBlock3d, BasicBlock3d], 
                   [3, 4, 6, 3], **kwargs)
    return model

def resnet50_2d3d_full(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet2d3d_full([Bottleneck2d, Bottleneck2d, Bottleneck3d, Bottleneck3d], 
                   [3, 4, 6, 3], **kwargs)
    return model

def resnet101_2d3d_full(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet2d3d_full([Bottleneck2d, Bottleneck2d, Bottleneck3d, Bottleneck3d], 
                   [3, 4, 23, 3], **kwargs)
    return model

def resnet152_2d3d_full(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet2d3d_full([Bottleneck2d, Bottleneck2d, Bottleneck3d, Bottleneck3d], 
                   [3, 8, 36, 3], **kwargs)
    return model

def resnet200_2d3d_full(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet2d3d_full([Bottleneck2d, Bottleneck2d, Bottleneck3d, Bottleneck3d], 
                   [3, 24, 36, 3], **kwargs)
    return model

def neq_load(model, name):
    ''' load pre-trained model in a not-equal way,
    when new model has been modified '''
    pretrained_dict = model_zoo.load_url(model_urls[name])
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def neq_load_customized(model, pretrained_dict):
    ''' load pre-trained model in a not-equal way,
    when new model has been partially modified '''
    model_dict = model.state_dict()
    tmp = {}
    print('\n=======Check Weights Loading======')
    print('Weights not used from pretrained file:')
    for k, v in pretrained_dict.items():
        if k in model_dict:
            tmp[k] = v
        else:
            print(k)
    print('---------------------------')
    print('Weights not loaded into new model:')
    for k, v in model_dict.items():
        if k not in pretrained_dict:
            print(k)
    print('===================================\n')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model


if __name__ == '__main__':
    mymodel = resnet18_2d3d(shortcut_type='B',
                            sample_size=128,
                            sample_duration=16,
                            batchnorm='insnorm')
    mydata = torch.FloatTensor(4, 3, 16, 128, 128)
    nn.init.normal_(mydata)
    import ipdb; ipdb.set_trace()
    mymodel(mydata)
