'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
'''
Early Exit Branch使用了depth convolution network 配合全连接层
'''
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

class VGG(nn.Module):
    def __init__(self, args=None, vgg_name="VGG19", freeze = False, class_nums=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, class_nums)
        self.targets =  self._layer_detact(self.features, nn.MaxPool2d)
        self.early_exists = self._make_early_exit([64,128,256,512,512], 32, class_nums=class_nums)
        # print(self.targets)
        self.args = args
        # self.fc_out = True if args is None else self.args.fc_out
        self.fc_out = False
        print("fc_out", self.fc_out)
        if freeze:
            for p in self.parameters():
                    p.requires_grad=False
        self.out_dims = eval(self.args.out_dims)[-5:]
        self.fc_layers = self._make_fc()

    def _layer_detact(self, layers, target_layer):
        length = len(layers)
        print("length", length)
        print([i for i in range(length) if isinstance(layers[i], target_layer)])
        return [i for i in range(length) if isinstance(layers[i], target_layer)]

    def _make_fc(self):
        if self.args.pool_out == "avg":
            layers = [
                nn.AdaptiveAvgPool1d(output) for output in self.out_dims
            ]
        elif self.args.pool_out == "max":
            layers = [
                nn.AdaptiveMaxPool1d(output) for output in self.out_dims
            ]
        return nn.Sequential(*layers)

    def _make_early_exit(self, inp_channels, inp_dim, class_nums = 10):
        '''
        inp_channel:输入通道数
        inp_dim:输入图片维度
        out_num:总共有几层需要添加early exit
        '''
        early_exit_models = nn.ModuleList()
        for inp_channel in inp_channels:
            inp_dim = inp_dim // 2
            #         inp_channel = inp_channel *2
            fc_dim = inp_dim ** 2 * inp_channel
            # print(inp_channel, inp_dim)
            model = early_exit_Branch(inp=inp_channel, oup=inp_channel, fc_inp=fc_dim, class_nums=class_nums)
            early_exit_models.append(model)
        return early_exit_models

    def forward(self, x):
        feature_maps = []
        fc_layer = 0
        # print("target", self.targets)
        step = 0
        for i, feature in enumerate(self.features):
            x = feature(x)
            if i in self.targets:
                # print(i, "x size", x.size())
                # print(feature)
                if self.fc_out:
                    # print(fc_layer)
                    # print("out1", x.size())
                    out = self.fc_layers[fc_layer](x.view(x.size(0), x.size(1), -1))    #将向量铺平
                    # print("out2", out.size())
                    if self.args.pool_out == "max":
                        out, _ = out.max(dim=1)
                    else:
                        out = out.mean(dim=1)
                    fc_layer += 1
                    feature_maps.append(torch.squeeze(out))
                else:
                    out = self.early_exists[step](x)
                    # print(out.shape)
                    feature_maps.append(out)
                    step += 1
        # print("final x", x.size())
        out = x.view(x.size(0), -1)
        # print(out.size())
        out = self.classifier(out)
        # print("final out", out.size())
        # out = F.softmax(out, dim=1)
        # feature_maps[-1] = out
        feature_maps.append(out)
        return feature_maps

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



'''
VGG net 
5 layers output size
dim of each of the layer is [5000, 1000, 500, 200, 10]

'''
def test():
    net = VGG(args, vgg_name='VGG19', freeze=True)
    x = torch.randn(8,3,224,224)
    y = net(x)
    print(len(y))
    for item in y:
        print("item_size", item.size())


# import argparse
# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--fc_out', default=1, type=int, help='if immediate output from fc-layer')
# parser.add_argument('--out_dims', default="[5000,1000,500,200,10]", type=str, help='the dims of output pooling layers')
# parser.add_argument('--pool_out', default="max", type=str, help='the type of pooling layer of output')
# args = parser.parse_args()
# # from loss import CrossEntropy
# test()
