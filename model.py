import torch
from torch import nn

import torch.nn.functional as F

import json, math
from functools import reduce
from operator import mul
# import torchreid
from torchvision import models
from torch.hub import load_state_dict_from_url
from configuration import configure_model

def load_model_from_config(config_path, input_size):
    c, h, w = input_size['channels'], input_size['height'], input_size['width']
    with open(config_path, 'r') as f:
        config = json.load(f)
    model = configure_model(config, [c, h, w])
    return model.cuda()

class SymReLU(torch.nn.Module):
    def __init__(self):
        super(SymReLU, self).__init__()

    def forward(self, x):
        return torch.max(torch.tensor([-1]), torch.min(torch.tensor([1]), x))


class ConvAct(torch.nn.Module):
    def __init__(self, in_ch, out_ch, f, s, p):
        super(ConvAct, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, f, stride=s, padding=p)  # padding_mode='reflect')
        self.act = nn.Softsign()

    def forward(self, x):
        return self.act(self.conv(x))


class KorNet(torch.nn.Module):
    def __init__(self):
        super(KorNet, self).__init__()

        self.conv0 = ConvAct(1, 16, (3, 3), 1, 0)
        self.conv1 = ConvAct(16, 16, (5, 5), 2, 2)
        self.conv2 = ConvAct(16, 16, (3, 3), 1, 1)
        self.conv3 = ConvAct(16, 24, (5, 5), 2, 2)
        self.conv4 = ConvAct(24, 24, (3, 3), 1, 1)
        self.conv5 = ConvAct(24, 24, (3, 3), 1, 1)
        self.final = nn.Linear(24 * 9 * 9, 25 * 1 * 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.final(x.flatten(start_dim=1))
        return x


class FaceNet(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceNet, self).__init__()

        # Сверточные слои
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception-подобные блоки (упрощенные)
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        # Глобальный средний пулинг
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Полносвязный слой для эмбеддинга
        self.fc = nn.Linear(1024, embedding_size)

    def forward(self, x):
        # Применение сверточных слоев
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)

        # Inception-блоки
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        # Глобальный средний пулинг
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Эмбеддинг
        x = self.fc(x)
        # L2-нормализация
        x = F.normalize(x, p=2, dim=1)

        return x


# Упрощенный Inception-блок
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, conv1x1, conv3x3_reduce, conv3x3, conv5x5_reduce, conv5x5, pool_proj):
        super(InceptionBlock, self).__init__()

        # Ветка 1x1 свертки
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, conv1x1, kernel_size=1),
            nn.BatchNorm2d(conv1x1),
            nn.ReLU(inplace=True)
        )

        # Ветка 3x3 свертки
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, conv3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(conv3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv3x3_reduce, conv3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv3x3),
            nn.ReLU(inplace=True)
        )

        # Ветка 5x5 свертки
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, conv5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(conv5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv5x5_reduce, conv5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(conv5x5),
            nn.ReLU(inplace=True)
        )

        # Ветка пулинга
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


class OneShotNet(nn.Module):
    """
    A Convolutional Siamese Network for One-Shot Learning.

    Siamese networks learn image representations via a supervised metric-based
    approach. Once tuned, their learned features can be leveraged for one-shot
    learning without any retraining.

    References
    ----------
    - Koch et al., https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
    """

    def __init__(self):
        super(OneShotNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(inplace=True),  # 128@42*42
            nn.MaxPool2d(2),  # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(inplace=True),  # 128@18*18
            nn.MaxPool2d(2),  # 128@9*9
            nn.Conv2d(128, 256, 4),
            nn.ReLU(inplace=True),  # 256@6*6
        )
        self.liner = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def sub_forward(self, x):
        """
        Forward pass the input image through 1 subnetwork.

        Args
        ----
        - x: a Variable of size (B, C, H, W). Contains either the first or
          second image pair across the input batch.

        Returns
        -------
        - out: a Variable of size (B, 4096). The hidden vector representation
          of the input vector x.
        """
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        """
        Forward pass the input image pairs through both subtwins. An image
        pair is composed of a left tensor x1 and a right tensor x2.

        Concretely, we compute the component-wise L1 distance of the hidden
        representations generated by each subnetwork, and feed the difference
        to a final fc-layer followed by a sigmoid activation function to
        generate a similarity score in the range [0, 1] for both embeddings.

        Args
        ----
        - x1: a Variable of size (B, C, H, W). The left image pairs along the
          batch dimension.
        - x2: a Variable of size (B, C, H, W). The right image pairs along the
          batch dimension.

        Returns
        -------
        - probas: a Variable of size (B, 1). A probability scalar indicating
          whether the left and right input pairs, along the batch dimension,
          correspond to the same class. We expect the network to spit out
          values near 1 when they belong to the same class, and 0 otherwise.
        """
        # encode image pairs
        h1 = self.sub_forward(x1)
        h2 = self.sub_forward(x2)

        # compute l1 distance
        diff = torch.abs(h1 - h2)

        # score the similarity between the 2 encodings
        scores = self.out(diff)

        # return scores (without sigmoid) and use bce_with_logit
        # for increased numerical stability
        return scores


class LightweightEmbeddingNet(nn.Module):
    def __init__(self, image_size):
        super(LightweightEmbeddingNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Compute flattened size dynamically based on input size
        dummy_input = torch.zeros(1, image_size['channels'], image_size['height'], image_size['width'])
        flattened_size = self._get_flattened_size(dummy_input)

        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, image_size['output_dim'])

    def backbone(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return x

    def _get_flattened_size(self, x):
        x = self.backbone(x)
        # print("backbone", x.shape)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.backbone(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = F.normalize(self.fc2(x), p=2, dim=1)
        return x


class LightweightEmbeddingNet2(nn.Module):
    def __init__(self, image_size):
        super(LightweightEmbeddingNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn5 = nn.BatchNorm2d(64)
        # self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Compute flattened size dynamically based on input size
        dummy_input = torch.zeros(1, image_size['channels'], image_size['height'], image_size['width'])
        flattened_size = self._get_flattened_size(dummy_input)

        self.fc1 = nn.Linear(flattened_size, image_size['output_dim'])
        # self.fc2 = nn.Linear(256, image_size['output_dim'])

    def backbone(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x

    def _get_flattened_size(self, x):
        x = self.backbone(x)
        # print("backbone", x.shape)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.backbone(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = self.fc1(x)
        # x = F.normalize(self.fc2(x), p=2, dim=1)
        return x


class ResNet50(nn.Module):
    def __init__(self, image_size, pretrained=True):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=pretrained)

        # self.resnet50 = nn.Sequential(*list(self.resnet.children())[:-1])
        num_features = self.resnet50.fc.in_features

        self.resnet50.fc = nn.Linear(num_features, image_size['output_dim'])
        # dummy_input = torch.zeros(1, image_size['channels'], image_size['height'], image_size['width'])
        # flattened_size = self._get_flattened_size(dummy_input)

        # self.fc = nn.Linear(flattened_size, image_size['output_dim'])
        self.normalize = nn.functional.normalize

    # def _get_flattened_size(self, x):
    #     x = self.resnet50(x)
    #     return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.resnet50(x)
        x = self.normalize(x, p=2, dim=-1)
        return x


class MobileNetV2(nn.Module):
    def __init__(self, image_size, pretrained=True):
        super(MobileNetV2, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)

        self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-1])

        dummy_input = torch.zeros(1, image_size['channels'], image_size['height'], image_size['width'])
        flattened_size = self._get_flattened_size(dummy_input)

        self.fc = nn.Linear(flattened_size, image_size['output_dim'])
        self.normalize = nn.functional.normalize

    def _get_flattened_size(self, x):
        x = self.mobilenet(x)
        print(x.shape)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.mobilenet(x).squeeze()
        x = self.fc(x)
        x = self.normalize(x, p=2, dim=1)
        return x


# class OSNetFeatureExtractor(nn.Module):
#     def __init__(self, embedding_dim=64, pretrained=True):
#         super(OSNetFeatureExtractor, self).__init__()
#         # Создаем OSNet-AIN
#         self.osnet = build_model('osnet_ain_x1_0', num_classes=1000, pretrained=pretrained)
#         # Заменяем классификационный слой
#         self.osnet.fc = nn.Linear(512, embedding_dim)
#         self.normalize = nn.functional.normalize
#
#     def forward(self, x):
#         # Извлекаем признаки с помощью OSNet
#         x = self.osnet(x)
#         x = self.normalize(x, p=2, dim=1)
#         return x

'''Slightly modified implementation of OSNet from the paper
"Omni-Scale Feature Learning for Person Re-Identification" by
Kaiyang Zhou, Yongxin Yang, Andrea Cavallaro, Tao Xiang
(https://arxiv.org/abs/1905.00953)

Author: Connor Anderson
'''


# __all__ = ['OSNet']


def passthrough(x):
    '''Noop layer'''
    return x


def conv1x1(inc, outc, linear=False):
    '''1x1 conv -> batchnorm -> (optional) ReLU'''
    layers = [torch.nn.Conv2d(inc, outc, 1, bias=False),
              torch.nn.BatchNorm2d(outc)]
    if not linear:
        layers.append(torch.nn.ReLU(inplace=True))
    return torch.nn.Sequential(*layers)


def conv3x3(inc, outc, stride=1):
    '''3x3 conv -> batchnorm -> ReLU'''
    return torch.nn.Sequential(
        torch.nn.Conv2d(inc, outc, 3, padding=1, stride=stride, bias=False),
        torch.nn.BatchNorm2d(outc),
        torch.nn.ReLU(inplace=True)
    )


def convlite(inc, outc):
    '''Lite conv layer. 1x1 conv -> 3x3 depthwise conv -> batchnorm -> ReLU'''
    return torch.nn.Sequential(
        torch.nn.Conv2d(inc, outc, 1, bias=False),
        torch.nn.Conv2d(outc, outc, 3, padding=1, groups=outc, bias=False),
        torch.nn.BatchNorm2d(outc),
        torch.nn.ReLU(inplace=True)
    )


def convstack(inc, outc, n=1):
    '''A stack of n convlite layers'''
    convs = convlite(inc, outc)
    if n > 1:
        convs = [convs] + [convlite(outc, outc) for i in range(n - 1)]
        convs = torch.nn.Sequential(*convs)
    return convs


class Gate(torch.nn.Module):
    '''Unified Aggregation Gate.

    Args:
        c (int): number of channels (input and output are the same)
    '''

    def __init__(self, c):
        super().__init__()
        self.gate = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(c, c // 16, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(c // 16, c, 1),
            torch.nn.Sigmoid())

    def forward(self, x):
        g = self.gate(x)
        x = x * g
        return x


class Bottleneck(torch.nn.Module):
    '''OSNet bottleneck layer (figure 4 in the paper).

    Args:
        inc (int): number of input feature channels
        outc (int): number of output feature channels
        t (int): number of scales
        reduction (int): factor to reduce/expand the number of feature
            channels before/after multiscale layers
    '''

    def __init__(self, inc, outc, t=4, reduction=4):
        super().__init__()
        midc = inc // reduction
        self.conv1 = conv1x1(inc, midc)
        self.streams = torch.nn.ModuleList([
            convstack(midc, midc, n=i + 1) for i in range(t)
        ])
        self.gate = Gate(midc)
        self.conv2 = conv1x1(midc, outc, linear=True)
        self.project = (passthrough if inc == outc else
                        conv1x1(inc, outc, linear=True))

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = [s(x) for s in self.streams]
        x = sum([self.gate(xi) for xi in x])
        x = self.conv2(x)
        x = self.project(identity) + x
        x = torch.nn.functional.relu(x, inplace=True)
        return x


class OSNet(torch.nn.Module):
    '''OmniScale network.

    Args:
        n_class (int): number of classes for classification
    '''

    def __init__(self, image_size):
        super().__init__()
        # replace the 7x7 with 3 3x3s
        self.conv1 = torch.nn.Sequential(
            conv3x3(3, 32, stride=2),
            conv3x3(32, 32),
            conv3x3(32, 64))
        self.maxpool = torch.nn.MaxPool2d(3, 2)
        self.conv2 = torch.nn.Sequential(
            Bottleneck(64, 256),
            Bottleneck(256, 256),
            conv1x1(256, 256),
            torch.nn.AvgPool2d(2, 2))
        self.conv3 = torch.nn.Sequential(
            Bottleneck(256, 384),
            Bottleneck(384, 384),
            conv1x1(384, 384),
            torch.nn.AvgPool2d(2, 2))
        self.conv4 = torch.nn.Sequential(
            Bottleneck(384, 512),
            Bottleneck(512, 512),
            conv1x1(512, 512),
            torch.nn.AvgPool2d(2, 2))
        # add extra avg pool and extra 1x1 conv
        self.conv5 = conv1x1(512, 512)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        # replace the extra fc (512 x 512) with a single classifier
        self.fc = torch.nn.Linear(512, image_size["output_dim"])

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = x.flatten(1)
        # print(x.shape)
        x = self.fc(x)
        return x


def ChooseModel(model_name: str, image_size: dict):
    if model_name == "KorNet":
        return KorNet().cuda()
    elif model_name == "OneShotNet":
        return OneShotNet().cuda()
    elif model_name == "Light":
        return LightweightEmbeddingNet(image_size).cuda()
    elif model_name == "Light2":
        return LightweightEmbeddingNet2(image_size).cuda()
    elif model_name == "ResNet50":
        return ResNet50(image_size).cuda()
    elif model_name == "MobileNet":
        return MobileNetV2(image_size).cuda()
    elif model_name == "OSNet":
        return OSNet(image_size).cuda()
    elif model_name == "FaceNet":
        return FaceNet().cuda()
    else:
        print("Model is not defined!!!")
        exit(-1)
