import torch
from torch import nn
from torchvision.transforms import v2

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


def prepare_augmentation():
    train_transform = v2.Compose(
        [
            v2.RandomHorizontalFlip(0.4),
            v2.RandomVerticalFlip(0.1),
            v2.RandomApply(transforms=[v2.RandomRotation(degrees=(0, 90))], p=0.5),
            v2.RandomApply(transforms=[v2.ColorJitter(brightness=0.3, hue=0.1)], p=0.3),
            v2.RandomApply(transforms=[v2.GaussianBlur(kernel_size=(5, 9))], p=0.3),
            # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        ]
    )

    transforms = v2.Compose(
        [
            v2.RandomApply(transforms=[v2.RandomRotation(degrees=(-5, 5), fill=1)], p=0.0),
            v2.RandomApply(transforms=[v2.Compose([
                                        v2.RandomResize(int(37 * 0.7), int(37 * 0.9)),
                                        v2.Resize(size=(37, 37))
                                    ])], p=0.0),
            v2.RandomApply(transforms=[v2.RandomPerspective(0.15, fill=1)], p=1.0)
            # v2.RandomApply(transforms=[v2.functional.perspective(startpoints=[[0, 0], [0, 37], [37, 37], [37, 0]],
            # 													 endpoints=[[0, 0], [0, 37], [uniRand(), 37], [uniRand(), 0]],
            # 													 fill=1)], p=1.0)
        ]
    )

    return transforms


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
