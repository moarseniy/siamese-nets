import torch
from torch import nn
from model_components import *
from torchvision import transforms


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
    torch.manual_seed(17)
    augmentation = torch.nn.Sequential(
        transforms.RandomRotation(degrees=5),
        transforms.RandomPerspective(),
        # transforms.v2.RandomResize(),
    )
    return torch.jit.script(augmentation)


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

    # def summary(self):
    #     summary(self, [(1, 32, 200)])


class UrduRecongizer(nn.Module):
    def __init__(self, vocab_size, backbone, num_recurrent_units):
        self.num_recurrent_units = num_recurrent_units
        super(UrduRecongizer, self).__init__()
        if backbone == "MultiscaleCNN":
            self.backbone = MultiscaleCNN()
        elif backbone == "OnescaleCNN":
            self.backbone = OnescaleCNN()
        elif backbone == "OnescaleCNN_NOBN":
            self.backbone = OnescaleCNN_NOBN()
        elif backbone == "UNet":
            self.backbone = UNet(1)
        elif backbone == "DebugModel":
            self.backbone = DebugModel()
        else:
            print("Backbone not valid")
            exit(-1)

        self.attention = AttentionDecoder(num_recurrent_units,
                                          256, 256,
                                          vocab_size)
        print("Initialized Recognizer with alph_size", vocab_size)

    def forward(self, inputs, save_dir, epoch):
        emb = self.backbone(inputs, save_dir, epoch)
        ans, att_w = self.attention(emb, save_dir)
        # return torch.nn.LogSoftmax(dim=1)(ans), att_w
        return ans, att_w

    def summary(self):
        summary(self, [(1, 32, 200)])


class UrduRecongizerDebug(nn.Module):
    def __init__(self, vocab_size, backbone, num_recurrent_units):
        self.num_recurrent_units = num_recurrent_units
        super(UrduRecongizerDebug, self).__init__()
        if backbone == "MultiscaleCNN":
            self.backbone = MultiscaleCNN()
        elif backbone == "OnescaleCNN":
            self.backbone = OnescaleCNN()
        elif backbone == "OnescaleCNN_NOBN":
            self.backbone = OnescaleCNN_NOBN()
        elif backbone == "UNet":
            self.backbone = UNet(1)
        elif backbone == "DebugModel":
            self.backbone = DebugModel()
        else:
            print("Backbone not valid")
            exit(-1)

        self.attention = AttentionDecoder(num_recurrent_units,
                                          256, 256,
                                          vocab_size)
        print("Initialized Recognizer with alph_size", vocab_size)

    def forward(self, inputs, save_dir=None):
        emb = self.backbone(inputs, save_dir)
        ans, att_w = self.attention(emb, save_dir)
        return ans, att_w

    def summary(self):
        summary(self, [(1, 32, 200)])
