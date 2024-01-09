import os
import os.path as op

import torch
from model import *
from train_utils import export_fm

if __name__ == "__main__":
    torch.cuda.set_device(0)

    triplet_loss = nn.TripletMarginLoss(margin=15.0).cuda()
    model = KorNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_loss = 0.0

    anchor = torch.randn(1, 1, 37, 37, requires_grad=True).cuda()
    positive = torch.randn(1, 1, 37, 37, requires_grad=True).cuda()
    negative = torch.randn(1, 1, 37, 37, requires_grad=True).cuda()

    anchor.retain_grad()
    positive.retain_grad()
    negative.retain_grad()

    export_fm(anchor, '/home/arseniy/results/anchor_' + str(0) + '.txt')
    export_fm(positive, '/home/arseniy/results/positive_' + str(0) + '.txt')
    export_fm(negative, '/home/arseniy/results/negative_' + str(0) + '.txt')

    anchor_out, positive_out, negative_out = model(anchor), model(positive), model(negative)

    anchor_out.retain_grad()
    positive_out.retain_grad()
    negative_out.retain_grad()

    export_fm(anchor_out, '/home/arseniy/results/anchor_out_' + str(0) + '.txt')
    export_fm(positive_out, '/home/arseniy/results/positive_out_' + str(0) + '.txt')
    export_fm(negative_out, '/home/arseniy/results/negative_out_' + str(0) + '.txt')

    Loss = triplet_loss(anchor_out, positive_out, negative_out)

    print("epoch %d Train [Loss]", Loss.item())

    # optimizer.zero_grad()
    Loss.backward()

    export_fm(anchor_out.grad, '/home/arseniy/results/anchor_grad_' + str(0) + '.txt')
    export_fm(positive_out.grad, '/home/arseniy/results/positive_grad_' + str(0) + '.txt')
    export_fm(negative_out.grad, '/home/arseniy/results/negative_grad_' + str(0) + '.txt')

    # optimizer.step()

    # train_loss += Loss.item()

    print("epoch %d Train [Loss]", Loss.item())




