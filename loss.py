import torch
from torch import nn

def ChooseLoss(cfg):
    loss_type = cfg["type"]
    if loss_type == 'triplet':
        loss = nn.TripletMarginLoss(margin=cfg['alpha_margin']).cuda()
    elif loss_type == 'contrastive':
        loss = ContrastiveLoss(margin=cfg['alpha_margin']).cuda()
    elif loss_type == 'BCELoss':
        loss = nn.BCELoss().cuda()
    elif loss_type == 'BCEWithLogitsLoss':
        loss = nn.BCEWithLogitsLoss().cuda()
    elif loss_type == 'metric':
        loss = MetricLoss(margin=cfg['alpha_margin'])
    else:
        print('No Loss found!')
        exit(-1)
    return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin  # margin or radius

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        pos = (1 - label) * torch.pow(euclidean_distance, 2)
        neg = (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

        loss_contrastive = torch.mean(pos + neg)
        return loss_contrastive

        # euc_dist = torch.nn.functional.pairwise_distance(y1, y2)
        #
        # if d == 0:
        #     return torch.mean(torch.pow(euc_dist, 2))  # distance squared
        # else:  # d == 1
        #     delta = self.margin - euc_dist  # sort of reverse distance
        #     delta = torch.clamp(delta, min=0.0, max=None)
        #     return torch.mean(torch.pow(delta, 2))  # mean over all rows


class MetricLoss(nn.Module):
    def __init__(self, margin):
        super(MetricLoss, self).__init__()
        self.margin = margin

    def forward(self, anc, pos, neg, anc_ideal, pos_ideal, neg_ideal):
        ro, tau, xi = 0.1, 1.0, 1.0

        d_AN = nn.functional.pairwise_distance(anc, neg)
        d_AP = nn.functional.pairwise_distance(anc, pos)

        d_AIdeal = nn.functional.pairwise_distance(anc, anc_ideal)
        d_PIdeal = nn.functional.pairwise_distance(pos, pos_ideal)
        d_NIdeal = nn.functional.pairwise_distance(neg, neg_ideal)

        f = nn.Softplus(threshold=20000)
        triplet_loss = nn.TripletMarginLoss()

        g1 = ro * d_AP
        g2 = tau * f(d_AP - d_AN + self.margin)  # triplet_loss(anc, pos, neg) #
        g3 = xi * (d_AIdeal + d_PIdeal + d_NIdeal) / 3.0

        loss_metric = (g1 * g1) + (g2 * g2) + (g3 * g3)
        return loss_metric.mean()
