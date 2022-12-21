import torch.nn as nn
import torch


class RCNN(nn.Module):
    def __init__(self, vgg_base, label2target):

        super().__init__()
        feature_dim = 25088
        self.base_model = vgg_base
        self.cls_score = nn.Linear(feature_dim, len(label2target))

        self.bbox = nn.Sequential(nn.Linear(feature_dim, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, 4),
                                  nn.Tanh())

        self.cel = nn.CrossEntropyLoss()
        self.l1 = nn.L1Loss()

    def forward(self, single_input):

        features = self.base_model(single_input)

        reg = self.bbox(features)

        clss = self.cls_score(features)

        return clss, reg

    def calc_loss(self, probs, _deltas, labels, deltas):

        detection_loss = self.cel(probs, labels)

        ixs, = torch.where(labels != 0)

        _deltas = _deltas[ixs]
        deltas = deltas[ixs]

        self.lmb = 10

        if len(ixs) > 0:

            regression_loss = self.l1(_deltas, deltas)

            return detection_loss + self.lmb * regression_loss, \
                   detection_loss.detach(), regression_loss.detach()

        else:

            regression_loss = 0.0

            return detection_loss + self.lmb * regression_loss, \
                   detection_loss.detach(), regression_loss


def train_batch(inputs, model, optimizer, criterion):

    inpt, clss, deltas = inputs

    model.train()

    optimizer.zero_grad()

    _clss, _deltas = model(inpt)

    loss, ce_loss, l1_loss = criterion(_clss, _deltas, clss, deltas)

    _, _clss = _clss.max(-1)

    accs = clss == _clss

    loss.backward()

    optimizer.step()

    return loss.detach(), ce_loss, l1_loss, accs.cpu().numpy()

@torch.no_grad()
def validate_batch(inputs, model, criterion):

    inpt, clss, deltas = inputs()

    model.eval()

    _clss, _deltas = model(inpt)

    loss, ce_loss, l1_loss = criterion(_clss, _deltas, clss, deltas)

    _, _clss = _clss.max(-1)

    accs = clss == _clss

    return loss, ce_loss, l1_loss, accs.cpu().numpy()

