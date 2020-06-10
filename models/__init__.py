from config import STORAGE_SUB_PATH

import os

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.hub import load_state_dict_from_url

from torchvision.models import ResNet
from torchvision.models.detection import KeypointRCNN, keypointrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.resnet import BasicBlock, model_urls

device = torch.device("cpu")


class AgeModel(nn.Module):
    def __init__(self):
        super(AgeModel, self).__init__()
        # 224 x 224
        self.conv1 = nn.Conv2d(3, 8, 5)  # (220 - 4) / 2 = 108
        self.conv2 = nn.Conv2d(8, 16, 5)  # (108 - 4) / 2 = 52
        self.conv3 = nn.Conv2d(16, 16, 5)  # (52 - 4) / 2 = 24
        self.conv4 = nn.Conv2d(16, 4, 5)  # (24 - 4) / 2 = 10

        self.fc1 = nn.Linear(4 * 10 * 10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        x = x.view(-1, 4 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FineTunedResnet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        keypoint_rcnn = keypointrcnn_resnet50_fpn(pretrained=pretrained)
        self.backbone = keypoint_rcnn.backbone
        self.head_conv0 = nn.Conv2d(256, 64, (7, 7))
        self.head_conv1 = nn.Conv2d(256, 64, (5, 5))
        self.head_conv2 = nn.Conv2d(256, 64, (3, 3))
        self.head_conv3 = nn.Conv2d(256, 64, (3, 3))
        self.head_conv_pool = nn.Conv2d(256, 64, (3, 3))
        self.fc = nn.Linear(64 * (50 ** 2 + 24 ** 2 + 12 ** 2 + 5 ** 2 + 2 ** 2), 1)

    def forward(self, x, **kwargs):
        features = self.backbone.forward(x, **kwargs)
        try:
            convolved_0 = self.head_conv0.forward(features['0'])
            convolved_1 = self.head_conv1.forward(features['1'])
            convolved_2 = self.head_conv2.forward(features['2'])
            convolved_3 = self.head_conv3.forward(features['3'])
            convolved_pool = self.head_conv_pool.forward(features['pool'])

            flattened = [torch.flatten(v, 1)
                         for v in [convolved_0, convolved_1, convolved_2, convolved_3, convolved_pool]]
            x = self.fc(torch.cat(flattened, dim=1))
            return torch.sigmoid(x)
        except:
            print({k: f.shape for k, f in features.items()})
            print(convolved_0.shape, convolved_1.shape, convolved_2.shape, convolved_3.shape, convolved_pool.shape)
            raise

    def flattened_children(self, depth):
        children = list(self.children())
        flattened_children = []
        if depth is None:
            depth = 100000  # just big enough
        for _ in range(depth):
            for child in children:
                grandchild = list(child.children())
                if grandchild:
                    flattened_children += grandchild
                else:
                    flattened_children.append(child)
            if children == flattened_children:
                break
            children, flattened_children = flattened_children, []
        return children

    def freeze(self, n_last_unfreezed=3, depth=3):
        children = self.flattened_children(depth=depth)
        for ic, child in enumerate(children):
            requires_grad = ic > len(children) - n_last_unfreezed
            for param in child.parameters():
                param.requires_grad = requires_grad

    def unfreeze(self, depth=3):
        children = self.flattened_children(depth=depth)
        for child in children:
            for param in child.parameters():
                param.requires_grad = True


def finetuned_resnet50(pretrained=False):
    model = FineTunedResnet(pretrained=pretrained)
    return model


def save_model(model, postfix=None):
    fname = os.path.join(STORAGE_SUB_PATH, 'age_model')
    if postfix:
        fname += '_' + postfix
    fname += '.pth'
    torch.save(model, fname)


def get_model(fname):
    fname = os.path.join(STORAGE_SUB_PATH, fname)
    return torch.load(fname, map_location=device)


def save_model_state(model, postfix=None):
    fname = os.path.join(STORAGE_SUB_PATH, 'age_model')
    if postfix:
        fname += '_' + postfix
    fname += '.state'
    torch.save(model.state_dict(), fname)


def load_model_state(model, fname):
    fname = os.path.join(STORAGE_SUB_PATH, fname)
    return model.load_state_dict(torch.load(fname, map_location=device))
