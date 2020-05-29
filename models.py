device = "cpu"

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


# class FineTunedMTCNN(MTCNN):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.
#
#
#     def forward(self, img, save_path=None, return_prob=False):
#         return super().forward(img, save_path, return_prob)
class FineTunedResnet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        keypoint_rcnn = keypointrcnn_resnet50_fpn(pretrained=pretrained)
        self.backbone = keypoint_rcnn.backbone
        self.fc = nn.Linear(64 * (49**2 + 24**2 + 12**2 + 5**2 + 2**2), 1)
        self.head_conv0 = nn.Conv2d(256, 64, (7, 7))
        self.head_conv1 = nn.Conv2d(256, 64, (5, 5))
        self.head_conv2 = nn.Conv2d(256, 64, (3, 3))
        self.head_conv3 = nn.Conv2d(256, 64, (3, 3))
        self.head_conv_pool = nn.Conv2d(256, 64, (3, 3))

    def forward(self, x, **kwargs):
        features = self.backbone.forward(x, **kwargs)
        convolved_0 = self.head_conv0.forward(features[0])
        convolved_1 = self.head_conv1.forward(features[1])
        convolved_2 = self.head_conv2.forward(features[2])
        convolved_3 = self.head_conv3.forward(features[3])
        convolved_pool = self.head_conv_pool.forward(features['pool'])

        flattened = [torch.flatten(v, 1)
                     for v in [convolved_0, convolved_1, convolved_2, convolved_3, convolved_pool]]
        x = self.fc(torch.cat(flattened, dim=1))
        return torch.sigmoid(x)

    def freeze(self, n_last_unfreezed=3):
        children = list(self.children())
        for ic, child in enumerate(children):
            requires_grad = ic > len(children) - n_last_unfreezed
            for param in child.parameters():
                param.requires_grad = requires_grad

    def unfreeze(self):
        for child in self.children():
            for param in child.parameters():
                param.requires_grad = True


def finetuned_resnet50(pretrained=False):
    model = FineTunedResnet(pretrained=pretrained)
    return model
