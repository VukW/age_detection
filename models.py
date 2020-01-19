import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock


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

class FineTunedResnet(ResNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = super().forward(x)
        return torch.sigmoid(x)


def finetuned_resnet34():
    model = FineTunedResnet(block=BasicBlock, layers=[3, 4, 6, 3])
    return model
