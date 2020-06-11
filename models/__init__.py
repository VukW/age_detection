from PIL.Image import Image
from facenet_pytorch.models.mtcnn import MTCNN, fixed_image_standardization
from facenet_pytorch.models.utils.detect_face import extract_face

from config import STORAGE_SUB_PATH

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models.detection import keypointrcnn_resnet50_fpn
import numpy as np

from utils.pytorch_wrapper import infer_image

device = torch.device("cpu")

augmentation = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   transforms.ToPILImage(),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ColorJitter(0.02, 0.02, 0.02, 0.02),
                                   transforms.RandomResizedCrop((224, 224), scale=(0.9, 1.0),
                                                                ratio=(9 / 10, 10 / 9)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


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
        features = self.backbone.forward(x, )
        convolved_0 = self.head_conv0.forward(features['0'])
        convolved_1 = self.head_conv1.forward(features['1'])
        convolved_2 = self.head_conv2.forward(features['2'])
        convolved_3 = self.head_conv3.forward(features['3'])
        convolved_pool = self.head_conv_pool.forward(features['pool'])

        flattened = [torch.flatten(v, 1)
                     for v in [convolved_0, convolved_1, convolved_2, convolved_3, convolved_pool]]
        x = self.fc(torch.cat(flattened, dim=1))
        return torch.sigmoid(x)

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


class PredictionError(BaseException):
    def __init__(self, msg):
        self.msg = msg


class CustomMTCNN(MTCNN):
    def forward(self, img, **kwargs):
        """Simplified MTCNN forward step; no saving to file, no prob returning, only one image per call,
        BUT it returns a bbox"""
        with torch.no_grad():
            box_im, prob_im = self.detect(img)

        # Process all bounding boxes and probabilities
        if box_im is None:
            return None, None

        if not self.keep_all:
            box_im = box_im[[0]]

        faces_im = []
        for i, box in enumerate(box_im):
            face = extract_face(img, box, self.image_size, self.margin, None)
            if self.post_process:
                face = fixed_image_standardization(face)
            faces_im.append(face)

        if self.keep_all:
            faces_im = torch.stack(faces_im)
        else:
            faces_im = faces_im[0]
            box_im = box_im[0]

        return faces_im, box_im


class FullModel:
    def __init__(self, age_model_state_filepath, n_tta_transforms=3):
        self.face_model = CustomMTCNN(image_size=224, margin=20, device=device, min_face_size=150, select_largest=True)

        self.age_model = finetuned_resnet50(pretrained=False)
        load_model_state(self.age_model, age_model_state_filepath)

        self.n_tta_transforms = n_tta_transforms
        self.norm_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.tta_transforms = augmentation

    def predict(self, image):
        if not isinstance(image, Image):
            raise PredictionError("No image found")

        image = image.convert('RGB')
        face, box = self.face_model(image)

        if face is None:
            raise PredictionError('No face found')

        face_img = transforms.ToPILImage()((face + 1) / 2)
        real_ratio = (box[3] - box[1]) / (box[2] - box[0])
        face_img = face_img.resize((224, int(real_ratio * 224)))

        prediction = []
        prediction.append(infer_image(self.age_model,
                                      self.norm_transform(face).unsqueeze(0)).item())
        for _ in range(self.n_tta_transforms):
            prediction.append(infer_image(self.age_model,
                                          self.tta_transforms(face).unsqueeze(0)).item())

        print(prediction)
        prediction = np.mean(prediction)
        return face_img, prediction
