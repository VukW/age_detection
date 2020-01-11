import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, DataLoader, Dataset
import time

from imdb_dataset import IMDBDataset
from utils.visdom import VisdomLinePlotter


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


def resnet34():
    model = torchvision.models.resnet34(pretrained=False)
    model.fc = nn.Linear(512, 1)
    return model


def infer_model(loader, device, with_target=False):
    all_outputs = []
    all_targets = []
    for i, data in enumerate(loader):
        if with_target:
            inputs, target = data
            target = target['age'].float().view(-1, 1)
            all_targets.append(target)
        else:
            inputs = data
        inputs = inputs.to(device)

        outputs = model(inputs)
        all_outputs.append(outputs)

    if with_target:
        return torch.cat(all_outputs), torch.cat(all_targets)
    else:
        return torch.cat(all_outputs)


VERBOSE_FREQUENCY = 30

if __name__ == '__main__':
    print("Please, start visdom with `python -m visdom.server` (default location: http://localhost:8097)")

    transform = transforms.Compose([transforms.Resize((220, 220)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    print('creating dataset..')
    # dataset = IMDBDataset('imdb_crop/00', transforms=transform)

    # train_size = int(len(dataset) * 0.8)
    # test_size = len(dataset) - train_size
    # print(f"train:{train_size}, test: {test_size}")
    # train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

    train_dataset = IMDBDataset('imdb_crop', transforms=transform,
                                numbers_list=[str(100 + ic)[-2:] for ic in range(60)],
                                bad_images='imdb_dataset_bad_images.json')
    val_dataset = IMDBDataset('imdb_crop', transforms=transform,
                              numbers_list=[str(100 + ic)[-2:] for ic in range(60, 100)],
                              bad_images='imdb_dataset_bad_images.json')
    print(f"train:{len(train_dataset)}, val: {len(val_dataset)}")

    # model = AgeModel()
    model = resnet34()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    print('creating loaders..')
    train_loader = DataLoader(dataset=train_dataset, batch_size=8)
    val_loader = DataLoader(dataset=val_dataset, batch_size=20)

    criteria = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.6)

    plotter = VisdomLinePlotter(env_name='Tutorial Plots')

    start = time.time()
    print('start training..')
    for epoch in range(10):
        losses = []
        for i, data in enumerate(train_loader, 0):
            inputs, target = data
            inputs = inputs.to(device)
            target = target['age'].float().view(-1, 1).to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criteria(outputs, target)
            loss.backward()

            optimizer.step()

            losses.append(loss.item())

            if i % VERBOSE_FREQUENCY == VERBOSE_FREQUENCY - 1:
                print("[{}, {}], Loss: {}".format(epoch + 1, i + 1, np.mean(losses[-VERBOSE_FREQUENCY:])))

            plotter.plot(f'loss_epoch_{epoch + 1}', 'train', 'Batch loss', i, losses[-1])
        plotter.plot('loss', 'train', 'Epoch Loss', epoch + 1, np.mean(losses))
        print(f"==={epoch + 1}: loss {np.mean(losses)}")
        scheduler.step(epoch)
    print("Finished Training")
    print(time.time() - start, 'secs')

    torch.save(model, 'age_model.pth')
    print("Saved")

    # model = torch.load(PATH)
    # model.eval()
