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
from facenet_pytorch.models.mtcnn import MTCNN
from torch.hub import load_state_dict_from_url
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, DataLoader, Dataset
import time

from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, model_urls

from imdb_dataset import IMDBDataset
from utils.visdom import VisdomLinePlotter

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

    # train_dataset = IMDBDataset('imdb_crop', transforms=transform,
    #                             numbers_list=[str(100 + ic)[-2:] for ic in range(60)],
    #                             bad_images='imdb_dataset_bad_images.json')
    val_dataset = IMDBDataset('imdb_crop', transforms=transform,
                              numbers_list=[str(100 + ic)[-2:] for ic in range(60, 100)],
                              bad_images='imdb_dataset_bad_images.json')
    print(f"val: {len(val_dataset)}")

    model = torch.load('age_model.pth')
    model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    print('creating loaders..')
    val_loader = DataLoader(dataset=val_dataset, batch_size=20)

    plotter = VisdomLinePlotter(env_name='Infer quality')

    start = time.time()
    print('start infering..')
    try:
        losses = []
        criteria = nn.MSELoss()
        for i, data in enumerate(val_loader, 0):
            inputs, target = data
            inputs = inputs.to(device)
            target = target['age'].float().view(-1, 1).to(device)

            outputs = model(inputs)
            batch_loss = criteria(outputs, target)
            losses.append(batch_loss.item())

            if i % VERBOSE_FREQUENCY == VERBOSE_FREQUENCY - 1:
                print("[{}], Loss: {}".format(i + 1, np.mean(losses[-VERBOSE_FREQUENCY:])))

            plotter.plot(f'loss_infer', 'val', 'Infer batch loss', i, losses[-1])
    finally:
        # scheduler.step(epoch)
        print("Finished infering")
        print(time.time() - start, 'secs')
