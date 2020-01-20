import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision
from PIL import Image
import os
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, DataLoader, Dataset
import time

from imdb_dataset import IMDBDataset
from utils.visdom import VisdomLinePlotter, VisdomLinePrinter
from models import finetuned_resnet34


def infer_model(model, loader, device, with_target=False):
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


def save_model(model, postfix=None):
    fname = 'age_model'
    if postfix:
        fname += '_' + postfix
    fname += '.pth'
    torch.save(model, fname)


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
    model = finetuned_resnet34(pretrained=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    print('creating loaders..')
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=50, shuffle=False)

    criteria = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

    plotter = VisdomLinePlotter(env_name='Train quality')

    start = time.time()
    print('start training..')
    try:
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
                scheduler.step()

                losses.append(loss.item())

                if i % VERBOSE_FREQUENCY == VERBOSE_FREQUENCY - 1:
                    print("[{}, {}], Loss: {}".format(epoch + 1, i + 1, np.mean(losses[-VERBOSE_FREQUENCY:])))

                plotter.plot(f'loss_epoch_{epoch + 1}', 'train', 'Batch loss', i, losses[-1])
            plotter.plot('loss', 'train', 'Epoch Loss', epoch + 1, np.mean(losses))
            print(f"===[{int(time.time() - start)}] {epoch + 1}: loss {np.mean(losses)}")
            save_model(model, postfix='epoch_' + str(epoch + 1))
    finally:
        # scheduler.step(epoch)
        print("Finished Training")
        print(time.time() - start, 'secs')

        save_model(model)
        print("Saved")
