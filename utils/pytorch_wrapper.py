from abc import ABC, abstractmethod
from typing import List

from PIL.Image import Image
import torch
from torchvision.transforms import functional as F


class VerboseCallback(ABC):
    @abstractmethod
    def call(self, batch_number, losses):
        pass


class VisdomCallback(VerboseCallback):
    def __init__(self, plotter, epoch):
        self.plotter = plotter
        self.epoch = epoch

    def call(self, batch_number, losses):
        self.plotter.plot(f'loss_epoch_{self.epoch + 1}', 'train', 'Batch loss',
                          batch_number, losses[-1])


def train_epoch(model, train_loader, optimizer, criteria,
                scheduler=None,
                device=None,
                verbose_frequency=None,
                verbose_callbacks: List[VerboseCallback] = None):
    losses = []
    device = device or torch.device('cpu')
    for batch_number, data in enumerate(train_loader):
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
        if verbose_frequency:
            if batch_number % verbose_frequency == verbose_frequency - 1:
                for callback in verbose_callbacks:
                    callback.call(batch_number, losses)


def evaluate_loss(model, data_loader, criteria, device=None):
    outputs, targets = infer(model, data_loader, device=device, with_target=True)
    loss = criteria(outputs, targets)
    return loss.item()


def infer(model, data_loader, device=None, with_target=False):
    outputs = []
    targets = []
    device = device or torch.device('cpu')
    for batch_number, data in enumerate(data_loader):
        inputs = data
        if with_target:
            inputs, target = data
            target = target['age'].float().view(-1, 1).to(device)
            targets.append(target)
        inputs = inputs.to(device)

        outputs.append(model(inputs))

        if with_target:
            return torch.cat(outputs), torch.cat(targets)
        else:
            return torch.cat(outputs)


def infer_image(model, image, device=None):
    device = device or torch.device('cpu')
    data = image.unsqueeze(0)
    return model(data.to(device))
