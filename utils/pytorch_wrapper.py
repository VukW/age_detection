from abc import ABC, abstractmethod
from typing import List

import torch
from torchvision import transforms

augmentation = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   transforms.ToPILImage(),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ColorJitter(0.02, 0.02, 0.02, 0.02),
                                   transforms.RandomResizedCrop((224, 224), scale=(0.9, 1.0),
                                                                ratio=(9 / 10, 10 / 9)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


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


def evaluate_loss(model, data_loader, criteria, device=None, n_tta=0):
    outputs, targets = infer(model, data_loader, device=device, with_target=True, n_tta=n_tta)
    loss = criteria(outputs, targets)
    return loss.item()


def infer(model, data_loader, device=None, with_target=False, n_tta=0):
    outputs = []
    targets = []
    device = device or torch.device('cpu')
    with torch.no_grad():
        for batch_number, data in enumerate(data_loader):
            inputs = data
            if with_target:
                inputs, target = data
                target = target['age'].float().view(-1, 1).to(device)
                targets.append(target)

            batch_outputs = infer_images(model, inputs, n_tta=n_tta)
            outputs.append(batch_outputs)

    if with_target:
        return torch.cat(outputs), torch.cat(targets)
    else:
        return torch.cat(outputs)


def infer_images(model, images, device=None, n_tta=0):
    device = device or torch.device('cpu')
    if len(images.shape) == 3:
        images = images.unsqueeze(0).to(device)

    prediction = model(images)

    for _ in range(n_tta):
        augmented_images = torch.cat([augmentation(img).unsqueeze(0) for img in images])
        prediction += model(augmented_images.to(device))
    prediction /= 1 + n_tta
    return prediction
