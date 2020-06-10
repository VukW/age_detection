import numpy as np
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import os
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import time

from imdb_dataset import IMDBDataset
from utils.pytorch_wrapper import train_epoch, evaluate_loss, VerboseCallback
from utils.visdom import VisdomLinePrinter
from models import finetuned_resnet50, save_model, get_model, load_model_state, FineTunedResnet


class VisdomCallback(VerboseCallback):
    def __init__(self, plotter, epoch):
        self.plotter = plotter
        self.epoch = epoch

    def call(self, batch_number, losses):
        self.plotter.plot(f'loss_epoch_{self.epoch + 1}', 'train', 'Batch loss',
                          batch_number, losses[-1])


class PrinterCallback(VerboseCallback):
    def __init__(self, epoch, smoothing_interval=1):
        self.plotter = plotter
        self.epoch = epoch
        self.smooth = smoothing_interval

    def call(self, batch_number, losses):
        smoothed_loss = np.mean(losses[-self.smooth:])
        print(f"[{epoch + 1}, {batch_number + 1}], Loss: {smoothed_loss}")


def MAPELoss(output, target):
    return torch.mean(torch.abs(output - target) / target)


VERBOSE_FREQUENCY = 2

if __name__ == '__main__':
    print("Please, start visdom with `python -m visdom.server` (default location: http://localhost:8097)")

    train_transforms = [transforms.RandomHorizontalFlip()]
    val_transforms = []
    common_transforms = [transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    train_transforms = transforms.Compose(train_transforms + common_transforms)
    val_transforms = transforms.Compose(val_transforms + common_transforms)

    print('creating dataset..')
    # dataset = IMDBDataset('imdb_crop/00', transforms=transform)

    # train_size = int(len(dataset) * 0.8)
    # test_size = len(dataset) - train_size
    # print(f"train:{train_size}, test: {test_size}")
    # train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

    train_dataset = IMDBDataset('imdb_crop_clean_224/imdb_crop', transforms=train_transforms,
                                numbers_list=[str(100 + ic)[-2:] for ic in range(60)],
                                preload_images=False)
    val_dataset = IMDBDataset('imdb_crop_clean_224/imdb_crop', transforms=val_transforms,
                              numbers_list=[str(100 + ic)[-2:] for ic in range(60, 100)],
                              preload_images=False)
    print(f"train:{len(train_dataset)}, val: {len(val_dataset)}")

    # model = AgeModel()
    model = finetuned_resnet50(pretrained=False)
    load_model_state(model, 'age_model_latest.state')
    # model = get_model('age_model_latest.pth')

    # model.freeze(3)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    print('creating loaders..')
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=50, shuffle=False)

    criteria = MAPELoss
    optimizer = optim.Adam(model.parameters(), lr=0.015)
    scheduler = StepLR(optimizer, step_size=60, gamma=0.99)

    plotter = VisdomLinePrinter(env_name='Train quality')

    model.eval()
    train_loss = evaluate_loss(model, train_loader, criteria, device=device)
    val_loss = evaluate_loss(model, val_loader, criteria, device=device)

    #     plotter.plot(f'loss_epoch_{epoch + 1}', 'train', 'Batch loss', i, losses[-1])
    print('train loss:', train_loss)
    print('val loss:', val_loss)

    try:
        start = time.time()
        print('start training..')

        for epoch in range(30):
            model.freeze(epoch + 2)
            model.train()
            train_epoch(model, train_loader,

                        device=device,

                        optimizer=optimizer,
                        scheduler=scheduler,
                        criteria=criteria,
                        verbose_frequency=VERBOSE_FREQUENCY,
                        verbose_callbacks=[VisdomCallback(plotter=plotter, epoch=epoch)]
                        )

            model.eval()
            train_loss = evaluate_loss(model, train_loader, criteria, device=device)
            val_loss = evaluate_loss(model, val_loader, criteria, device=device)

            #     plotter.plot(f'loss_epoch_{epoch + 1}', 'train', 'Batch loss', i, losses[-1])
            plotter.plot('loss', 'train', 'Epoch Loss', epoch + 1, train_loss)
            plotter.plot('loss', 'val', 'Epoch Loss', epoch + 1, val_loss)
            print(f"===[{int(time.time() - start)}] {epoch + 1}: train_loss {train_loss}, val_loss {val_loss}")
            save_model(model, postfix='epoch_' + str(epoch + 1))
            if epoch == 1:
                model.unfreeze()
    finally:
        # scheduler.step(epoch)
        print("Finished Training")
        print(time.time() - start, 'secs')

        save_model(model)
        print("Saved")
