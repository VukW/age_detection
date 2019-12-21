import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, DataLoader, Dataset

from utils.visdom import VisdomLinePlotter


@dataclass
class ImageDescription:
    # img_name satisfies the pattern NM_RM_DOB_YEAR, f.ex. nm0001400_rm4110713856_1965-9-9_1984
    # NM: idx of person on the image, `nmXXXXXXX`, 23k unique values
    # RM: idx of photo source, 306k unique values
    # DOB: date of birth, YYYY-mm-dd
    # YEAR: year when photo was made
    folder: str
    file_name: str
    nm: str
    rm: str
    dob: str
    photo_year: int

    def __post_init__(self):
        self.path = os.path.join(self.folder, self.file_name)

        year_of_birth = int(self.dob.split('-')[0])
        self.age = self.photo_year - year_of_birth


class IMDBDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = []
        for folder, dirs_list, files_list in os.walk(root):
            for filename in files_list:
                if not filename.endswith('.jpg'):
                    continue
                nm, rm, dob, photo_year = str(filename).split('.')[0].split('_')
                photo_year = int(photo_year)
                self.imgs.append(ImageDescription(folder, filename, nm, rm, dob, photo_year))

        self.imgs = sorted(self.imgs, key=lambda img: os.path.join(img.folder, img.file_name))

    def __getitem__(self, idx):
        # load images ad masks
        desc = self.imgs[idx]
        img: Image = Image.open(desc.path).convert("RGB")

        image_id = torch.tensor([idx])

        target = {}
        target["image_id"] = image_id
        target["age"] = desc.age / 100

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


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


if __name__ == '__main__':
    print("Please, start visdom with `python -m visdom.server` (default location: http://localhost:8097)")

    net = AgeModel()

    transform = transforms.Compose([transforms.Resize((220, 220)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    print('creating dataset..')
    dataset = IMDBDataset('imdb_crop/00', transforms=transform)

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    print(f"train:{train_size}, test: {test_size}")
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

    print('creating loaders..')
    train_loader = DataLoader(dataset=train_dataset, batch_size=16)
    val_loader = DataLoader(dataset=val_dataset, batch_size=20)

    criteria = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.6)

    plotter = VisdomLinePlotter(env_name='Tutorial Plots')

    print('start training..')
    for epoch in range(10):
        losses = []
        for i, data in enumerate(train_loader, 0):
            inputs, target = data

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criteria(outputs, target['age'].float().view(-1, 1))
            loss.backward()

            optimizer.step()

            losses.append(loss.item())

            if i % 10 == 9:
                print("[{}, {}], Loss: {}".format(epoch + 1, i + 1, np.mean(losses[-10:])))

            plotter.plot(f'loss_epoch_{epoch}', 'train', 'Batch loss', i, losses[-1])
        plotter.plot('loss', 'train', 'Epoch Loss', epoch, np.mean(losses))
        scheduler.step(epoch)
    print("Finished Training")

    torch.save(net, 'age_model.pth')
    print("Saved")

    # model = torch.load(PATH)
    # model.eval()
