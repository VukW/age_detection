import json
import os
from dataclasses import dataclass

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm


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
    def __init__(self, root, transforms, numbers_list=None, bad_images=None, preload_images=False):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = []
        self.imgs_data = []
        if numbers_list:
            numbers_list = [os.path.join(root, part) for part in numbers_list]
        if isinstance(bad_images, str):
            bad_images = self._read_bad_images_from_json(bad_images)
        bad_images = bad_images or []
        for folder, dirs_list, files_list in os.walk(root):
            if numbers_list is not None and folder not in numbers_list:
                continue

            for filename in files_list:
                if not filename.endswith('.jpg'):
                    continue
                nm, rm, dob, photo_year = str(filename).split('.')[0].split('_')
                photo_year = int(photo_year)
                desc = ImageDescription(folder, filename, nm, rm, dob, photo_year)
                if desc.path in bad_images:
                    continue
                if desc.age > 100:
                    continue
                self.imgs.append(desc)

        self.imgs = sorted(self.imgs, key=lambda img: os.path.join(img.folder, img.file_name))
        if preload_images:
            for desc in tqdm(self.imgs):
                self.imgs_data.append(self._read_image(desc.path))

    def __getitem__(self, idx):
        # load images ad masks
        desc = self.imgs[idx]
        if self.imgs_data:
            img: Image = self.imgs_data[idx]
        else:
            img: Image = self._read_image(desc.path)

        image_id = torch.tensor([idx])

        target = {}
        target["image_id"] = image_id
        target["age"] = desc.age / 100

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def _read_bad_images_from_json(bad_images_filepath):
        with open(bad_images_filepath, 'r') as fin:
            bad_images = json.load(fin)  # dict filename -> blocking_reason
            return set(bad_images)

    @staticmethod
    def _read_image(img_path):
        img = Image.open(img_path).convert("RGB")
        return img
