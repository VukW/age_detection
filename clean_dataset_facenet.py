import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from imdb_dataset import IMDBDataset
from facenet_pytorch import MTCNN

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = MTCNN(image_size=220, margin=20, device=device, min_face_size=150, select_largest=True)

bad_images_file_path = 'imdb_dataset_bad_images.json'


dataset = IMDBDataset('imdb_crop', transforms=None, preload_images=False)
loader = DataLoader(dataset=dataset, batch_size=32)

result = {}

pbar_multiple = tqdm(total=len(dataset), desc='multiple faces', position=0)
pbar_empty = tqdm(total=len(dataset), desc='no faces', position=1)
pbar_ok = tqdm(total=len(dataset), desc='ok', position=2)
for ic in tqdm(range(len(dataset)), position=3):
    img, target = dataset[ic]
    desc = dataset.imgs[ic]
    inputs = [img]
    outpt, probs = model(inputs,
                         save_path=os.path.join('imdb_crop_clean_220', desc.folder, desc.file_name),
                         return_prob=True)
    if len(outpt) > 1:
        # That should never happen because of choosen params, but still is left for a case
        result[desc.path] = 'multiple'
        pbar_multiple.update(1)
    elif not outpt:
        result[desc.path] = 'empty'
        pbar_empty.update(1)
    elif outpt[0] is None:
        result[desc.path] = 'empty'
        pbar_empty.update(1)
    else:
        # result[desc.desc.path] = 'ok'
        pbar_ok.update(1)
    if ic % 3000 == 0:
        with open(bad_images_file_path, 'w') as fout:
            json.dump(result, fout)
with open(bad_images_file_path, 'w') as fout:
    json.dump(result, fout)
