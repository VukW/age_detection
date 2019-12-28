import json

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm.auto import tqdm

from imdb_dataset import IMDBDataset
from facenet_pytorch import MTCNN

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = MTCNN(image_size=224, margin=0, device=device)

transform = transforms.Compose([transforms.Resize((224, 224))])


dataset = IMDBDataset('imdb_crop', transforms=transform)
loader = DataLoader(dataset=dataset, batch_size=32)

result = {}

pbar_multiple = tqdm(total=len(dataset), desc='multiple faces', position=0)
pbar_empty = tqdm(total=len(dataset), desc='no faces', position=1)
pbar_ok = tqdm(total=len(dataset), desc='ok', position=2)
for ic in tqdm(range(len(dataset)), position=3):
    img, target = dataset[ic]
    desc = dataset.imgs[ic]
    inputs = [img]
    outpt = model(inputs)
    if len(outpt) > 1:
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
        with open('cleaned_dataset.json', 'w') as fout:
            json.dump(result, fout)
with open('cleaned_dataset.json', 'w') as fout:
    json.dump(result, fout)
