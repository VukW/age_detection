A simple demo service of recognizing user's age by photo. Live demo can be found here: https://vukw-age-detection.xyz/

### Model training

Finally, I've used non-pretrained ResNet50 model with FPN, taken from  https://pytorch.org/docs/stable/torchvision/models.html#keypoint-r-cnn  

#### Data

Used dataset: IMDB-Wiki, https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
To load dataset, run

```(bash)
load_dataset.sh
```
    
#### Clean faces

Dataset is a bit dirty; there are images without faces, or images with too bad quality. I used FaceNet (https://pypi.org/project/facenet-pytorch/) to clean data, cut faces and save images. To reproduce, run

```(bash)
python clean_dataset_facenet.py
```

#### Visdom server

Start visdom server to store & plot loss during model fitting

```(bash)
python -m visdom.server
```

#### Model training

```(bash)
python train.py
```

It saves model state after every epoch to the current directory. Pretrained state (`age_model_latest.state`) can be found in this repo. It gives about 0.2 MAPE
   
### Model serving

Web UI uses Flask + bootstrap to upload images & predict ages. You can start an app like this:

```(bash)
env FLASK_APP=app flask run
```

Live demo is located here: https://vukw-age-detection.xyz/
