from io import BytesIO
import base64

import requests
import torch
from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
from flask import Flask, request, render_template, flash, redirect, send_file
from flask_bootstrap import Bootstrap
from torchvision.transforms import transforms

from models import load_model_state, finetuned_resnet50, AgeModel, device
from utils.pytorch_wrapper import infer_image
from torchvision.transforms import functional as F

app = Flask(__name__)
bootstrap = Bootstrap(app)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

face_model = MTCNN(image_size=224, margin=20, device=device, min_face_size=150, select_largest=True)
age_model = finetuned_resnet50(pretrained=False)
load_model_state(age_model, "age_model_latest.state")
print(age_model)
# age_model = AgeModel()
# load_model_state(age_model, "age_model_custom_epoch_30.state")

common_transforms = [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(common_transforms)


@app.route('/', methods=['GET', 'POST'])
def hello():
    name = request.args.get("name", "World")
    return render_template("index.html", name=name)


@app.route('/result', methods=['POST'])
def predict():
    image: Image.Image = None
    if request.files:
        first_file = list(request.files.values())[0]
        # if user does not select file, browser also
        # submit an empty part without filename
        if first_file.filename != '':
            image = Image.open(first_file)

    if not image and request.form:
        url = request.form['url']
        snapshot_base64 = request.form['snapshot']
        if url:
            file = requests.get(url)
            image = Image.open(BytesIO(file.content))
        elif snapshot_base64:
            data = base64.b64decode(snapshot_base64.split(',')[-1])
            image = Image.open(BytesIO(data))

    if not image:
        flash('No file chosen', category="error")
        return redirect('/')

    image = image.convert('RGB')
    outpt = face_model(image)

    if outpt is None:
        flash('No face found', category="error")
        return redirect('/')

    face: Image = F.to_pil_image((outpt + 1) / 2)

    buffered = BytesIO()
    face.thumbnail((224, 224))
    face.save(buffered, format="PNG")
    base64_content = "data: image/png; base64, " + base64.b64encode(buffered.getvalue()).decode("utf-8")
    flash(base64_content)

    prediction = infer_image(age_model, transform(outpt).unsqueeze(0))
    return render_template('result.html', predicted_age=prediction.item() * 100)
