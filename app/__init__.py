from io import BytesIO
import base64

from PIL import Image
from flask import Flask, request, render_template, flash, redirect, send_file
from flask_bootstrap import Bootstrap
from torchvision.transforms import transforms

from models import load_model_state, finetuned_resnet50, AgeModel
from utils.pytorch_wrapper import infer_image

app = Flask(__name__)
bootstrap = Bootstrap(app)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# model = finetuned_resnet50(pretrained=False)
model = AgeModel()
load_model_state(model, "age_model_custom_epoch_30.state")

common_transforms = [
    transforms.Resize((220, 220)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(common_transforms)


@app.route('/', methods=['GET', 'POST'])
def hello():
    name = request.args.get("name", "World")
    return render_template("index.html", name=name)


@app.route('/result', methods=['POST'])
def predict():
    file = None
    if request.files:
        first_file = list(request.files.values())[0]
        # if user does not select file, browser also
        # submit an empty part without filename
        if first_file.filename != '':
            file = first_file

    if file:
        base64_content = "data:" + file.content_type + ";base64, " + base64.b64encode(file.read()).decode("utf-8")
        flash(base64_content)
        image = Image.open(file)
        prediction = infer_image(model, transform(image))
        return render_template('result.html', predicted_age=prediction.item() * 100)

    flash('No file chosen', category="error")
    return redirect('/')