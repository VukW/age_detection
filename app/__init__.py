from io import BytesIO
import base64

import requests
from PIL import Image
from flask import Flask, request, render_template, flash, redirect
from flask_bootstrap import Bootstrap

from models import FullModel, PredictionError

app = Flask(__name__)
bootstrap = Bootstrap(app)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = FullModel("age_model_latest.state", n_tta_transforms=3)


@app.route('/', methods=['GET', 'POST'])
def hello():
    name = request.args.get("name", "World")
    return render_template("index.html", name=name)


@app.route('/result', methods=['POST'])
def predict():
    image: Image.Image = None
    url = request.form.get('url')
    snapshot_base64 = request.form.get('snapshot')
    if snapshot_base64:
        data = base64.b64decode(snapshot_base64.split(',')[-1])
        image = Image.open(BytesIO(data))
    elif url:
        file = requests.get(url)
        image = Image.open(BytesIO(file.content))
    elif request.files.get('f'):
        first_file = request.files['f']
        # if user does not select file, browser also
        # submit an empty part without filename
        if (first_file.filename != '') and first_file.content_type in {'image/png', 'image/jpeg'}:
            image = Image.open(first_file)

    if not image:
        flash('No file chosen', category="error")
        return redirect('/')

    image = image.convert('RGB')

    try:
        face, predicted_age = model.predict(image)
    except PredictionError as e:
        flash(e.msg, "error")
        return redirect('/')

    buffered = BytesIO()
    face.thumbnail((224, 224))
    face.save(buffered, format="PNG")
    base64_content = "data: image/png; base64, " + base64.b64encode(buffered.getvalue()).decode("utf-8")
    flash(base64_content)

    return render_template('result.html', predicted_age=predicted_age * 100)
