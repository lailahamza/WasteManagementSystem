from flask import Flask, request, jsonify

from app.torch_utils import image_transformation, predict_image

app = Flask(__name__)

ALLOWED_EXTENSION = {'png', 'jpeg', 'jpg'}


def allowed_file(filename):
    # xx.png
    return '.' in filename and filename.rsplit('.', 1)[
        1].lower() in ALLOWED_EXTENSION  # this will return true if our filename is xxx.ext


# STEPS :
# 1 load image
# 2 image -> tensor
# 3 prediction
# 4 return json

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

    try:
        img_bytes = file.read()
        tensor = image_transformation(img_bytes)
        prediction = predict_image(tensor)
        data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
        return jsonify(data)

    except:
        return jsonify({'error': 'error during prediction'})
