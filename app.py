from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.serialization
import torchvision

# CheXNet class labels
CHEXNET_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

# Load CheXNet model at startup
chexnet_model = None
def load_chexnet_model():
    global chexnet_model
    chexnet_model = models.densenet121(weights=None)
    num_ftrs = chexnet_model.classifier.in_features
    chexnet_model.classifier = torch.nn.Linear(num_ftrs, 14)
    checkpoint = torch.load('chexnet_checkpoint.pth', map_location=torch.device('cpu'), weights_only=False)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        # If it's a model object, get its state_dict
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        # Fix classifier key names if needed
        if "classifier.0.weight" in new_state_dict and "classifier.0.bias" in new_state_dict:
            new_state_dict["classifier.weight"] = new_state_dict.pop("classifier.0.weight")
            new_state_dict["classifier.bias"] = new_state_dict.pop("classifier.0.bias")
        chexnet_model.load_state_dict(new_state_dict)
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        chexnet_model.load_state_dict(new_state_dict)
    else:
        chexnet_model.load_state_dict(checkpoint)
    chexnet_model.eval()

load_chexnet_model()

# Image preprocessing for CheXNet
chexnet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    diagnosis = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Placeholder for model prediction
            diagnosis = predict_xray(filepath)
            return render_template('result.html', diagnosis=diagnosis, filename=filename)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def predict_xray(image_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = chexnet_transform(img).unsqueeze(0)
    with torch.no_grad():
        output = chexnet_model(input_tensor)
        probs = torch.sigmoid(output).squeeze().numpy()
    # Get top 3 findings with highest probability
    top_indices = probs.argsort()[-3:][::-1]
    results = []
    for idx in top_indices:
        results.append(f"{CHEXNET_LABELS[idx]}: {probs[idx]*100:.1f}%")
    return ", ".join(results)

torch.serialization.add_safe_globals([torchvision.models.densenet.DenseNet])

if __name__ == '__main__':
    app.run(debug=True)