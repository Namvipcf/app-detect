from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import os
import numpy as np
import torch
from torchvision import models
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import io
import base64


app = Flask(__name__)

# Load model
def get_transforms(train=False):
    if train:
        transform = A.Compose([
            A.Resize(600, 600), # our input size can be 600px
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.1),
            A.ColorJitter(p=0.1),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.Compose([
            A.Resize(600, 600), # our input size can be 600px
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    return transform

# dataset_path = "Dataset"
# coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
# categories = coco.cats
# n_classes = len(categories.keys()) + 1
# classes = [i[1]['name'] for i in categories.items()]

classes = ['Classes', 'oto', 'xe bus', 'xe dap', 'xe may', 'xe tai']  # Replace with your actual classes
n_classes = len(classes) + 1

model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(torch.load('Model_final.pth', map_location=device))
model.eval()

# Function to predict image
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    transform = get_transforms(False)
    transformed = transform(image=image, bboxes=[])
    image = transformed['image']
    image = image.float().div(255)
    image = image.unsqueeze(0)
    with torch.no_grad():
        prediction = model(image)
    pred = prediction[0]
    # pred = {k: v[pred['scores'] > 0.5] for k, v in pred.items()}
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(draw_bounding_boxes(image[0].mul(255).byte(),
                                   pred['boxes'][pred['scores'] > 0.5],
                                   [classes[i] for i in pred['labels'][pred['scores'] > 0.5].tolist()],
                                   width=4
                                  ).permute(1, 2, 0))
    # Instead of saving the image to a file, save it to a byte array
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)  # Move the pointer back to the beginning of the buffer

    return buffer


# Function to load image
import os

@app.route('/', methods=['GET', 'POST'])
def load_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            # Create 'uploads' directory if it doesn't exist
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            file_path = os.path.join('uploads', filename)
            file.save(file_path)  # Save the file
            image_buffer = predict_image(file_path)
            return send_file(image_buffer, mimetype='image/png')

    return render_template('index.html')
        
@app.route('/', methods=['GET', 'POST'])
def index():
    result_img = None
    if request.method == 'POST':
        result_img = load_image()
    return render_template('index.html', result_img=result_img)

if __name__ == "__main__":
    app.run(debug=True)