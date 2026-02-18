# rpoutes.py
from flask import Blueprint, render_template, session, redirect, url_for, request, current_app, send_from_directory, flash
from functools import wraps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import os
import uuid
import cv2

bp = Blueprint(
    'app4',
    __name__,
    template_folder='templates',
    static_folder='static'
)

INSTANCE_DIR = os.path.join(os.path.dirname(__file__), 'instance')
MAX_IMAGE_SIZE = 10 * 1024 * 1024

# ------------------------
# Login required decorator
# ------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('app2.login'))
        return f(*args, **kwargs)
    return decorated_function

# ------------------------
# ResNet-18 Model
# ------------------------
class ResNetModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

# ------------------------
# Grad-CAM Implementation
# ------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        grad = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        if grad is not None:
            self.gradients = grad.detach()

    def __call__(self, input_tensor, class_idx=None):
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.clone().detach().requires_grad_(True).to(device)

        outputs = self.model(input_tensor)
        if class_idx is None:
            class_idx = outputs.argmax(dim=1).item()

        self.model.zero_grad()
        outputs[:, class_idx].sum().backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed")

        grads = self.gradients[0].cpu().numpy()
        acts = self.activations[0].cpu().numpy()
        weights = np.mean(grads, axis=(1, 2))

        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for idx, weight in enumerate(weights):
            cam += weight * acts[idx]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam /= (cam.max() + 1e-8)

        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        return cam
# ------------------------
# Device setup
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Image transforms
# ------------------------
test_transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ------------------------
# Load trained model
# ------------------------
model_path = os.path.join(os.path.dirname(__file__), 'breast_cancer_cnn_model_updated.pth')
model = ResNetModel(num_classes=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

gradcam = GradCAM(model, model.resnet.layer4[-1])

# ------------------------
# Helpers
# ------------------------
def _ensure_instance_dir():
    os.makedirs(INSTANCE_DIR, exist_ok=True)


def _clear_instance_dir():
    _ensure_instance_dir()
    for name in os.listdir(INSTANCE_DIR):
        path = os.path.join(INSTANCE_DIR, name)
        if os.path.isfile(path):
            os.remove(path)


def _save_plot(fig, filename):
    _ensure_instance_dir()
    path = os.path.join(INSTANCE_DIR, filename)
    fig.savefig(path, format='png', bbox_inches='tight')
    plt.close(fig)
    return path

def create_cam_overlay(original_pil, cam_map, out_path, alpha=0.45):
    """Blend Grad-CAM map with original image using a simple gradient colormap."""
    orig = np.array(original_pil).astype(np.float32) / 255.0
    h, w = orig.shape[:2]

    cam_resized = cv2.resize(cam_map, (w, h))
    cam_resized -= cam_resized.min()
    cam_resized /= (cam_resized.max() + 1e-8)

    cmap = plt.get_cmap('inferno')
    heatmap = cmap(cam_resized)[..., :3]

    overlay = (heatmap * alpha) + (orig * (1 - alpha))
    overlay = np.clip(overlay, 0, 1)
    overlay_uint8 = (overlay * 255).astype(np.uint8)

    Image.fromarray(overlay_uint8).save(out_path)
    return out_path

def clear_upload_folder(folder_path):
    if not os.path.isdir(folder_path):
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass


@bp.route('/instance/<path:filename>')
def instance_file(filename):
    _ensure_instance_dir()
    return send_from_directory(INSTANCE_DIR, filename)

# ------------------------
# Main route
# ------------------------
@bp.route('/', methods=['GET', 'POST'])
@login_required
def page():
    if request.method == 'POST':
        if 'images' not in request.files:
            return render_template('home3.html', error="No image uploaded")

        files = request.files.getlist('images')
        if len(files) == 0:
            return render_template('home3.html', error="No image selected")

        upload_folder = os.path.join(current_app.static_folder, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        clear_upload_folder(upload_folder)
        _clear_instance_dir()

        results = []

        for file in files:
            if file and file.filename != '':
                try:
                    file.stream.seek(0, os.SEEK_END)
                    file_size = file.stream.tell()
                    file.stream.seek(0)
                    if file_size > MAX_IMAGE_SIZE:
                        flash(f"File too large: {file.filename} exceeds 10 MB limit.")
                        continue

                    # Save uploaded image
                    image_filename = file.filename
                    image_path = os.path.join(upload_folder, image_filename)
                    file.save(image_path)

                    # Load & preprocess
                    image = Image.open(image_path).convert('RGB')
                    preprocessed_image = test_transform(image).unsqueeze(0).to(device)

                    # Predict
                    with torch.no_grad():
                        outputs = model(preprocessed_image)
                        probs = F.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probs, 1)
                        class_names = ['IDC(-)', 'IDC(+)']
                        predicted_class = class_names[predicted.item()]
                        # Map to user-friendly cancer status
                        cancer_status = "Cancer Positive" if predicted_class == "IDC(+)" else "Cancer Negative"

                        prob_idc_neg, prob_idc_pos = probs[0][0].item(), probs[0][1].item()

                    # Grad-CAM
                    pred_idx = int(predicted.item())
                    cam_map = gradcam(preprocessed_image, class_idx=pred_idx)
                    cam_filename = f"cam_{os.path.splitext(image_filename)[0]}.png"
                    cam_path = os.path.join(upload_folder, cam_filename)
                    create_cam_overlay(image, cam_map, cam_path, alpha=0.5)

                    # Explanation
                    heat_strength = cam_map.mean()
                    heat_phrase = "strong localized regions" if heat_strength > 0.35 else "moderate regions" if heat_strength > 0.15 else "diffuse regions"
                    if predicted_class == "IDC(+)":
                        explanation_text = (
                            f"The model predicts Cancer Positive (IDC+) with a confidence of {confidence.item()*100:.2f}%. "
                            f"This indicates the presence of Invasive Ductal Carcinoma (IDC), a common type of breast cancer where abnormal cells are detected in the breast tissue. "
                            f"The prediction is based on patterns identified in the image by a deep learning model (ResNet-18), which was trained on histopathology images to distinguish between cancerous and non-cancerous tissues. "
                            f"The Grad-CAM visualization shows {heat_phrase} of focus, highlighting areas in the image that the model considers most indicative of cancer. "
                            f"Brighter regions in the heatmap suggest higher importance in the model's decision. "
                            f"Please note that this is an automated prediction and should not be considered a definitive diagnosis. Consult a medical professional for a comprehensive evaluation."
                        )
                    else:
                        explanation_text = (
                            f"The model predicts Cancer Negative (IDC-) with a confidence of {confidence.item()*100:.2f}%. "
                            f"This suggests that the image does not show signs of Invasive Ductal Carcinoma (IDC), indicating the absence of cancerous cells in the analyzed tissue. "
                            f"The prediction is made by a deep learning model (ResNet-18) trained to identify patterns in histopathology images. "
                            f"The Grad-CAM visualization shows {heat_phrase} of focus, indicating the areas the model analyzed to make this prediction. "
                            f"Brighter regions in the heatmap highlight areas of interest, though in this case, they support a non-cancerous prediction. "
                            f"While this result is encouraging, it is not a substitute for a professional medical diagnosis. Please consult a healthcare provider for confirmation."
                        )

                    # Image plots
                    fig_img = plt.figure(figsize=(12, 4.5), facecolor="#0a0a0a")
                    plt.subplot(1, 3, 1)
                    plt.imshow(image)
                    plt.title(f"Original Image\nPrediction: {predicted_class} ({confidence.item():.2f})", color="#f4f2ef")
                    plt.axis('off')

                    preprocessed_img_np = preprocessed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    preprocessed_img_np = preprocessed_img_np * 0.5 + 0.5
                    preprocessed_img_np = np.clip(preprocessed_img_np, 0, 1)
                    plt.subplot(1, 3, 2)
                    plt.imshow(preprocessed_img_np)
                    plt.title("Preprocessed Image", color="#f4f2ef")
                    plt.axis('off')

                    cam_display = Image.open(cam_path).convert('RGB')
                    plt.subplot(1, 3, 3)
                    plt.imshow(cam_display)
                    plt.title("Grad-CAM Overlay", color="#f4f2ef")
                    plt.axis('off')

                    plt.tight_layout()
                    image_plot_name = f"image_plot_{uuid.uuid4().hex}.png"
                    _save_plot(fig_img, image_plot_name)

                    # Probability plot
                    fig_prob = plt.figure(figsize=(7, 3.5), facecolor="#0a0a0a")
                    classes = ['IDC(-)', 'IDC(+)']
                    probabilities = [prob_idc_neg, prob_idc_pos]
                    ax = sns.barplot(x=probabilities, y=classes, color="#00e5ff")
                    ax.set_facecolor("#0a0a0a")
                    plt.xlim(0, 1)
                    plt.xlabel("Probability", color="#f4f2ef")
                    plt.title("Prediction Probability Distribution", color="#f4f2ef")
                    ax.tick_params(colors="#d7d2cc")
                    for spine in ax.spines.values():
                        spine.set_color("#2a221b")
                    prob_plot_name = f"prob_plot_{uuid.uuid4().hex}.png"
                    _save_plot(fig_prob, prob_plot_name)

                    # Append result
                    results.append({
                        "filename": image_filename,
                        "predicted_class": predicted_class,
                        "cancer_status": cancer_status,
                        "confidence": f"{confidence.item():.4f}",
                        "prob_idc_neg": f"{prob_idc_neg:.4f}",
                        "prob_idc_pos": f"{prob_idc_pos:.4f}",
                        "image_plot": url_for('app4.instance_file', filename=image_plot_name),
                        "prob_plot": url_for('app4.instance_file', filename=prob_plot_name),
                        "uploaded_image": f"uploads/{image_filename}",
                        "cam_image": f"uploads/{cam_filename}",
                        "explanation_text": explanation_text
                    })

                except Exception as e:
                    return render_template('home3.html', error=f"Error processing image {file.filename}: {str(e)}")

        return render_template('home3.html', results=results)

    return render_template('home3.html')