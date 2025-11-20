# routes.py
import base64
import io
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Blueprint, render_template, session, redirect, url_for, request, flash
from functools import wraps
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
from io import BytesIO

bp = Blueprint('app3', __name__, template_folder='templates')

# --------------------------------------------------------------- #
# 1. Login decorator
# --------------------------------------------------------------- #
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('app2.login'))
        return f(*args, **kwargs)
    return decorated

# --------------------------------------------------------------- #
# 2. Model & helpers (loaded **once**)
# --------------------------------------------------------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- 2.1  Load your trained checkpoint ---------------------------------
# Put the file next to routes.py  OR  inside a folder called "models"
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'resnet18_model_001.pth')
# If you keep it in a sub-folder, use:
# MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'resnet18_model_001.pth')

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# Initialise the architecture (no pretrained weights)
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 5)   # 5 classes

# Load the *state_dict* (this is what you saved with torch.save(model.state_dict(), ...))
try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
except Exception as e:
    raise RuntimeError(f"Failed to load model weights from {MODEL_PATH}: {e}")

model = model.to(device)
model.eval()

# --------------------------------------------------------------- #
# 2.2  Classes & status
# --------------------------------------------------------------- #
classes = [
    "Colon Adenocarcinoma",
    "Colon Benign Tissue",
    "Lung Adenocarcinoma",
    "Lung Benign Tissue",
    "Lung Squamous Cell Carcinoma"
]

cancer_status = {
    "Colon Adenocarcinoma": "Colon Adenocarcinoma Cancer Positive (+)",
    "Lung Adenocarcinoma": "Lung Adenocarcinoma Cancer Positive (+)",
    "Lung Squamous Cell Carcinoma": "Lung Squamous Cell Carcinoma Cancer Positive (+)",
    "Colon Benign Tissue": "Colon Benign Tissue Cancer Negative (-)",
    "Lung Benign Tissue": "Lung Benign Tissue Cancer Negative (-)"
}

# --------------------------------------------------------------- #
# 2.3  Image transform
# --------------------------------------------------------------- #
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --------------------------------------------------------------- #
# 2.4  Grad-CAM helper
# --------------------------------------------------------------- #
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, m, i, o):
        self.activations = o.detach()

    def _save_gradient(self, m, gi, go):
        g = go[0] if isinstance(go, tuple) else go
        if g is not None:
            self.gradients = g.detach()

    def __call__(self, input_tensor, class_idx=None):
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.clone().detach().requires_grad_(True).to(device)

        out = self.model(input_tensor)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()

        self.model.zero_grad()
        out[:, class_idx].sum().backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed")

        grads = self.gradients[0].cpu().numpy()
        acts  = self.activations[0].cpu().numpy()
        weights = np.mean(grads, axis=(1, 2))

        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam /= (cam.max() + 1e-8)

        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        return cam

gradcam = GradCAM(model, model.layer4[-1])

# --------------------------------------------------------------- #
# 2.5  Helper functions
# --------------------------------------------------------------- #
def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    t = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    t = std * t + mean
    return np.clip(t, 0, 1)

# --------------------------------------------------------------- #
# 3. Route – GET = form, POST = analyse
# --------------------------------------------------------------- #
@bp.route('/', methods=['GET', 'POST'])
@login_required
def page():
    report = None
    if request.method == 'POST':
        if 'images' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['images']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        try:
            # ---- Load image -------------------------------------------------
            raw_image = Image.open(file.stream).convert('RGB')

            # ---- Pre-process before classify --------------------------------
            processed_image = raw_image.resize((224, 224), Image.Resampling.LANCZOS)
            input_tensor = transform(processed_image).unsqueeze(0).to(device)

            # ---- Forward pass -----------------------------------------------
            with torch.no_grad():
                out = model(input_tensor)
                probs = F.softmax(out, dim=1)[0]
                pred_idx = int(probs.argmax().item())
                confidence = float(probs[pred_idx].item())
                pred_class = classes[pred_idx]

            # ---- Grad-CAM ---------------------------------------------------
            cam = gradcam(input_tensor, pred_idx)

            # ---- Overlay ----------------------------------------------------
            processed_np = (denormalize(input_tensor[0]) * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = (0.4 * heatmap.astype(np.float32) +
                       0.6 * processed_np.astype(np.float32))
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            overlay_pil = Image.fromarray(overlay)

            # ---- Bar chart --------------------------------------------------
            probs_cpu = probs.detach().cpu().numpy()
            plt.figure(figsize=(11, 6.5))
            bars = plt.bar(classes, probs_cpu,
                           color='#b3e2ff', edgecolor='navy', linewidth=1.5)
            bars[pred_idx].set_color('#e74c3c')
            plt.title("Model Confidence per Class", fontsize=19, weight='bold', pad=20)
            plt.ylabel("Probability", fontsize=14)
            plt.ylim(0, 1.05)
            plt.xticks(rotation=30, ha='right', fontsize=11)
            plt.yticks(fontsize=11)
            for i, p in enumerate(probs_cpu):
                plt.text(i, p + 0.02, f"{p:.1%}", ha='center', va='bottom',
                         fontsize=11, weight='bold')
            plt.grid(axis='y', linestyle='--', alpha=0.35)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=140, bbox_inches='tight')
            plt.close()
            bar_b64 = base64.b64encode(buf.getvalue()).decode()

            # ---- Build report -----------------------------------------------
            report = {
                'original_b64' : pil_to_b64(raw_image),
                'processed_b64': pil_to_b64(Image.fromarray(processed_np)),
                'overlay_b64'   : pil_to_b64(overlay_pil),
                'bar_b64'       : bar_b64,
                'pred_class'    : pred_class,
                'confidence'    : confidence,
                'status'        : cancer_status.get(pred_class, "Unknown"),
                'probabilities' : [(c, float(p)) for c, p in zip(classes, probs_cpu)],
            }

        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(request.url)

    return render_template('home2.html', report=report)