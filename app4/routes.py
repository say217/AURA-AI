# rpoutes.py
from flask import Blueprint, render_template, session, redirect, url_for, request, current_app
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
import base64
import os
import cv2

bp = Blueprint(
    'app4',
    __name__,
    template_folder='templates',
    static_folder='static'
)

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
    def __init__(self, model, target_layer=None, device='cpu'):
        self.model = model
        self.model.eval()
        self.device = device
        if target_layer is None:
            target_layer = self.model.resnet.layer4
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        if isinstance(self.target_layer, nn.Sequential):
            module_to_hook = self.target_layer[-1]
        else:
            module_to_hook = self.target_layer

        module_to_hook.register_forward_hook(forward_hook)
        module_to_hook.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not collect activations/gradients.")

        activations = self.activations[0]
        gradients = self.gradients[0]

        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam_np = cam.cpu().numpy()
        if cam_np.max() != 0:
            cam_np = cam_np - cam_np.min()
            cam_np = cam_np / (cam_np.max() + 1e-9)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

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

gradcam = GradCAM(model=model, target_layer=model.resnet.layer4, device=device)

# ------------------------
# Helpers
# ------------------------
def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return img_base64

def create_cam_overlay(original_pil, cam_map, out_path, alpha=0.5, colormap=cv2.COLORMAP_JET):
    orig = np.array(original_pil)
    orig_bgr = orig[..., ::-1].copy()
    cam_resized = cv2.resize(cam_map, (orig.shape[1], orig.shape[0]))
    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    overlay_bgr = cv2.addWeighted(heatmap, alpha, orig_bgr, 1 - alpha, 0)
    overlay_rgb = overlay_bgr[..., ::-1]
    overlay_pil = Image.fromarray(overlay_rgb)
    overlay_pil.save(out_path)
    return out_path

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

        results = []

        for file in files:
            if file and file.filename != '':
                try:
                    # Save uploaded image
                    upload_folder = os.path.join(current_app.static_folder, 'uploads')
                    os.makedirs(upload_folder, exist_ok=True)
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
                    input_tensor_for_grad = preprocessed_image.clone().detach().to(device)
                    input_tensor_for_grad.requires_grad = True
                    outputs_for_grad = model(input_tensor_for_grad)
                    pred_idx = int(outputs_for_grad.argmax(dim=1).item())
                    cam_map = gradcam.generate_cam(input_tensor_for_grad, target_class=pred_idx)
                    cam_filename = f"cam_{os.path.splitext(image_filename)[0]}.png"
                    cam_path = os.path.join(upload_folder, cam_filename)
                    create_cam_overlay(image, cam_map, cam_path, alpha=0.5)

                    # Explanation
                    heat_strength = cam_map.mean()
                    heat_phrase = "strong localized regions" if heat_strength > 0.35 else "moderate regions" if heat_strength > 0.15 else "diffuse regions"
                    if predicted_class == "IDC(+)":
                        explanation_text = (
                            f"The model predicts **Cancer Positive (IDC+)** with a confidence of {confidence.item()*100:.2f}%. "
                            f"This indicates the presence of Invasive Ductal Carcinoma (IDC), a common type of breast cancer where abnormal cells are detected in the breast tissue. "
                            f"The prediction is based on patterns identified in the image by a deep learning model (ResNet-18), which was trained on histopathology images to distinguish between cancerous and non-cancerous tissues. "
                            f"The Grad-CAM visualization shows {heat_phrase} of focus, highlighting areas in the image that the model considers most indicative of cancer. "
                            f"Brighter regions in the heatmap suggest higher importance in the model's decision. "
                            f"Please note that this is an automated prediction and should not be considered a definitive diagnosis. Consult a medical professional for a comprehensive evaluation."
                        )
                    else:
                        explanation_text = (
                            f"The model predicts **Cancer Negative (IDC-)** with a confidence of {confidence.item()*100:.2f}%. "
                            f"This suggests that the image does not show signs of Invasive Ductal Carcinoma (IDC), indicating the absence of cancerous cells in the analyzed tissue. "
                            f"The prediction is made by a deep learning model (ResNet-18) trained to identify patterns in histopathology images. "
                            f"The Grad-CAM visualization shows {heat_phrase} of focus, indicating the areas the model analyzed to make this prediction. "
                            f"Brighter regions in the heatmap highlight areas of interest, though in this case, they support a non-cancerous prediction. "
                            f"While this result is encouraging, it is not a substitute for a professional medical diagnosis. Please consult a healthcare provider for confirmation."
                        )

                    # Image plots
                    fig_img = plt.figure(figsize=(8, 4))
                    plt.subplot(1, 3, 1)
                    plt.imshow(image)
                    plt.title(f"Original Image\nPrediction: {predicted_class} ({confidence.item():.2f})")
                    plt.axis('off')

                    preprocessed_img_np = preprocessed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    preprocessed_img_np = preprocessed_img_np * 0.5 + 0.5
                    preprocessed_img_np = np.clip(preprocessed_img_np, 0, 1)
                    plt.subplot(1, 3, 2)
                    plt.imshow(preprocessed_img_np)
                    plt.title("Preprocessed Image")
                    plt.axis('off')

                    cam_display = Image.open(cam_path).convert('RGB')
                    plt.subplot(1, 3, 3)
                    plt.imshow(cam_display)
                    plt.title("Grad-CAM Overlay")
                    plt.axis('off')

                    plt.tight_layout()
                    image_plot = plot_to_base64(fig_img)
                    plt.close(fig_img)

                    # Probability plot
                    fig_prob = plt.figure(figsize=(5, 3))
                    classes = ['IDC(-)', 'IDC(+)']
                    probabilities = [prob_idc_neg, prob_idc_pos]
                    sns.barplot(x=probabilities, y=classes)
                    plt.xlim(0, 1)
                    plt.xlabel("Probability")
                    plt.title("Prediction Probability Distribution")
                    prob_plot = plot_to_base64(fig_prob)
                    plt.close(fig_prob)

                    # Append result
                    results.append({
                        "filename": image_filename,
                        "predicted_class": predicted_class,
                        "cancer_status": cancer_status,
                        "confidence": f"{confidence.item():.4f}",
                        "prob_idc_neg": f"{prob_idc_neg:.4f}",
                        "prob_idc_pos": f"{prob_idc_pos:.4f}",
                        "image_plot": image_plot,
                        "prob_plot": prob_plot,
                        "uploaded_image": f"uploads/{image_filename}",
                        "cam_image": f"uploads/{cam_filename}",
                        "explanation_text": explanation_text
                    })

                except Exception as e:
                    return render_template('home3.html', error=f"Error processing image {file.filename}: {str(e)}")

        return render_template('home3.html', results=results)

    return render_template('home3.html')