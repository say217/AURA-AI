import io
import os
import uuid
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import tensorflow as tf
from flask import Blueprint, flash, redirect, render_template, request, send_file, send_from_directory, session, url_for
from functools import wraps
from keras.preprocessing import image #ignore

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt


bp = Blueprint(
    'app5',
    __name__,
    template_folder='templates',
    static_folder='static'
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_DIR = Path(os.getenv("HF_CACHE_DIR", PROJECT_ROOT / "huggingface_cache"))
HF_REPO_ID = os.getenv("HF_APP5_REPO", "say89/BrainTumour90MNH6602")
HF_REPO_TYPE = os.getenv("HF_APP5_REPO_TYPE", "space")
HF_MODEL_FILE = os.getenv("HF_APP5_FILE", "brain_tumor_resnet101_finetuned_v00.3.keras")
MODEL_PATH = HF_CACHE_DIR / HF_MODEL_FILE
INSTANCE_DIR = os.path.join(os.path.dirname(__file__), "instance")
MAX_IMAGE_SIZE = 10 * 1024 * 1024
IMG_SIZE = (224, 224)
CLASS_NAMES = ["brain_glioma", "brain_menin", "brain_tumor"]
LAST_CONV_LAYER = "conv5_block3_out"

def _ensure_model_available() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH

    if hf_hub_download is None:
        raise RuntimeError(
            "huggingface_hub is not installed. Add it to requirements.txt to download the model."
        )

    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_API_TOKEN")

    try:
        # Prefer placing the file directly under the configured cache folder
        # so deployments that start with an empty cache will self-heal.
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            filename=HF_MODEL_FILE,
            local_dir=str(HF_CACHE_DIR),
            local_dir_use_symlinks=False,
            token=token
        )
        return Path(downloaded)
    except TypeError:
        # Fallback for older huggingface_hub versions without local_dir.
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            filename=HF_MODEL_FILE,
            cache_dir=str(HF_CACHE_DIR),
            token=token
        )

        downloaded_path = Path(downloaded)
        if downloaded_path != MODEL_PATH and not MODEL_PATH.exists():
            MODEL_PATH.write_bytes(downloaded_path.read_bytes())

        return MODEL_PATH


model = tf.keras.models.load_model(str(_ensure_model_available()))
model.trainable = False

assert LAST_CONV_LAYER in [l.name for l in model.layers]


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('app2.login'))
        return f(*args, **kwargs)
    return decorated_function


def _ensure_instance_dir():
    os.makedirs(INSTANCE_DIR, exist_ok=True)


def _clear_instance_dir():
    _ensure_instance_dir()
    for name in os.listdir(INSTANCE_DIR):
        path = os.path.join(INSTANCE_DIR, name)
        if os.path.isfile(path):
            os.remove(path)


def _save_fig(fig, filename):
    _ensure_instance_dir()
    path = os.path.join(INSTANCE_DIR, filename)
    fig.savefig(path, format="png", bbox_inches="tight")
    plt.close(fig)
    return path


def _predict_image(file_obj):
    file_obj.stream.seek(0)
    img_bytes = io.BytesIO(file_obj.read())
    img = image.load_img(img_bytes, target_size=IMG_SIZE)
    img_bytes.close()
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = tf.keras.applications.resnet.preprocess_input(img_arr)

    preds = model(img_arr, training=False)[0].numpy()
    idx = int(np.argmax(preds))
    return img_arr, preds, idx, img


def _gradcam(img_arr, class_idx):
    grad_model = tf.keras.models.Model(
        model.input,
        [model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_arr, training=False)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))

    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam = tf.maximum(cam, 0)
    cam /= tf.reduce_max(cam) + tf.keras.backend.epsilon()

    return cam.numpy()


def _plot_prediction_probs(preds, class_names):
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor="#0c0a08")
    ax.set_facecolor("#0c0a08")
    bars = ax.bar(class_names, preds, color="#b28a5b")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability", color="#d7d2cc")
    ax.set_title("Classification Scores", color="#f4f2ef")
    ax.tick_params(colors="#d7d2cc")

    for spine in ax.spines.values():
        spine.set_color("#3b332a")

    ax.grid(axis="y", color="#3b332a", alpha=0.45, linestyle="--")

    for bar, p in zip(bars, preds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{p:.2f}",
            ha="center",
            va="bottom",
            color="#f4f2ef"
        )

    fig.tight_layout()
    return fig


def _render_gradcam_overlay(img, cam, alpha=0.4):
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    cam_resized = tf.image.resize(
        cam[..., None],
        img.shape[:2]
    ).numpy().squeeze()

    cam_resized = cam_resized / (cam_resized.max() + 1e-8)

    heatmap = plt.cm.jet(cam_resized)[:, :, :3]
    heatmap = np.uint8(255 * heatmap)

    overlay = np.clip(
        img * (1 - alpha) + heatmap * alpha,
        0, 255
    ).astype("uint8")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlay)
    ax.axis("off")
    ax.set_title("Grad-CAM Heatmap (Model Attention)")
    fig.tight_layout()
    return fig


@bp.route('/instance/<path:filename>')
def instance_file(filename):
    _ensure_instance_dir()
    return send_from_directory(INSTANCE_DIR, filename)


@bp.route('/download-images')
@login_required
def download_images():
    images_dir = PROJECT_ROOT / 'images'
    archive_name = 'all-images.zip'
    archive_buffer = io.BytesIO()

    with zipfile.ZipFile(archive_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(images_dir):
            for name in files:
                file_path = Path(root) / name
                if file_path.is_file():
                    arcname = file_path.relative_to(images_dir)
                    zipf.write(file_path, arcname.as_posix())

    archive_buffer.seek(0)
    return send_file(
        archive_buffer,
        as_attachment=True,
        download_name=archive_name,
        mimetype='application/zip'
    )

@bp.route('/', methods=['GET', 'POST'])
@login_required
def page():
    if request.method == 'POST':

        # Check file upload
        if 'images' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('images')
        if not files or all(file.filename == '' for file in files):
            flash('No selected files')
            return redirect(request.url)

        results = []
        _clear_instance_dir()

        for file in files:
            try:
                file.stream.seek(0, os.SEEK_END)
                file_size = file.stream.tell()
                file.stream.seek(0)

                if file_size > MAX_IMAGE_SIZE:
                    flash(
                        f"File too large: {file.filename} exceeds 10 MB limit."
                    )
                    continue

                img_arr, preds, idx, raw_img = _predict_image(file)
                cam = _gradcam(img_arr, idx)

                prob_fig = _plot_prediction_probs(preds, CLASS_NAMES)
                gradcam_fig = _render_gradcam_overlay(raw_img, cam)

                top_label = CLASS_NAMES[idx]
                top_score = float(preds[idx])

                original_name = f"original_{uuid.uuid4().hex}.png"
                prob_name = f"prob_{uuid.uuid4().hex}.png"
                gradcam_name = f"gradcam_{uuid.uuid4().hex}.png"

                _ensure_instance_dir()
                raw_img.save(os.path.join(INSTANCE_DIR, original_name), format="PNG")
                _save_fig(prob_fig, prob_name)
                _save_fig(gradcam_fig, gradcam_name)

                results.append({
                    'filename': file.filename,
                    'top_label': top_label,
                    'top_score': top_score,
                    'top_score_percent': f"{top_score:.2%}",
                    'explanation': (
                        f"Prediction: <strong>{top_label}</strong> "
                        f"({top_score:.2%})"
                    ),
                    'additional_explanation': (
                        "This model provides probability scores for all three classes "
                        "along with a Grad-CAM heatmap to highlight influential regions."
                    ),
                    'original_image': url_for('app5.instance_file', filename=original_name),
                    'prob_plot': url_for('app5.instance_file', filename=prob_name),
                    'results_plot': url_for('app5.instance_file', filename=gradcam_name)
                })

            except Exception as e:
                flash(f'Error processing image {file.filename}: {str(e)}')

        return render_template('home4.html', results=results)

    return render_template('home4.html')
