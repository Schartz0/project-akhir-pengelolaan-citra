from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import onnxruntime as ort
from datetime import datetime

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB max

# ─── KONSTANTA ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_coral_model.onnx')
CLASS_NAMES = ['Healthy', 'Unhealthy', 'Dead']
CLASS_LABELS_ID = {
    'Healthy':   'Sehat',
    'Unhealthy': 'Tidak Sehat',
    'Dead':      'Mati',
}
CLASS_COLORS = {
    'Healthy':   '#27ae60',
    'Unhealthy': '#f39c12',
    'Dead':      '#e74c3c',
}
DISPLAY_MAX_SIZE = 800

# ImageNet normalization params
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ─── LOAD MODEL (sekali saja saat startup) ─────────────────────────────────────
def load_model(model_path):
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    print(f"[INFO] ONNX model loaded: {model_path}")
    return session

ort_session = load_model(MODEL_PATH)

# ─── PREPROCESSING ─────────────────────────────────────────────────────────────
def white_balance(img):
    """White balance menggunakan LAB color space."""
    result = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    avg_a = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 2])
    result[:, :, 1] -= (avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1
    result[:, :, 2] -= (avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

def clahe_enhance(img, clip_limit=2.0, tile_size=8):
    """CLAHE contrast enhancement."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

def dehaze_dark_channel(img, omega=0.95, t0=0.1):
    """Dark channel prior dehazing untuk koreksi underwater."""
    img_norm = img.astype(np.float32) / 255.0
    dark = np.min(img_norm, axis=2)
    dark = cv2.GaussianBlur(dark, (15, 15), 0)
    atmospheric = float(np.percentile(dark, 95))
    if atmospheric < 1e-6:
        atmospheric = 1e-6
    transmission = np.clip(1 - omega * dark / atmospheric, t0, 1)
    result = np.zeros_like(img_norm)
    for i in range(3):
        result[:, :, i] = (img_norm[:, :, i] - atmospheric) / transmission + atmospheric
    return np.clip(result * 255, 0, 255).astype(np.uint8)

def full_preprocess(img_rgb):
    """Pipeline preprocessing: white balance → dehaze → CLAHE."""
    img = white_balance(img_rgb)
    img = dehaze_dark_channel(img)
    img = clahe_enhance(img)
    return img

def preprocess_for_model(img_rgb):
    """Resize, normalize, dan ubah ke NCHW float32 untuk ONNX input."""
    img = cv2.resize(img_rgb, (224, 224)).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    # HWC → CHW → NCHW
    img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
    return img.astype(np.float32)

# ─── HELPER ────────────────────────────────────────────────────────────────────
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def resize_for_display(img):
    h, w = img.shape[:2]
    if max(h, w) <= DISPLAY_MAX_SIZE:
        return img
    scale = DISPLAY_MAX_SIZE / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def img_to_base64(img_rgb, quality=85):
    display = resize_for_display(img_rgb)
    img_bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return "data:image/png;base64," + data

# ─── PLOT FUNGSI ───────────────────────────────────────────────────────────────
def create_probability_bar(probabilities, predicted_class):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    probs  = [probabilities[c] * 100 for c in CLASS_NAMES]
    colors = [CLASS_COLORS[c] if c == predicted_class else '#d0d0d0' for c in CLASS_NAMES]
    labels = [CLASS_LABELS_ID[c] for c in CLASS_NAMES]

    bars = ax.bar(labels, probs, color=colors, edgecolor='white', linewidth=1.5, zorder=3)
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{prob:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylim([0, 115])
    ax.set_ylabel('Probabilitas (%)', fontsize=9)
    ax.set_title('Distribusi Probabilitas Kelas', fontsize=11, fontweight='bold', pad=12)
    ax.tick_params(labelsize=10)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return plot_to_base64(fig)

def create_preprocessing_comparison(raw, processed):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle('Perbandingan Histogram Sebelum & Sesudah Preprocessing', fontsize=11, fontweight='bold')

    for col, (img, title) in enumerate([(raw, 'Asli'), (processed, 'Setelah Preprocessing')]):
        axes[0, col].imshow(img)
        axes[0, col].set_title(title, fontsize=10, fontweight='bold')
        axes[0, col].axis('off')
        for ch, (color, label) in enumerate(zip(['red', 'green', 'blue'], ['R', 'G', 'B'])):
            hist = cv2.calcHist([img], [ch], None, [256], [0, 256])
            axes[1, col].plot(hist, color=color, alpha=0.7, linewidth=1, label=label)
        axes[1, col].set_xlim([0, 255])
        axes[1, col].set_xlabel('Intensitas', fontsize=8)
        axes[1, col].set_ylabel('Frekuensi', fontsize=8)
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].tick_params(labelsize=7)

    fig.tight_layout()
    return plot_to_base64(fig)

def create_confidence_gauge(confidence, predicted_class):
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(aspect="equal"))
    val   = confidence * 100
    color = CLASS_COLORS[predicted_class]
    ax.pie([val, 100 - val], colors=[color, '#eeeeee'], startangle=90,
           wedgeprops=dict(width=0.4, edgecolor='white'))
    ax.text(0, 0, f'{val:.1f}%', ha='center', va='center',
            fontsize=20, fontweight='bold', color=color)
    ax.set_title(f'Confidence\n{CLASS_LABELS_ID[predicted_class]}', fontsize=10, fontweight='bold', pad=5)
    fig.tight_layout()
    return plot_to_base64(fig)

# ─── INFERENCE ─────────────────────────────────────────────────────────────────
def predict(img_rgb):
    processed   = full_preprocess(img_rgb)
    input_tensor = preprocess_for_model(processed)

    input_name  = ort_session.get_inputs()[0].name
    outputs     = ort_session.run(None, {input_name: input_tensor})
    logits      = outputs[0][0]           # shape (3,)
    probs       = softmax(logits)

    predicted_idx   = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence      = float(probs[predicted_idx])
    probabilities   = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}

    return {
        'predicted_class':    predicted_class,
        'predicted_label_id': CLASS_LABELS_ID[predicted_class],
        'confidence':         confidence,
        'probabilities':      probabilities,
        'timestamp':          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'raw_img':            img_rgb,
        'processed_img':      processed,
    }

# ─── ROUTES ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({'error': 'Gambar harus diupload'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({'error': 'Gagal membaca gambar. Pastikan format file valid (JPG/PNG).'}), 400

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result  = predict(img_rgb)

        return jsonify({
            'predicted_class':    result['predicted_class'],
            'predicted_label_id': result['predicted_label_id'],
            'confidence':         result['confidence'],
            'probabilities':      result['probabilities'],
            'timestamp':          result['timestamp'],
            'img_raw':            img_to_base64(result['raw_img']),
            'img_processed':      img_to_base64(result['processed_img']),
            'chart_probs':        create_probability_bar(result['probabilities'], result['predicted_class']),
            'chart_preprocess':   create_preprocessing_comparison(result['raw_img'], result['processed_img']),
            'chart_gauge':        create_confidence_gauge(result['confidence'], result['predicted_class']),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
