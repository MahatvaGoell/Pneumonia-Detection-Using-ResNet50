from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model


MODEL_PATH = Path("models/pneumonia_model.h5")
IMAGE_SIZE = (224, 224)
_model = None


def get_model():
    """Load the model only when a prediction is requested."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model file not found. Train the model first, then place "
                "pneumonia_model.h5 inside the models folder."
            )
        _model = load_model(MODEL_PATH)
    return _model


def result_card(title, confidence, kind, note):
    icon = "✓" if kind == "normal" else "!"
    return f"""
    <div class="result-card {kind}">
        <div class="result-icon">{icon}</div>
        <div>
            <p class="eyebrow">SCREENING RESULT</p>
            <h2>{title}</h2>
            <p class="result-note">{note}</p>
        </div>
        <div class="confidence">
            <span>CONFIDENCE</span>
            <strong>{confidence:.1f}%</strong>
        </div>
    </div>
    """


def empty_result():
    return """
    <div class="result-card empty-result">
        <div class="result-icon">+</div>
        <div>
            <p class="eyebrow">READY WHEN YOU ARE</p>
            <h2>Your analysis will appear here</h2>
            <p class="result-note">Upload a chest X-ray, then select Analyze X-ray.</p>
        </div>
    </div>
    """


def predict_xray(image):
    if image is None:
        return empty_result()

    try:
        image = np.asarray(image)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        image = cv2.resize(image, IMAGE_SIZE).astype("float32") / 255.0
        prediction = float(get_model().predict(np.expand_dims(image, axis=0), verbose=0)[0][0])

        if prediction > 0.5:
            return result_card(
                "Pneumonia detected",
                prediction * 100,
                "alert",
                "This image shows features associated with pneumonia.",
            )

        return result_card(
            "No pneumonia detected",
            (1 - prediction) * 100,
            "normal",
            "This image is classified as normal by the model.",
        )
    except FileNotFoundError as error:
        return f"""
        <div class="result-card error-result">
            <div class="result-icon">!</div>
            <div><p class="eyebrow">MODEL UNAVAILABLE</p><h2>Analysis cannot start yet</h2>
            <p class="result-note">{error}</p></div>
        </div>"""
    except Exception:
        return """
        <div class="result-card error-result">
            <div class="result-icon">!</div>
            <div><p class="eyebrow">ANALYSIS ERROR</p><h2>We could not read this image</h2>
            <p class="result-note">Please choose a clear JPG or PNG chest X-ray and try again.</p></div>
        </div>"""


CSS = """
:root { --ink: #102a43; --muted: #627d98; --line: #d9e2ec; --blue: #1479c9; --navy: #0b1f33; --mint: #e7f8f2; }
body, .gradio-container { background: #f5f8fb !important; color: var(--ink) !important; font-family: Inter, ui-sans-serif, system-ui, sans-serif !important; }
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; padding: 28px 20px 38px !important; }
footer { display: none !important; }
.hero { padding: 26px 4px 30px; }
.badge, .eyebrow { color: var(--blue); font-size: 0.72rem; font-weight: 800; letter-spacing: 0.13em; margin: 0 0 8px; }
.hero h1 { color: var(--navy); font-size: clamp(2rem, 5vw, 3.55rem); line-height: 1.04; letter-spacing: -0.05em; margin: 0; }
.hero p { color: var(--muted); font-size: 1.06rem; line-height: 1.6; margin: 14px 0 0; max-width: 660px; }
.workspace { align-items: stretch !important; gap: 22px !important; }
.panel { background: #fff; border: 1px solid var(--line); border-radius: 20px; box-shadow: 0 12px 32px rgba(24, 55, 85, 0.08); min-height: 520px; padding: 22px !important; }
.panel h3 { color: var(--navy); font-size: 1.08rem; margin: 0 0 4px; }
.panel-caption { color: var(--muted); font-size: 0.9rem; margin: 0 0 17px; }
#xray-input { border: 2px dashed #b7c9d9 !important; border-radius: 14px !important; height: 380px !important; overflow: hidden; }
#xray-input .image-container, #xray-input .wrap { min-height: 374px !important; }
#xray-input img { object-fit: contain !important; }
#analyze-button { background: var(--blue) !important; border: 0 !important; border-radius: 11px !important; font-weight: 750 !important; height: 48px !important; margin-top: 18px !important; transition: transform .2s ease, background .2s ease !important; }
#analyze-button:hover { background: #0963ab !important; transform: translateY(-1px); }
.result-card { align-items: center; border: 1px solid var(--line); border-radius: 16px; display: flex; gap: 15px; margin-top: 18px; min-height: 136px; padding: 19px; }
.result-card h2 { color: var(--navy); font-size: 1.28rem; line-height: 1.2; margin: 0 0 7px; }
.result-note { color: var(--muted); font-size: .9rem; line-height: 1.45; margin: 0; }
.result-icon { align-items: center; background: #e8f1f8; border-radius: 50%; color: var(--blue); display: flex; flex: 0 0 42px; font-size: 1.25rem; font-weight: 800; height: 42px; justify-content: center; }
.confidence { border-left: 1px solid var(--line); margin-left: auto; min-width: 100px; padding-left: 15px; text-align: right; }
.confidence span { color: var(--muted); display: block; font-size: .65rem; font-weight: 800; letter-spacing: .09em; }
.confidence strong { color: var(--navy); display: block; font-size: 1.45rem; margin-top: 4px; }
.normal { background: var(--mint); border-color: #c7ebdf; }.normal .result-icon { background: #c9f0e2; color: #168260; }
.alert { background: #fff4f2; border-color: #ffd6ce; }.alert .result-icon { background: #ffe0da; color: #c34a3b; }
.error-result { background: #fff8e9; border-color: #f2dfac; }.error-result .result-icon { background: #fbe8b7; color: #8d6500; }
.empty-result { background: #f8fbfd; }
.disclaimer { color: var(--muted); font-size: .8rem; line-height: 1.5; margin: 21px 5px 0; text-align: center; }
@media (max-width: 700px) { .gradio-container { padding: 16px 12px 24px !important; }.hero { padding: 16px 4px 22px; }.panel { min-height: auto; }.confidence { min-width: 80px; }.result-card { align-items: flex-start; }.result-card h2 { font-size: 1.1rem; } }
"""


with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=CSS, title="LungLens | Pneumonia Screening") as interface:
    gr.HTML("""
        <section class="hero">
            <p class="badge">AI-ASSISTED CHEST X-RAY SCREENING</p>
            <h1>Clear insight from every X-ray.</h1>
            <p>LungLens uses a ResNet50 model to help identify signs of pneumonia from chest X-ray images in seconds.</p>
        </section>
    """)

    with gr.Row(elem_classes="workspace"):
        with gr.Column(elem_classes="panel", scale=1):
            gr.HTML("<h3>01 · Add an X-ray</h3><p class='panel-caption'>Upload a JPG or PNG image to begin.</p>")
            image_input = gr.Image(type="numpy", label="Chest X-ray", sources=["upload"], elem_id="xray-input", height=380)
            analyze = gr.Button("Analyze X-ray", variant="primary", elem_id="analyze-button")

        with gr.Column(elem_classes="panel", scale=1):
            gr.HTML("<h3>02 · Review result</h3><p class='panel-caption'>The model’s classification and confidence appear below.</p>")
            result = gr.HTML(empty_result())
            gr.HTML("""
                <div style="margin-top:28px; padding-top:20px; border-top:1px solid #d9e2ec;">
                    <p class="eyebrow">HOW IT WORKS</p>
                    <p class="result-note">Your image is resized to 224 × 224 pixels and evaluated by a ResNet50 classifier trained on chest X-ray images.</p>
                </div>
            """)

    gr.HTML("<p class='disclaimer'>For educational and research use only. This tool is not a medical diagnosis and must not replace evaluation by a qualified healthcare professional.</p>")
    analyze.click(predict_xray, inputs=image_input, outputs=result)
    image_input.clear(empty_result, outputs=result)


if __name__ == "__main__":
    interface.launch()
