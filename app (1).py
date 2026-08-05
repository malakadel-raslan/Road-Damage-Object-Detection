"""
Road Damage Detection — Gradio Web App
----------------------------------------
Upload a road image and get back detected damage (potholes, cracks, etc.)
with bounding boxes, using a YOLOv8 model trained in the companion notebook.

Run locally:
    pip install -r requirements.txt
    python app.py

Then open the local URL Gradio prints (usually http://127.0.0.1:7860).
"""

import os
import logging

import gradio as gr
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("road-damage-app")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Put your trained weights file in this same folder (or set MODEL_PATH env var)
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
CONFIDENCE_DEFAULT = 0.25
SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", 7860))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model weights not found at '{MODEL_PATH}'. "
        "Place your best.pt file next to app.py, or set the MODEL_PATH "
        "environment variable to point to it."
    )

log.info("Loading model from %s", MODEL_PATH)
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names  # e.g. {0: "pothole", 1: "crack", ...}

# Distinct accent per class, cycling through the hazard palette so results
# scan quickly even with several damage types on screen at once.
CLASS_COLORS = ["#FF6B35", "#FFC145", "#3FB27F", "#5EA8ED", "#C792EA"]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def detect_damage(image, confidence: float):
    """Run YOLO detection on an uploaded image and return the annotated image
    plus an HTML summary of what was found."""
    if image is None:
        return None, _empty_state("Upload an image to begin inspection.")

    try:
        results = model.predict(source=image, conf=confidence, verbose=False)[0]
    except Exception as exc:  # keep the UI alive on a bad/corrupt upload
        log.exception("Inference failed")
        return None, _empty_state(f"Couldn't process that image ({exc}).")

    annotated = results.plot()  # numpy array, RGB, boxes already drawn

    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return annotated, _empty_state("No damage detected at this confidence threshold.")

    counts = {}
    for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
        name = CLASS_NAMES.get(int(cls_id), str(int(cls_id)))
        counts.setdefault(name, []).append(conf)

    return annotated, _results_html(counts, total=len(boxes))


def _empty_state(message: str) -> str:
    return f"""
    <div class="rd-empty">
        <span class="rd-empty-icon">&#9888;</span>
        <span>{message}</span>
    </div>
    """


def _results_html(counts: dict, total: int) -> str:
    rows = ""
    for i, (name, confs) in enumerate(sorted(counts.items(), key=lambda x: -len(x[1]))):
        color = CLASS_COLORS[i % len(CLASS_COLORS)]
        avg_conf = sum(confs) / len(confs)
        rows += f"""
        <div class="rd-stat-row">
            <span class="rd-swatch" style="background:{color}"></span>
            <span class="rd-stat-name">{name}</span>
            <span class="rd-stat-count">&times;{len(confs)}</span>
            <span class="rd-stat-conf">{avg_conf:.0%} avg conf.</span>
        </div>
        """
    return f"""
    <div class="rd-results">
        <div class="rd-results-head">
            <span class="rd-results-total">{total}</span>
            <span class="rd-results-label">detection{"s" if total != 1 else ""} found</span>
        </div>
        {rows}
    </div>
    """


# ---------------------------------------------------------------------------
# Visual identity — asphalt / hazard-signage theme
# ---------------------------------------------------------------------------
FONTS_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#16181C",
    background_fill_primary="#1F2226",
    background_fill_secondary="#1A1C20",
    border_color_primary="#2E3238",
    body_text_color="#E5E7EB",
    body_text_color_subdued="#9CA3AF",
    button_primary_background_fill="#FF6B35",
    button_primary_background_fill_hover="#FF8555",
    button_primary_text_color="#16181C",
    block_background_fill="#1F2226",
    block_border_color="#2E3238",
    block_label_text_color="#9CA3AF",
    slider_color="#FF6B35",
)

CUSTOM_CSS = """
@keyframes rd-stripe-scroll { from { background-position: 0 0; } to { background-position: 56px 0; } }
@keyframes rd-fade-up { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
@keyframes rd-scan { 0% { top: -10%; } 100% { top: 110%; } }

.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
#rd-header, .rd-card { animation: rd-fade-up 0.5s ease both; }

/* ---- Header ---- */
#rd-header {
    padding: 32px 4px 0 4px;
}
#rd-header h1 {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: #F4F4F5;
    margin: 0;
    line-height: 1;
}
#rd-header h1 span { color: #FF6B35; }
#rd-header p {
    font-family: 'Inter', sans-serif;
    color: #9CA3AF;
    margin: 8px 0 16px 0;
    font-size: 0.95rem;
    max-width: 46ch;
}

/* Header meta chips */
.rd-chips { display: flex; gap: 8px; flex-wrap: wrap; }
.rd-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 10px;
    border: 1px solid #2E3238;
    border-radius: 999px;
    background: #1A1C20;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.04em;
    color: #9CA3AF;
}
.rd-chip-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #3FB27F;
    box-shadow: 0 0 0 3px rgba(63,178,127,0.18);
}

/* ---- Signature element: animated hazard-stripe divider ---- */
.rd-hazard-divider {
    height: 8px;
    margin: 22px 0 30px 0;
    border-radius: 2px;
    background-image: repeating-linear-gradient(
        135deg,
        #FF6B35, #FF6B35 14px,
        #16181C 14px, #16181C 28px
    );
    background-size: 56px 8px;
    opacity: 0.9;
    animation: rd-stripe-scroll 2.4s linear infinite;
}

/* ---- Section labels ---- */
.rd-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #FFC145;
    margin: 0 0 12px 2px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.rd-eyebrow::after {
    content: "";
    flex-grow: 1;
    height: 1px;
    background: #2E3238;
}

/* ---- Card wrapper ---- */
.rd-card {
    background: #1A1C20;
    border: 1px solid #2E3238;
    border-radius: 12px;
    padding: 18px;
    transition: border-color 0.2s ease;
}
.rd-card:hover { border-color: #3A3F47; }

/* ---- Buttons ---- */
button.primary, .rd-card button.primary {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    font-size: 1.02rem !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    box-shadow: 0 0 0 rgba(255,107,53,0);
}
button.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(255,107,53,0.28) !important;
}

/* ---- Results panel ---- */
.rd-empty {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 18px 16px;
    border: 1px dashed #2E3238;
    border-radius: 8px;
    color: #9CA3AF;
    font-size: 0.92rem;
    animation: rd-fade-up 0.3s ease both;
}
.rd-empty-icon { color: #FFC145; font-size: 1.1rem; }

.rd-results { animation: rd-fade-up 0.3s ease both; }
.rd-results-head {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid #2E3238;
}
.rd-results-total {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    color: #FF6B35;
    line-height: 1;
}
.rd-results-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9CA3AF;
}

.rd-stat-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 6px;
    border-bottom: 1px solid #24272C;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    border-radius: 6px;
    transition: background 0.15s ease;
}
.rd-stat-row:hover { background: #1F2226; }
.rd-stat-row:last-child { border-bottom: none; }
.rd-swatch {
    width: 10px; height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
    box-shadow: 0 0 0 3px rgba(255,255,255,0.04);
}
.rd-stat-name {
    color: #F4F4F5;
    font-weight: 500;
    text-transform: capitalize;
    flex-grow: 1;
}
.rd-stat-count {
    font-family: 'JetBrains Mono', monospace;
    color: #FFC145;
    font-size: 0.85rem;
}
.rd-stat-conf {
    font-family: 'JetBrains Mono', monospace;
    color: #9CA3AF;
    font-size: 0.78rem;
    min-width: 92px;
    text-align: right;
}

/* Scan-line accent over the result image, evokes an active inspection pass */
.rd-scan-wrap { position: relative; overflow: hidden; border-radius: 8px; }
.rd-scan-wrap::before {
    content: "";
    position: absolute;
    left: 0; right: 0; top: -10%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #FF6B35, transparent);
    opacity: 0.9;
    animation: rd-scan 2.6s ease-in-out infinite;
    pointer-events: none;
}

/* ---- Footer ---- */
#rd-footer {
    margin-top: 10px;
    padding: 16px 4px 24px 4px;
    color: #6B7280;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
}
#rd-footer .rd-dot { opacity: 0.5; }
"""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
with gr.Blocks(theme=THEME, css=CUSTOM_CSS, head=FONTS_HEAD, title="Road Damage Detection") as demo:
    with gr.Column(elem_id="rd-header"):
        gr.HTML(
            "<h1>Road <span>Damage</span> Detection</h1>"
            "<p>Upload a road surface image for automated pothole and crack inspection, "
            "powered by a fine-tuned YOLOv8 model.</p>"
            f"""<div class="rd-chips">
                <span class="rd-chip"><span class="rd-chip-dot"></span>Model ready</span>
                <span class="rd-chip">{len(CLASS_NAMES)} damage classes</span>
                <span class="rd-chip">Local inference only</span>
            </div>"""
        )
    gr.HTML('<div class="rd-hazard-divider"></div>')

    with gr.Row():
        with gr.Column(scale=5, elem_classes="rd-card"):
            gr.HTML('<div class="rd-eyebrow">01 — Input</div>')
            input_image = gr.Image(type="pil", label="Road image", height=320)
            confidence_slider = gr.Slider(
                minimum=0.05, maximum=0.95, value=CONFIDENCE_DEFAULT, step=0.05,
                label="Confidence threshold",
            )
            submit_btn = gr.Button("Run inspection", variant="primary")

        with gr.Column(scale=5, elem_classes="rd-card"):
            gr.HTML('<div class="rd-eyebrow">02 — Findings</div>')
            with gr.Column(elem_classes="rd-scan-wrap"):
                output_image = gr.Image(type="numpy", label="Annotated result", height=320)
            output_summary = gr.HTML(_empty_state("Upload an image to begin inspection."))

    gr.HTML(
        '<div id="rd-footer">'
        '<span>YOLOv8</span><span class="rd-dot">&middot;</span>'
        '<span>Local inference</span><span class="rd-dot">&middot;</span>'
        '<span>No data leaves this machine</span>'
        '</div>'
    )

    submit_btn.click(
        fn=detect_damage,
        inputs=[input_image, confidence_slider],
        outputs=[output_image, output_summary],
    )
    input_image.change(
        fn=detect_damage,
        inputs=[input_image, confidence_slider],
        outputs=[output_image, output_summary],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=SERVER_PORT)
