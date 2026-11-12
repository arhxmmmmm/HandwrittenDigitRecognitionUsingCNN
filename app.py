"""Streamlit user interface - Handwritten Digit Recognition System."""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from preprocess import to_mnist

MODEL_PATH = os.path.join(os.path.dirname(__file__), "outputs", "digit_cnn.keras")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "mnist.npz")

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 4rem; max-width: 1100px;}
    .title {text-align:center; font-size:2rem; font-weight:700; color:#12285c; margin-bottom:0;}
    .sub {text-align:center; color:#6b7280; font-size:.85rem; margin-bottom:1.2rem;}
    .card {border-radius:10px; padding:1.4rem 1rem; text-align:center; margin-bottom:.8rem;}
    .ok {background:#e7f8ec; border:1px solid #86d9a0;}
    .warn {background:#fff6e0; border:1px solid #f0c36b;}
    .digit {font-size:4.2rem; font-weight:800; line-height:1; color:#12285c;}
    .conf {font-size:1.25rem; font-weight:700; color:#1f7a3d;}
    .lbl {font-size:.8rem; color:#6b7280;}
    </style>
    <div class="title">✍️ Handwritten Digit Recognition System</div>
    <div class="sub">AI-powered digit classification using a Convolutional Neural Network (CNN)</div>
    <hr style="margin-top:0">
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model():
    from tensorflow import keras

    return keras.models.load_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_test_set():
    d = np.load(DATA_PATH)
    return d["x_test"], d["y_test"]


try:
    model = load_model()
    st.success("Digit recognition model loaded successfully.")
except Exception as e:  # pragma: no cover
    st.error(f"Could not load the model at {MODEL_PATH}. Run train.py first. ({e})")
    st.stop()


def predict(arr28: np.ndarray):
    p = model.predict(arr28[None, ..., None].astype("float32"), verbose=0)[0]
    return int(p.argmax()), p


def show_result(p_digit: int, probs: np.ndarray, processed: np.ndarray):
    conf = float(probs[p_digit])
    css = "ok" if conf >= 0.80 else "warn"
    st.markdown(
        f"""<div class="card {css}"><div class="lbl">Predicted Digit</div>
        <div class="digit">{p_digit}</div>
        <div class="conf">{conf*100:.2f}%</div>
        <div class="lbl">Prediction Confidence</div></div>""",
        unsafe_allow_html=True,
    )
    if conf < 0.80:
        st.warning("Low confidence - the image may be unclear or may not contain a single digit.")
    else:
        st.info(f"The model recognises this image as the digit {p_digit}.")
    st.markdown("**Class probabilities**")
    st.bar_chart(pd.DataFrame({"Probability": probs}, index=[str(i) for i in range(10)]), height=220)
    c1, c2 = st.columns([1, 3])
    c1.image(np.clip(processed, 0, 255).astype(np.uint8), caption="Model input (28×28)", width=110)


left, right = st.columns(2, gap="large")

with left:
    st.subheader("📤 Input Digit")
    mode = st.radio(
        "Input method",
        ["Upload image", "Draw digit", "Test-set sample"],
        horizontal=True,
        label_visibility="collapsed",
    )
    processed = None
    true_label = None

    if mode == "Upload image":
        f = st.file_uploader("Choose a digit image", type=["png", "jpg", "jpeg", "bmp"])
        inv = st.selectbox(
            "Background", ["Auto-detect", "Dark digit on light paper", "Light digit on dark background"]
        )
        if f is not None:
            img = Image.open(f)
            st.image(img, caption="Uploaded Image", width=280)
            processed = to_mnist(img, {"Auto-detect": "auto", "Dark digit on light paper": "yes"}.get(inv, "no"))

    elif mode == "Draw digit":
        try:
            from streamlit_drawable_canvas import st_canvas

            st.caption("Draw a single digit (0-9) in the black box with the mouse or touch.")
            res = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=18,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                update_streamlit=True,
                return_image_data=True,
                key="canvas",
            )
            if res.image_data is not None and res.image_data[..., :3].max() > 0:
                processed = to_mnist(Image.fromarray(res.image_data[..., :3].astype("uint8")), "no")
        except Exception as e:
            st.error(f"Drawing canvas is unavailable in this environment ({e}). Use another input method.")

    else:
        xt, yt = load_test_set()
        if "idx" not in st.session_state:
            st.session_state.idx = 0
        a, b = st.columns([1, 1])
        idx = a.number_input("Test image index (0-9999)", 0, len(xt) - 1, st.session_state.idx, step=1)
        if b.button("🎲 Random sample", use_container_width=True):
            idx = int(np.random.randint(len(xt)))
            st.session_state.idx = idx
        st.image(xt[idx], caption=f"MNIST test image #{idx}", width=220, clamp=True)
        processed = xt[idx].astype("float32")
        true_label = int(yt[idx])

with right:
    st.subheader("📊 Recognition Results")
    if processed is None:
        st.info("Provide a digit image on the left to see the prediction.")
    else:
        d, p = predict(processed)
        show_result(d, p, processed)
        if true_label is not None:
            (st.success if d == true_label else st.error)(f"Ground-truth label: {true_label}")

st.markdown("---")
st.caption("Model: 3-block CNN trained on MNIST · 28×28 grayscale input · Softmax over 10 classes")
