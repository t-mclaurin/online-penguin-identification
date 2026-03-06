from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

from model import build_embedding_model


# ============================================================
# Config
# ============================================================
APP_TITLE = "Penguin Identity Demo"
APP_SUBTITLE = (
    "Upload a penguin image. The app embeds it, compares it to precomputed "
    "identity centres, and returns the closest known penguin."
)

# Point these at the outputs from prepare_app_assets.py
APP_DIR = Path(__file__).resolve().parent
CENTRES_CSV = APP_DIR / "app_assets" / "identity_centres.csv"
DEFAULT_WEIGHTS = APP_DIR / "app_assets/model.weights.h5"
weights_path = str(DEFAULT_WEIGHTS)

IMAGE_SIZE = 224
EMBEDDING_DIM = 256
DROPOUT_RATE = 0.1
DISTANCE_THRESHOLD = 0.45
TOP_K = 5

#Check 
st.write("APP_DIR:", APP_DIR)
st.write("Weights exists:", DEFAULT_WEIGHTS, Path(DEFAULT_WEIGHTS).exists())
st.write("Centres CSV exists:", CENTRES_CSV, Path(CENTRES_CSV).exists())


# ============================================================
# Utilities
# ============================================================
def path_exists(path_str: str | Path | None) -> bool:
    return path_str is not None and Path(path_str).exists()


def resolve_from_app_dir(path_str: str | Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (APP_DIR / p).resolve()


# ============================================================
# Image preprocessing (matches evaluation script)
# ============================================================
def preprocess_pil_image(image: Image.Image, image_size: int) -> np.ndarray:
    image = image.convert("RGB")
    arr = np.asarray(image)
    tensor = tf.convert_to_tensor(arr)
    tensor = tf.image.resize(tensor, [image_size, image_size], antialias=True)
    tensor = tf.cast(tensor, tf.float32) / 255.0
    tensor = tf.expand_dims(tensor, axis=0)
    return tensor.numpy()


def normalize_embeddings(embeddings: tf.Tensor) -> tf.Tensor:
    return tf.math.l2_normalize(embeddings, axis=1)


# ============================================================
# Cached loaders
# ============================================================
@st.cache_resource
# ML models are a standard st.cache_resource use case.
def load_model(weights_path: str | Path, image_size: int, embedding_dim: int, dropout_rate: float):
    weights_path = resolve_from_app_dir(weights_path)
    model = build_embedding_model(
        input_shape=(image_size, image_size, 3),
        embedding_dim=embedding_dim,
        base_trainable=False,
        dropout_rate=dropout_rate,
    )
    model.load_weights(str(weights_path))
    return model


@st.cache_data
# Tabular / array-like data are a standard st.cache_data use case.
def load_gallery_assets(centres_csv_path: str | Path) -> Tuple[pd.DataFrame, np.ndarray]:
    centres_csv_path = resolve_from_app_dir(centres_csv_path)
    centres_df = pd.read_csv(centres_csv_path)

    required_cols = {"identity", "centre_index", "rep_image_path", "centre_embedding_file"}
    missing = required_cols - set(centres_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in centres CSV: {sorted(missing)}")

    if centres_df.empty:
        raise ValueError("The centres CSV is empty.")

    centre_file_candidates = centres_df["centre_embedding_file"].dropna().unique().tolist()
    if len(centre_file_candidates) == 0:
        raise ValueError("No centre_embedding_file values found in centres CSV.")

    centres_npy_path = resolve_from_app_dir(centre_file_candidates[0])
    centres = np.load(str(centres_npy_path)).astype(np.float32)

    if len(centres_df) != len(centres):
        raise ValueError(
            f"Mismatch between centres CSV rows ({len(centres_df)}) and centres array ({len(centres)})."
        )

    centres_df = centres_df.sort_values("centre_index").reset_index(drop=True)

    expected_indices = np.arange(len(centres_df))
    actual_indices = centres_df["centre_index"].to_numpy()
    if not np.array_equal(expected_indices, actual_indices):
        raise ValueError(
            "centre_index values must run from 0..N-1 and match the row order of the .npy file."
        )

    return centres_df, centres


# ============================================================
# Inference
# ============================================================
def embed_uploaded_image(model: tf.keras.Model, image: Image.Image, image_size: int) -> np.ndarray:
    x = preprocess_pil_image(image, image_size=image_size)
    emb = model(x, training=False)
    emb = normalize_embeddings(emb).numpy()[0].astype(np.float32)
    return emb


def compute_distances(query_embedding: np.ndarray, centre_embeddings: np.ndarray) -> np.ndarray:
    # centres and query are expected to be L2-normalized, matching evaluation logic
    sims = centre_embeddings @ query_embedding
    dist_sq = np.clip(2.0 - 2.0 * sims, a_min=0.0, a_max=None)
    dists = np.sqrt(dist_sq).astype(np.float32)
    return dists


def rank_identities(query_embedding: np.ndarray, centres_df: pd.DataFrame, centre_embeddings: np.ndarray, top_k: int):
    dists = compute_distances(query_embedding, centre_embeddings)
    order = np.argsort(dists)
    top_idx = order[:top_k]

    ranked = centres_df.iloc[top_idx].copy().reset_index(drop=True)
    ranked["distance"] = dists[top_idx]
    return ranked, dists


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.write(APP_SUBTITLE)

with st.sidebar:
    st.header("Configuration")
    weights_path = st.text_input("Weights (.h5)", value=str(DEFAULT_WEIGHTS))
    centres_csv_path = st.text_input("Centres CSV", value=str(CENTRES_CSV))
    image_size = st.number_input("Image size", min_value=32, max_value=2048, value=IMAGE_SIZE, step=32)
    embedding_dim = st.number_input("Embedding dim", min_value=2, max_value=4096, value=EMBEDDING_DIM, step=1)
    dropout_rate = st.number_input("Dropout rate", min_value=0.0, max_value=1.0, value=DROPOUT_RATE, step=0.05)
    threshold = st.number_input("Unknown threshold", min_value=0.0, max_value=2.0, value=DISTANCE_THRESHOLD, step=0.01)
    top_k = st.number_input("Top-K matches", min_value=1, max_value=20, value=TOP_K, step=1)

try:
    model = load_model(weights_path, int(image_size), int(embedding_dim), float(dropout_rate))
    centres_df, centre_embeddings = load_gallery_assets(centres_csv_path)
except Exception as e:
    st.error(f"Failed to load model or gallery assets: {e}")
    st.stop()

st.caption(f"Loaded {len(centres_df)} identity centres.")

uploaded_file = st.file_uploader(
    "Upload a penguin image",
    type=["jpg", "jpeg", "png"],
    width="stretch",
)

if uploaded_file is not None:
    try:
        uploaded_image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    except Exception as e:
        st.error(f"Could not read uploaded image: {e}")
        st.stop()

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Uploaded image")
        st.image(uploaded_image, width="stretch")

    with st.spinner("Embedding image and comparing to known identities..."):
        query_embedding = embed_uploaded_image(model, uploaded_image, image_size=int(image_size))
        ranked_df, all_distances = rank_identities(
            query_embedding=query_embedding,
            centres_df=centres_df,
            centre_embeddings=centre_embeddings,
            top_k=int(top_k),
        )

    winner = ranked_df.iloc[0]
    winner_name = str(winner["identity"])
    winner_distance = float(winner["distance"])
    winner_rep_image = str(winner.get("rep_image_path", ""))
    is_unknown = winner_distance > float(threshold)

    with right_col:
        st.subheader("Prediction")
        if is_unknown:
            st.warning(
                f"No confident identity match found. Closest match: {winner_name} "
                f"(distance = {winner_distance:.4f})."
            )
        else:
            st.success(f"Predicted penguin: {winner_name}")
            st.write(f"Distance to nearest centre: {winner_distance:.4f}")

        if path_exists(resolve_from_app_dir(winner_rep_image)):
            st.image(str(resolve_from_app_dir(winner_rep_image)), caption=f"Representative image: {winner_name}", width="stretch")
        else:
            st.info("Representative image not available for the best match.")

    with st.expander("Top matches", expanded=True):
        display_df = ranked_df[["identity", "distance", "rep_image_path"]].copy()
        st.dataframe(display_df, width="stretch", hide_index=True)

    with st.expander("Debug details"):
        st.write("Uploaded file name:", uploaded_file.name)
        st.write("Query embedding shape:", query_embedding.shape)
        st.write("Min distance:", float(np.min(all_distances)))
        st.write("Max distance:", float(np.max(all_distances)))

else:
    st.info("Upload a penguin image to run identity matching.")
