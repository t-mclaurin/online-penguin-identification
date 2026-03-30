from __future__ import annotations

import io
import os
import json
from pathlib import Path
from typing import Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
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
DEFAULT_WEIGHTS = APP_DIR / "app_assets" / "model.weights.h5"
EXAMPLE_IMAGE = APP_DIR / "app_assets" / "example_data" / "example_image.jpeg"
EXAMPLE_META = APP_DIR / "app_assets" / "example_data" / "example_penguins.json"

IMAGE_SIZE = 224
EMBEDDING_DIM = 256
DROPOUT_RATE = 0.1
DISTANCE_THRESHOLD = 0.45
TOP_K = 10


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

def resolve_from_example_dir(path_str: str | Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (EXAMPLE_META.parent / p).resolve()

@st.cache_data
def load_example_penguins():
    with open(EXAMPLE_META, "r", encoding="utf-8") as f:
        return json.load(f)


def get_clicked_penguin(click_x: int, click_y: int, penguins: list[dict]):
    for penguin in penguins:
        box = penguin["bbox"]
        if box["x1"] <= click_x <= box["x2"] and box["y1"] <= click_y <= box["y2"]:
            return penguin
    return None


def draw_selected_box(image_path: str | Path, selected_penguin: dict | None = None) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if selected_penguin is not None:
        draw = ImageDraw.Draw(image)
        box = selected_penguin["bbox"]
        draw.rectangle(
            [box["x1"], box["y1"], box["x2"], box["y2"]],
            outline="red",
            width=6,
        )
    return image

def resize_image_and_boxes(
    image: Image.Image,
    penguins: list[dict],
    target_width: int,
) -> tuple[Image.Image, list[dict]]:
    original_w, original_h = image.size

    if original_w <= target_width:
        return image, penguins

    scale = target_width / original_w
    new_w = target_width
    new_h = int(original_h * scale)

    resized_image = image.resize((new_w, new_h))

    scaled_penguins = []
    for penguin in penguins:
        scaled_penguin = dict(penguin)
        bbox = penguin["bbox"]
        scaled_penguin["bbox"] = {
            "x1": int(bbox["x1"] * scale),
            "y1": int(bbox["y1"] * scale),
            "x2": int(bbox["x2"] * scale),
            "y2": int(bbox["y2"] * scale),
        }
        scaled_penguins.append(scaled_penguin)

    return resized_image, scaled_penguins


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

weights_path = str(DEFAULT_WEIGHTS)
centres_csv_path = str(CENTRES_CSV)
image_size = IMAGE_SIZE
embedding_dim = EMBEDDING_DIM
dropout_rate = DROPOUT_RATE
threshold = DISTANCE_THRESHOLD
top_k = TOP_K

try:
    model = load_model(weights_path, int(image_size), int(embedding_dim), float(dropout_rate))
    centres_df, centre_embeddings = load_gallery_assets(centres_csv_path)
except Exception as e:
    st.error(f"Failed to load model or gallery assets: {e}")
    st.stop()

st.caption(f"Loaded {len(centres_df)} identity centres.")

uploaded_file = st.file_uploader(
    "Upload your own penguin image",
    type=["jpg", "jpeg", "png"],
    width="stretch",
)

with st.expander("Or try an example image", expanded=True):
    st.write("Click one of the penguins below to run a built-in example.")

    try:
        example_penguins = load_example_penguins()
        current_selected = st.session_state.get("selected_example_penguin")

        # Draw the currently selected box on the single displayed image
        boxed_image = draw_selected_box(EXAMPLE_IMAGE, current_selected)
        display_image, scaled_example_penguins = resize_image_and_boxes(
            boxed_image,
            example_penguins,
            target_width=700,
        )

        click_value = streamlit_image_coordinates(
            display_image,
            key="penguin_example_selector",
        )

        if click_value is not None:
            clicked_penguin = get_clicked_penguin(
                click_value["x"],
                click_value["y"],
                scaled_example_penguins,
            )

            if clicked_penguin is not None:
                selected_example_penguin = next(
                    p for p in example_penguins if p["id"] == clicked_penguin["id"]
                )

                current_id = current_selected["id"] if current_selected is not None else None
                new_id = selected_example_penguin["id"]

                if new_id != current_id:
                    st.session_state["selected_example_penguin"] = selected_example_penguin
                    st.rerun()

        selected_example_penguin = st.session_state.get("selected_example_penguin")

        if selected_example_penguin is not None:
            st.caption(f"Selected example: {selected_example_penguin['label']}")

    except Exception as e:
        st.warning(f"Example selector unavailable: {e}")
        selected_example_penguin = None

selected_input_label = None
selected_input_image = None

if uploaded_file is not None:
    selected_input_image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    selected_input_label = "Uploaded image"
elif st.session_state.get("selected_example_penguin") is not None:
    selected_example_penguin = st.session_state["selected_example_penguin"]
    example_crop_path = resolve_from_example_dir(selected_example_penguin["crop_path"])
    selected_input_image = Image.open(example_crop_path).convert("RGB")
    selected_input_label = selected_example_penguin.get("label", "Example image")


if selected_input_image is not None:
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader(selected_input_label)
        st.image(selected_input_image, width="stretch")

    with st.spinner("Embedding image and comparing to known identities..."):
        query_embedding = embed_uploaded_image(
            model,
            selected_input_image,
            image_size=int(image_size),
        )
        ranked_df, all_distances = rank_identities(
            query_embedding=query_embedding,
            centres_df=centres_df,
            centre_embeddings=centre_embeddings,
            top_k=int(top_k),
        )

    winner = ranked_df.iloc[0]
    winner_name = str(winner["identity"])
    winner_distance = float(winner["distance"])
    is_unknown = winner_distance > float(threshold)

    top_match_labels = [
        f"{row['identity']} (distance = {float(row['distance']):.4f})"
        for _, row in ranked_df.iterrows()
    ]

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

        selected_label = st.selectbox(
            "Inspect one of the top matches",
            options=top_match_labels,
            index=0,
        )
        selected_index = top_match_labels.index(selected_label)
        selected_match = ranked_df.iloc[selected_index]
        selected_name = str(selected_match["identity"])
        selected_distance = float(selected_match["distance"])
        selected_rep_image = str(selected_match.get("rep_image_path", ""))

        st.write(f"Selected match: **{selected_name}**")
        st.write(f"Distance: {selected_distance:.4f}")

        if path_exists(resolve_from_app_dir(selected_rep_image)):
            st.image(
                str(resolve_from_app_dir(selected_rep_image)),
                caption=f"Representative image: {selected_name}",
                width="stretch",
            )
        else:
            st.info("Representative image not available for this match.")

    with st.expander("Top matches", expanded=True):
        display_df = ranked_df[["identity", "distance", "rep_image_path"]].copy()
        st.dataframe(display_df, width="stretch", hide_index=True)

    with st.expander("Debug details"):
        if uploaded_file is not None:
            st.write("Uploaded file name:", uploaded_file.name)
        elif st.session_state.get("selected_example_penguin") is not None:
            st.write(
                "Selected example id:",
                st.session_state["selected_example_penguin"].get("id"),
            )
        st.write("Query embedding shape:", query_embedding.shape)
        st.write("Min distance:", float(np.min(all_distances)))
        st.write("Max distance:", float(np.max(all_distances)))

else:
    st.info("Upload a penguin image or click an example penguin to run identity matching.")