from __future__ import annotations

import base64
import hashlib
import io
import os
import json
from pathlib import Path
from typing import Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import plotly.graph_objects as go
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import tensorflow as tf

from model import build_embedding_model


# ============================================================
# Config
# ============================================================
APP_TITLE = "Penguin Identifier Demo"
APP_SUBTITLE = (
    "Upload an image of a Humboldt Penguin from London Zoo. The app embeds it, compares it to precomputed "
    "identity centres, and returns the closest matches from an index of 71 known penguins."
    "Humboldt Penguins have unique spot patterns on their fronts that humans and this model can use to tell them apart."
)

APP_DIR = Path(__file__).resolve().parent
CENTRES_CSV = APP_DIR / "app_assets" / "identity_centres.csv"
DEFAULT_WEIGHTS = APP_DIR / "app_assets" / "best.weights.h5"
EXAMPLE_IMAGE = APP_DIR / "app_assets" / "example_data" / "example_image.jpeg"
EXAMPLE_META = APP_DIR / "app_assets" / "example_data" / "example_penguins.json"
EXAMPLE_SETS_DIR = APP_DIR / "app_assets" / "example_sets"

IMAGE_SIZE = 224
EMBEDDING_DIM = 256
DROPOUT_RATE = 0.1
DISTANCE_THRESHOLD = 0.45
TOP_K = 10
CONFIDENCE_RING_RADIUS = 0.60
MAP_HEIGHT = 460
MAX_IMAGE_DISPLAY_HEIGHT = 550
EXAMPLE_SET_PREVIEW_HEIGHT = 320
UPLOAD_GUIDE_TEXT = """
**Image upload guide**

- This model can only identify Humboldt Penguins from London Zoo.
- Use a clear image with a single penguin as the main subject.
- Photos must be taken from the front or side of the penguin. 
- Crop tightly around the penguin when possible.
- Avoid heavy blur, occlusion, or very distant subjects.
- JPG, JPEG and PNG files are supported.
- Alternatively, click on a penguin in one of the examples below. 
"""


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


@st.cache_data

def load_example_sets() -> list[dict]:
    if not EXAMPLE_SETS_DIR.exists():
        return []

    valid_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    sets: list[dict] = []
    for folder in sorted(EXAMPLE_SETS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        images = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in valid_suffixes]
        if images:
            sets.append({"identity": folder.name, "images": images})
    return sets


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
        draw.rectangle([box["x1"], box["y1"], box["x2"], box["y2"]], outline="red", width=6)
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


def get_sample_signature(
    uploaded_file,
    selected_example_penguin: dict | None,
    selected_example_set_image_path: Path | None,
) -> str | None:
    if uploaded_file is not None:
        digest = hashlib.sha1(uploaded_file.getvalue()).hexdigest()[:16]
        return f"upload:{digest}"
    if selected_example_set_image_path is not None:
        return f"set:{selected_example_set_image_path.as_posix()}"
    if selected_example_penguin is not None:
        return f"example:{selected_example_penguin.get('id', 'unknown')}"
    return None


def prepare_image_for_display(image: Image.Image, max_height: int) -> Image.Image:
    display_image = image.copy()
    width, height = display_image.size
    if height <= max_height:
        return display_image
    scale = max_height / float(height)
    new_width = max(1, int(round(width * scale)))
    return display_image.resize((new_width, max_height))


def render_constrained_image(image: Image.Image, max_height: int):
    display_image = prepare_image_for_display(image, max_height=max_height)
    buffer = io.BytesIO()
    display_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    st.markdown(
        f"""
        <div style="height:{max_height}px; display:flex; align-items:center; justify-content:center; overflow:hidden;">
            <img src="data:image/png;base64,{encoded}"
                 style="max-height:{max_height}px; max-width:100%; width:auto; height:auto; object-fit:contain; display:block;" />
        </div>
        """,
        unsafe_allow_html=True,
    )


def add_image_border(image: Image.Image, color: str = "red", border_width: int = 2) -> Image.Image:
    bordered = image.copy().convert("RGB")
    draw = ImageDraw.Draw(bordered)
    width, height = bordered.size
    for offset in range(border_width):
        draw.rectangle(
            [offset, offset, max(offset, width - 1 - offset), max(offset, height - 1 - offset)],
            outline=color,
        )
    return bordered


def build_example_set_preview(image_path: Path, is_selected: bool, max_height: int) -> Image.Image:
    preview = Image.open(image_path).convert("RGB")
    preview = prepare_image_for_display(preview, max_height=max_height)
    if is_selected:
        preview = add_image_border(preview, color="red", border_width=2)
    return preview


def set_active_group_example(selected_penguin: dict):
    st.session_state["active_example_source"] = "group"
    st.session_state["selected_example_penguin"] = selected_penguin
    st.session_state["selected_example_set_image_path"] = None


def set_active_example_set(image_path: Path):
    st.session_state["active_example_source"] = "set"
    st.session_state["selected_example_set_image_path"] = str(image_path)
    st.session_state["selected_example_penguin"] = None


# ============================================================
# Image preprocessing
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
        raise ValueError("centre_index values must run from 0..N-1 and match the row order of the .npy file.")

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
    sims = centre_embeddings @ query_embedding
    dist_sq = np.clip(2.0 - 2.0 * sims, a_min=0.0, a_max=None)
    return np.sqrt(dist_sq).astype(np.float32)


def rank_identities(query_embedding: np.ndarray, centres_df: pd.DataFrame, centre_embeddings: np.ndarray, top_k: int):
    dists = compute_distances(query_embedding, centre_embeddings)
    order = np.argsort(dists)
    top_idx = order[:top_k]

    ranked = centres_df.iloc[top_idx].copy().reset_index(drop=True)
    ranked["distance"] = dists[top_idx]
    ranked["centre_row_index"] = top_idx
    return ranked, dists, order


# ============================================================
# Radial identity-space visualisation helpers
# ============================================================
def build_identity_space_projection(
    centres_df: pd.DataFrame,
    centre_embeddings: np.ndarray,
    all_distances: np.ndarray,
) -> pd.DataFrame:
    viz_df = centres_df[["identity", "centre_index", "rep_image_path"]].copy().reset_index(drop=True)
    viz_df["distance"] = all_distances.astype(np.float32)
    viz_df["rank"] = np.argsort(np.argsort(all_distances)) + 1

    base_xy = centre_embeddings[:, :2].astype(np.float32)
    base_xy = base_xy - base_xy.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(base_xy, axis=1)

    if np.all(norms < 1e-8):
        angles = np.linspace(0, 2 * np.pi, len(viz_df), endpoint=False)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    else:
        directions = np.zeros_like(base_xy)
        valid = norms > 1e-8
        directions[valid] = base_xy[valid] / norms[valid, None]
        if np.any(~valid):
            fallback_angles = np.linspace(0, 2 * np.pi, np.sum(~valid), endpoint=False)
            directions[~valid] = np.stack([np.cos(fallback_angles), np.sin(fallback_angles)], axis=1)

    coords = directions * all_distances[:, None]
    viz_df["x"] = coords[:, 0]
    viz_df["y"] = coords[:, 1]
    return viz_df.sort_values("rank").reset_index(drop=True)


def render_identity_space_chart(viz_df: pd.DataFrame, selected_identity: str, key: str, top_k: int):
    selected_idx_candidates = viz_df.index[viz_df["identity"] == selected_identity].tolist()
    selected_idx = selected_idx_candidates[0] if selected_idx_candidates else 0

    is_top_k = viz_df["rank"].to_numpy() <= int(top_k)
    identity_colors = np.where(is_top_k, "#4C78A8", "#66C2A5").astype(object)
    identity_sizes = np.where(is_top_k, 11, 8).astype(object)
    if 0 <= selected_idx < len(viz_df):
        identity_colors[selected_idx] = "#F2C94C"
        identity_sizes[selected_idx] = 14

    identity_text = np.where(is_top_k, viz_df["identity"].astype(str).to_numpy(), "")

    ring_angles = np.linspace(0, 2 * np.pi, 240)
    max_extent = max(
        0.8,
        float(np.max(np.abs(viz_df[["x", "y"]].to_numpy()))) if len(viz_df) else 0.0,
        CONFIDENCE_RING_RADIUS,
    ) * 1.15

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=CONFIDENCE_RING_RADIUS * np.cos(ring_angles),
            y=CONFIDENCE_RING_RADIUS * np.sin(ring_angles),
            mode="lines",
            line=dict(width=1.5, dash="dot", color="rgba(120,120,120,0.7)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0],
            y=[0.0],
            mode="markers+text",
            marker=dict(size=11, color="#B9D7EA"),
            text=["Sample"],
            textposition="top center",
            textfont=dict(size=12, color="#FFFFFF", family="Arial Black, Arial, sans-serif"),
            hovertemplate="Sample image<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=viz_df["x"],
            y=viz_df["y"],
            mode="markers+text",
            text=identity_text.tolist(),
            textposition="top center",
            marker=dict(size=identity_sizes.tolist(), color=identity_colors.tolist()),
            customdata=np.stack(
                [
                    viz_df["identity"].astype(str).to_numpy(),
                    np.round(viz_df["distance"].to_numpy(dtype=np.float32), 2),
                    viz_df["rank"].astype(int).to_numpy(),
                ],
                axis=1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Distance: %{customdata[1]:.2f}<br>"
                "Rank: %{customdata[2]}<extra></extra>"
            ),
            selected=dict(marker=dict(opacity=1.0), textfont=dict(color="#FFFFFF")),
            unselected=dict(marker=dict(opacity=1.0), textfont=dict(color="#FFFFFF")),
            textfont=dict(size=12, color="#FFFFFF", family="Arial Black, Arial, sans-serif"),
            cliponaxis=False,
            showlegend=False,
        )
    )

    fig.update_layout(
        xaxis=dict(range=[-max_extent, max_extent], showgrid=False, zeroline=False, showticklabels=False, title=None),
        yaxis=dict(
            range=[-max_extent, max_extent],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title=None,
            scaleanchor="x",
            scaleratio=1,
        ),
        height=MAP_HEIGHT,
        margin=dict(l=10, r=10, t=32, b=10),
        clickmode="event+select",
        dragmode=False,
    )

    return st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        on_select="rerun",
        selection_mode="points",
    )


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.write(APP_SUBTITLE)

weights_path = str(DEFAULT_WEIGHTS)
centres_csv_path = str(CENTRES_CSV)

try:
    model = load_model(weights_path, IMAGE_SIZE, EMBEDDING_DIM, DROPOUT_RATE)
    centres_df, centre_embeddings = load_gallery_assets(centres_csv_path)
except Exception as e:
    st.error(f"Failed to load model or gallery assets: {e}")
    st.stop()

st.caption(f"Loaded {len(centres_df)} identity centres.")
st.info("Upload a penguin image or click an example penguin to run identity matching.")

if "active_example_source" not in st.session_state:
    st.session_state["active_example_source"] = None
if "last_group_example_click" not in st.session_state:
    st.session_state["last_group_example_click"] = None

upload_col, guide_col = st.columns([5, 1])
with upload_col:
    uploaded_file = st.file_uploader("Upload your own penguin image", type=["jpg", "jpeg", "png"], width="stretch")
with guide_col:
    st.write("")
    with st.popover("Upload guide"):
        st.markdown(UPLOAD_GUIDE_TEXT)

with st.expander("Try selecting a penguin from an example image", expanded=True):
    st.write("Click one of the four, forward-facing penguins below to run it through the identification model. "
             "An object detection model is used to automatically handle this step at scale.")
    try:
        example_penguins = load_example_penguins()
        current_selected = st.session_state.get("selected_example_penguin")
        boxed_image = draw_selected_box(EXAMPLE_IMAGE, current_selected)
        display_image, scaled_example_penguins = resize_image_and_boxes(boxed_image, example_penguins, target_width=700)
        click_value = streamlit_image_coordinates(display_image, key="penguin_example_selector")

        if click_value is not None:
            click_signature = f"{click_value.get('x')}:{click_value.get('y')}"
            if click_signature != st.session_state.get("last_group_example_click"):
                st.session_state["last_group_example_click"] = click_signature
                clicked_penguin = get_clicked_penguin(click_value["x"], click_value["y"], scaled_example_penguins)
                if clicked_penguin is not None:
                    selected_example_penguin = next(p for p in example_penguins if p["id"] == clicked_penguin["id"])
                    set_active_group_example(selected_example_penguin)
                    st.rerun()

        selected_example_penguin = st.session_state.get("selected_example_penguin")
        if selected_example_penguin is not None:
            st.caption(f"Selected example: {selected_example_penguin['label']}")
    except Exception as e:
        st.warning(f"Example selector unavailable: {e}")

with st.expander("Or compare images of the same Humboldt penguin", expanded=True):
    example_sets = load_example_sets()
    if not example_sets:
        st.info("No example sets found in app_assets/example_sets.")
    else:
        if "example_set_index" not in st.session_state:
            st.session_state["example_set_index"] = 0

        current_index = int(st.session_state.get("example_set_index", 0)) % len(example_sets)
        example_set_labels = [f"Identity {idx + 1}" for idx in range(len(example_sets))]
        current_label = example_set_labels[current_index]

        selected_set_label = st.selectbox(
            "Choose an example set. All these images are new to the model",
            options=example_set_labels,
            index=current_index,
            key="example_set_selectbox",
        )

        if selected_set_label != current_label:
            current_index = example_set_labels.index(selected_set_label)
            st.session_state["example_set_index"] = current_index

        current_set = example_sets[current_index]
        st.caption("Choose one of the example photos below and compare model performance across variation in age, image quality, pose and more.")

        selected_example_set_path_str = st.session_state.get("selected_example_set_image_path")
        image_cols = st.columns(3, gap="medium")
        for idx, image_path in enumerate(current_set["images"][:3]):
            with image_cols[idx]:
                is_selected = selected_example_set_path_str == str(image_path)
                preview = build_example_set_preview(
                    image_path,
                    is_selected=is_selected,
                    max_height=EXAMPLE_SET_PREVIEW_HEIGHT,
                )
                render_constrained_image(preview, EXAMPLE_SET_PREVIEW_HEIGHT)
                if st.button(
                    f"Select",
                    key=f"select_example_set_image_{current_index}_{idx}_{image_path.name}",
                    use_container_width=True,
                ):
                    set_active_example_set(image_path)
                    st.rerun()

selected_input_label = None
selected_input_image = None
selected_example_penguin = st.session_state.get("selected_example_penguin")
selected_example_set_image_path_str = st.session_state.get("selected_example_set_image_path")
selected_example_set_image_path = Path(selected_example_set_image_path_str) if selected_example_set_image_path_str else None
active_example_source = st.session_state.get("active_example_source")

if uploaded_file is not None:
    selected_input_image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    selected_input_label = "Uploaded image"
elif active_example_source == "set" and selected_example_set_image_path is not None and selected_example_set_image_path.exists():
    selected_input_image = Image.open(selected_example_set_image_path).convert("RGB")
    selected_input_label = "Example image"
elif active_example_source == "group" and selected_example_penguin is not None:
    example_crop_path = resolve_from_example_dir(selected_example_penguin["crop_path"])
    selected_input_image = Image.open(example_crop_path).convert("RGB")
    selected_input_label = selected_example_penguin.get("label", "Example image")
elif selected_example_set_image_path is not None and selected_example_set_image_path.exists():
    selected_input_image = Image.open(selected_example_set_image_path).convert("RGB")
    selected_input_label = "Example image"
elif selected_example_penguin is not None:
    example_crop_path = resolve_from_example_dir(selected_example_penguin["crop_path"])
    selected_input_image = Image.open(example_crop_path).convert("RGB")
    selected_input_label = selected_example_penguin.get("label", "Example image")

sample_signature = get_sample_signature(uploaded_file, selected_example_penguin, selected_example_set_image_path)

if selected_input_image is not None:
    with st.spinner("Embedding image and comparing to known identities..."):
        query_embedding = embed_uploaded_image(model, selected_input_image, image_size=IMAGE_SIZE)
        ranked_df, all_distances, _ = rank_identities(
            query_embedding=query_embedding,
            centres_df=centres_df,
            centre_embeddings=centre_embeddings,
            top_k=TOP_K,
        )
        viz_df = build_identity_space_projection(centres_df=centres_df, centre_embeddings=centre_embeddings, all_distances=all_distances)

    winner_name = str(ranked_df.iloc[0]["identity"])

    if st.session_state.get("last_sample_signature") != sample_signature:
        st.session_state["selected_match_identity"] = winner_name
        st.session_state["last_sample_signature"] = sample_signature

    selected_identity = st.session_state.get("selected_match_identity", winner_name)
    if selected_identity not in set(viz_df["identity"].tolist()):
        selected_identity = winner_name
        st.session_state["selected_match_identity"] = selected_identity

    sample_col, map_col, rep_col = st.columns([1.15, 1.7, 1.15], gap="large")

    with sample_col:
        st.subheader(selected_input_label)
        render_constrained_image(selected_input_image, MAX_IMAGE_DISPLAY_HEIGHT)

    with map_col:
        st.subheader("Identity space")
        chart_key = f"identity_space_plot_{sample_signature}"
        chart_event = render_identity_space_chart(viz_df, selected_identity, key=chart_key, top_k=TOP_K)
        st.write(
            "This map represents the model's output, identities that are closer to the centre are considered more similar to the sample. "
            "Click on the points to explore the top matches and compare their representative images."
        )
        st.caption(
            "Note: Distances between the sample and the identities are accurate, but distances between the identities are only approximate. "
            f"The dotted ring marks an arbitrary confidence threshold of {CONFIDENCE_RING_RADIUS:.1f}."
        )

        try:
            selection = chart_event.selection if hasattr(chart_event, "selection") else None
            points = selection.get("points", []) if selection else []
            if points:
                point_data = points[0]
                if point_data.get("curve_number") == 2:
                    point_index = point_data.get("point_index")
                    if point_index is not None and 0 <= int(point_index) < len(viz_df):
                        clicked_identity = str(viz_df.iloc[int(point_index)]["identity"])
                        if clicked_identity != st.session_state.get("selected_match_identity"):
                            st.session_state["selected_match_identity"] = clicked_identity
                            st.rerun()
        except Exception:
            pass

    with rep_col:
        selected_row_df = viz_df[viz_df["identity"] == st.session_state["selected_match_identity"]]
        if selected_row_df.empty:
            selected_row_df = ranked_df.iloc[[0]]
        selected_row = selected_row_df.iloc[0]

        selected_name = str(selected_row["identity"])
        selected_rank = int(selected_row["rank"])
        selected_rep_image = str(selected_row.get("rep_image_path", ""))

        st.subheader(f"Prediction: #{selected_rank} {selected_name}")

        if path_exists(resolve_from_app_dir(selected_rep_image)):
            rep_image = Image.open(resolve_from_app_dir(selected_rep_image)).convert("RGB")
            render_constrained_image(rep_image, MAX_IMAGE_DISPLAY_HEIGHT)
        else:
            st.info("Representative image not available for this match.")

        st.markdown("Not your penguin? Compare to the next closest match:")
        nav_left, nav_mid, nav_right = st.columns([1, 3, 1])
        with nav_left:
            if st.button("←", use_container_width=True, disabled=selected_rank <= 1, key=f"prev_rank_{sample_signature}"):
                prev_identity = str(viz_df.iloc[selected_rank - 2]["identity"])
                st.session_state["selected_match_identity"] = prev_identity
                st.rerun()
        with nav_right:
            if st.button("→", use_container_width=True, disabled=selected_rank >= len(viz_df), key=f"next_rank_{sample_signature}"):
                next_identity = str(viz_df.iloc[selected_rank]["identity"])
                st.session_state["selected_match_identity"] = next_identity
                st.rerun()
else:
    st.info("Upload a penguin image or click an example penguin to run identity matching.")
