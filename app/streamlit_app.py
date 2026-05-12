"""Application Streamlit — Retinal Image Classifier (UI/UX pro).

Lancer depuis la racine du projet :
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_train_data, preprocess_single_image  # noqa: E402
from src.gradcam import gradcam_cnn, gradcam_resnet18, overlay_heatmap  # noqa: E402
from src.model import CNN, ResNet18Transfer, build_model  # noqa: E402

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------

CLASS_LABELS = {
    0: "Classe 0 — Qualité très basse",
    1: "Classe 1 — Qualité basse",
    2: "Classe 2 — Qualité moyenne",
    3: "Classe 3 — Qualité bonne",
    4: "Classe 4 — Qualité très bonne",
}
CLASS_SHORT = {0: "C0", 1: "C1", 2: "C2", 3: "C3", 4: "C4"}
CLASS_COLORS = {
    0: "#EF4444",
    1: "#F59E0B",
    2: "#EAB308",
    3: "#10B981",
    4: "#059669",
}

MODEL_DESCRIPTIONS = {
    "mlp": "Baseline pleinement connectée (sans prior spatial).",
    "cnn": "CNN compact maison (4 conv + 2 pooling + tête FC).",
    "resnet": "Small ResNet entraînée from scratch (skip connections).",
    "resnet18_tl": "ResNet18 pré-entraînée ImageNet, fine-tunée 5 classes.",
}

MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
DATA_DIR = ROOT / "data"
DEFAULT_MODEL = "cnn"
UNCERTAINTY_THRESHOLD = 0.35

# --------------------------------------------------------------------------
# Setup page
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Retinal Image Classifier — IFT 3395/6390",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Retinal Image Classifier — IFT 3395/6390. "
            "Démonstration d'inférence Deep Learning en ligne."
        ),
    },
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ---------- Page background ---------- */
        .main > div { padding-top: 1rem; }

        /* ---------- Hero card ---------- */
        .hero {
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 45%, #EC4899 100%);
            padding: 1.6rem 1.8rem;
            border-radius: 16px;
            color: white;
            box-shadow: 0 10px 30px rgba(99,102,241,0.25);
            margin-bottom: 1.2rem;
        }
        .hero h1 {
            color: white !important;
            margin: 0 0 0.3rem 0 !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }
        .hero p {
            color: rgba(255,255,255,0.92) !important;
            margin: 0;
            font-size: 0.95rem;
        }
        .hero .badges { margin-top: 0.7rem; }
        .hero .badges span {
            display: inline-block;
            background: rgba(255,255,255,0.18);
            border: 1px solid rgba(255,255,255,0.25);
            padding: 0.2rem 0.6rem;
            margin-right: 0.4rem;
            margin-top: 0.3rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 500;
            color: white;
            backdrop-filter: blur(8px);
        }

        /* ---------- Metric cards ---------- */
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(99,102,241,0.12);
        }
        div[data-testid="stMetricLabel"] p {
            font-size: 0.78rem !important;
            color: #6B7280 !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.7rem !important;
            font-weight: 700 !important;
            color: #111827 !important;
        }

        /* ---------- Tabs ---------- */
        button[data-baseweb="tab"] {
            font-weight: 600 !important;
            padding-top: 0.6rem !important;
            padding-bottom: 0.6rem !important;
        }
        div[data-baseweb="tab-list"] {
            gap: 0.4rem;
            border-bottom: 1px solid #E5E7EB;
        }

        /* ---------- Cards (custom blocks) ---------- */
        .card {
            background: white;
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 1rem 1.1rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            margin-bottom: 0.8rem;
        }
        .card-title {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #6B7280;
            margin-bottom: 0.4rem;
        }
        .pred-class {
            font-size: 1.6rem;
            font-weight: 700;
            color: #111827;
            margin: 0;
        }
        .pred-sub {
            color: #6B7280;
            font-size: 0.9rem;
            margin: 0.2rem 0 0 0;
        }
        .pill-ok {
            display: inline-block; padding: 0.2rem 0.55rem;
            border-radius: 999px; background: #DCFCE7; color: #166534;
            font-weight: 600; font-size: 0.8rem;
        }
        .pill-warn {
            display: inline-block; padding: 0.2rem 0.55rem;
            border-radius: 999px; background: #FEF3C7; color: #92400E;
            font-weight: 600; font-size: 0.8rem;
        }
        .pill-bad {
            display: inline-block; padding: 0.2rem 0.55rem;
            border-radius: 999px; background: #FEE2E2; color: #991B1B;
            font-weight: 600; font-size: 0.8rem;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #FAFBFF 0%, #F1F2F8 100%);
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 { color: #111827; }

        /* ---------- File uploader ---------- */
        section[data-testid="stFileUploaderDropzone"] {
            border: 2px dashed #C7D2FE !important;
            background: #F5F3FF !important;
            border-radius: 12px;
        }

        /* ---------- Footer ---------- */
        .footer {
            margin-top: 2rem; padding: 1rem 0;
            border-top: 1px solid #E5E7EB;
            color: #6B7280; font-size: 0.8rem; text-align: center;
        }

        /* hide Streamlit watermarks */
        #MainMenu, footer, header { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Chargement modèle & historique
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    weight_path = MODELS_DIR / f"{model_name}_best.pt"
    if not weight_path.exists():
        return None, None
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    name = checkpoint.get("model_name", model_name)
    if name == "resnet18_tl":
        model = build_model(
            "resnet18_tl",
            num_classes=int(checkpoint.get("num_classes", 5)),
            pretrained=False,
        )
    else:
        model = build_model(
            name,
            in_channels=3,
            num_classes=int(checkpoint.get("num_classes", 5)),
            dimension_of_image=int(checkpoint.get("input_size", 28)),
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


@st.cache_data(show_spinner=False)
def load_history(model_name: str):
    path = MODELS_DIR / f"{model_name}_history.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def one_example_per_class():
    if not (DATA_DIR / "train_data.pkl").exists():
        return None
    images, labels = load_train_data(DATA_DIR)
    out = {}
    for c in range(5):
        idx = np.where(labels.flatten() == c)[0]
        if len(idx):
            out[c] = images[idx[0]]
    return out


def predict(model, image: np.ndarray, imagenet_norm: bool, with_timing: bool = False):
    tensor = preprocess_single_image(image, imagenet_norm=imagenet_norm)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
    elapsed = (time.perf_counter() - t0) * 1000.0
    pred = int(np.argmax(probs))
    if with_timing:
        return pred, probs, elapsed
    return pred, probs


# --------------------------------------------------------------------------
# Plot helpers (Plotly)
# --------------------------------------------------------------------------


def plotly_layout(height: int = 320, title: str | None = None) -> dict:
    layout = {
        "margin": {"l": 30, "r": 10, "t": 40 if title else 10, "b": 30},
        "height": height,
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "font": {"family": "sans-serif", "color": "#374151", "size": 12},
        "xaxis": {"gridcolor": "#F1F5F9", "zerolinecolor": "#E5E7EB"},
        "yaxis": {"gridcolor": "#F1F5F9", "zerolinecolor": "#E5E7EB"},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(0,0,0,0)",
        },
    }
    if title:
        layout["title"] = {"text": title, "x": 0.0, "font": {"size": 14, "color": "#111827"}}
    return layout


def make_prob_bar(probs: np.ndarray, predicted: int) -> go.Figure:
    classes = [CLASS_SHORT[i] for i in range(len(probs))]
    colors = [
        CLASS_COLORS[i] if i == predicted else "#CBD5E1" for i in range(len(probs))
    ]
    fig = go.Figure(
        data=[
            go.Bar(
                x=classes,
                y=probs * 100,
                marker_color=colors,
                hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
                text=[f"{p * 100:.1f}%" for p in probs],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(**plotly_layout(height=280))
    fig.update_yaxes(title_text="Probabilité (%)", range=[0, max(100, float(probs.max() * 110))])
    fig.update_xaxes(title_text="")
    return fig


def make_history_loss(history: dict) -> go.Figure | None:
    if not history.get("train_loss"):
        return None
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history["train_loss"],
            mode="lines+markers",
            name="train loss",
            line={"color": "#6366F1", "width": 2.5},
            marker={"size": 6},
        )
    )
    if history.get("val_loss"):
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["val_loss"],
                mode="lines+markers",
                name="val loss",
                line={"color": "#EC4899", "width": 2.5, "dash": "dash"},
                marker={"size": 6},
            )
        )
    fig.update_layout(**plotly_layout(height=300, title="Loss"))
    fig.update_yaxes(title_text="Loss")
    fig.update_xaxes(title_text="Epoch")
    return fig


def make_history_acc(history: dict) -> go.Figure | None:
    if not history.get("train_acc"):
        return None
    epochs = list(range(1, len(history["train_acc"]) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=[a * 100 for a in history["train_acc"]],
            mode="lines+markers",
            name="train acc",
            line={"color": "#10B981", "width": 2.5},
            marker={"size": 6},
        )
    )
    if history.get("val_acc"):
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[a * 100 for a in history["val_acc"]],
                mode="lines+markers",
                name="val acc",
                line={"color": "#F59E0B", "width": 2.5, "dash": "dash"},
                marker={"size": 6},
            )
        )
    fig.update_layout(**plotly_layout(height=300, title="Accuracy"))
    fig.update_yaxes(title_text="Accuracy (%)", range=[0, 100])
    fig.update_xaxes(title_text="Epoch")
    return fig


def confidence_pill(pmax: float) -> str:
    if pmax >= 0.7:
        return f'<span class="pill-ok">Confiance élevée · {pmax * 100:.1f}%</span>'
    if pmax >= UNCERTAINTY_THRESHOLD:
        return f'<span class="pill-warn">Confiance modérée · {pmax * 100:.1f}%</span>'
    return f'<span class="pill-bad">Confiance faible · {pmax * 100:.1f}%</span>'


# --------------------------------------------------------------------------
# Sections UI
# --------------------------------------------------------------------------


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>🧠 Retinal Image Classifier</h1>
          <p>Inférence Deep Learning en ligne — IFT 3395/6390 · Compétition Kaggle 2 (Fall 2025)</p>
          <div class="badges">
            <span>🐍 PyTorch</span>
            <span>🖼️ Vision</span>
            <span>⚡ Inférence temps réel</span>
            <span>🔍 Grad-CAM</span>
            <span>📊 5 classes</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        available = sorted(
            p.stem.replace("_best", "") for p in MODELS_DIR.glob("*_best.pt")
        )
        if not available:
            st.warning(
                "Aucun modèle entraîné détecté.\n\n"
                "Entraînez-en un :\n```\npython train_and_save.py --model cnn\n```"
            )
            available = [DEFAULT_MODEL]

        model_name = st.selectbox(
            "Modèle",
            available,
            index=0,
            format_func=lambda n: f"{n.upper()}",
            help="Sélectionne le modèle entraîné à utiliser.",
        )
        if model_name in MODEL_DESCRIPTIONS:
            st.caption(MODEL_DESCRIPTIONS[model_name])

        st.divider()
        st.markdown("#### 🎓 Projet")
        st.markdown(
            "Classification d'images **rétiniennes 28×28** "
            "en **5 niveaux de qualité**.\n\n"
            "Cours : IFT 3395/6390 — Apprentissage Machine."
        )
        st.link_button(
            "🔗 Compétition Kaggle",
            "https://www.kaggle.com/competitions/ift-3395-6390-kaggle-2-competition-fall-2025",
            use_container_width=True,
        )

        st.divider()
        st.markdown("#### 🛠️ Lancer un entraînement")
        st.code(
            "python train_and_save.py \\\n  --model resnet18_tl \\\n  --epochs 15 --augment 5 \\\n  --lr-plateau --label-smoothing 0.05",
            language="bash",
        )

        st.divider()
        st.caption("v1.0 · UI/UX pro · MIT")

    return model_name


def render_metrics_row(checkpoint: dict) -> None:
    cols = st.columns(5)
    cols[0].metric("🧬 Modèle", str(checkpoint.get("model_name", "?")).upper())

    val_acc = checkpoint.get("val_accuracy")
    cols[1].metric(
        "🎯 Val. accuracy",
        f"{float(val_acc) * 100:.2f}%" if val_acc is not None else "—",
    )

    f1 = checkpoint.get("val_f1_macro")
    cols[2].metric("📈 F1 macro", f"{float(f1):.3f}" if f1 is not None else "—")

    roc = checkpoint.get("val_roc_auc_ovr_macro")
    cols[3].metric(
        "📐 ROC AUC OvR",
        f"{float(roc):.3f}" if (roc is not None and roc == roc) else "—",
    )

    n_params = checkpoint.get("num_parameters")
    cols[4].metric(
        "⚙️ Paramètres",
        f"{int(n_params):,}".replace(",", " ") if n_params is not None else "—",
    )


def render_history_panel(history: dict | None) -> None:
    if not history or not history.get("train_loss"):
        st.info("Pas d'historique disponible pour ce modèle.")
        return
    c1, c2 = st.columns(2)
    fig_loss = make_history_loss(history)
    fig_acc = make_history_acc(history)
    with c1:
        if fig_loss:
            st.plotly_chart(fig_loss, use_container_width=True, config={"displayModeBar": False})
    with c2:
        if fig_acc:
            st.plotly_chart(fig_acc, use_container_width=True, config={"displayModeBar": False})

    if history.get("lr"):
        with st.expander("📉 Learning rate par epoch", expanded=False):
            epochs = list(range(1, len(history["lr"]) + 1))
            fig_lr = go.Figure(
                go.Scatter(
                    x=epochs,
                    y=history["lr"],
                    mode="lines+markers",
                    line={"color": "#8B5CF6", "width": 2.5},
                    marker={"size": 5},
                )
            )
            fig_lr.update_layout(**plotly_layout(height=250))
            fig_lr.update_yaxes(title_text="Learning rate", type="log")
            fig_lr.update_xaxes(title_text="Epoch")
            st.plotly_chart(fig_lr, use_container_width=True, config={"displayModeBar": False})

    cols = st.columns(3)
    train_time = history.get("train_seconds")
    if train_time is not None:
        cols[0].metric("⏱️ Temps d'entraînement", f"{float(train_time):.1f} s")
    n_epochs = len(history.get("train_loss", []))
    cols[1].metric("📚 Epochs exécutés", str(n_epochs))
    final_val = history.get("val_acc", [None])[-1] if history.get("val_acc") else None
    if final_val is not None:
        cols[2].metric("🏁 Val acc (dernière)", f"{float(final_val) * 100:.2f}%")


def render_inference_tab(model, checkpoint: dict) -> None:
    imagenet_norm = bool(checkpoint.get("use_imagenet_norm", False))

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="card-title">📥 Source de l\'image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Glissez-déposez une image (PNG/JPG, idéalement 28×28 RGB)",
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=False,
            label_visibility="visible",
        )

        sample_images = sorted(ASSETS_DIR.glob("*.png")) if ASSETS_DIR.exists() else []
        sample_choice = None
        if sample_images:
            sample_choice = st.selectbox(
                "...ou choisissez une image d'exemple (assets)",
                options=["(aucune)"] + [p.name for p in sample_images],
            )

        image_array = None
        pil_image = None
        if uploaded is not None:
            pil_image = Image.open(uploaded).convert("RGB")
            image_array = np.array(pil_image)
        elif sample_choice and sample_choice != "(aucune)":
            pil_image = Image.open(ASSETS_DIR / sample_choice).convert("RGB")
            image_array = np.array(pil_image)

        if pil_image is not None:
            st.image(pil_image, caption=f"Image d'entrée · {pil_image.size[0]}×{pil_image.size[1]}", use_container_width=True)

    with right:
        if image_array is None:
            st.markdown(
                """
                <div class="card" style="text-align:center; padding:2.5rem 1rem;">
                  <div style="font-size:3rem;">🖼️</div>
                  <div class="card-title" style="margin-top:0.6rem;">En attente</div>
                  <p style="color:#6B7280; margin:0;">
                    Téléversez une image ou choisissez un exemple<br/>
                    pour lancer la prédiction.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        pred, probs, ms = predict(model, image_array, imagenet_norm, with_timing=True)
        pmax = float(probs[pred])

        st.markdown('<div class="card-title">🎯 Prédiction</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="card">
              <p class="pred-class">{CLASS_LABELS[pred]}</p>
              <p class="pred-sub">{confidence_pill(pmax)} &nbsp;·&nbsp; inférence en {ms:.1f} ms</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            make_prob_bar(probs, pred),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        with st.expander("🔍 Détails probabilités", expanded=False):
            df = pd.DataFrame(
                {
                    "classe": [CLASS_LABELS[i] for i in range(len(probs))],
                    "probabilité": [f"{p * 100:.2f}%" for p in probs],
                    "raw": probs,
                }
            )
            st.dataframe(df.drop(columns=["raw"]), use_container_width=True, hide_index=True)

    # Grad-CAM section (full width)
    st.divider()
    cam_col1, cam_col2 = st.columns([1, 3])
    with cam_col1:
        show_cam = st.toggle(
            "🔥 Activer Grad-CAM",
            value=False,
            help="Carte d'activation : montre où le modèle regarde pour décider.",
        )
    if show_cam:
        name = checkpoint.get("model_name", "")
        x = preprocess_single_image(image_array, imagenet_norm=imagenet_norm)
        x.requires_grad_(True)
        try:
            if name == "resnet18_tl" and isinstance(model, ResNet18Transfer):
                hm = gradcam_resnet18(model, x, pred)
            elif isinstance(model, CNN):
                hm = gradcam_cnn(model, x, pred)
            else:
                st.info("Grad-CAM disponible pour `cnn` et `resnet18_tl` seulement.")
                hm = None
            if hm is not None:
                ov = overlay_heatmap(image_array.astype(np.uint8), hm)
                gc1, gc2 = st.columns(2)
                with gc1:
                    st.image(image_array, caption="Original", use_container_width=True)
                with gc2:
                    st.image(ov, caption=f"Grad-CAM · classe prédite {CLASS_SHORT[pred]}", use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Grad-CAM : {e}")


def render_examples_tab() -> None:
    st.markdown('<div class="card-title">🖼️ Un exemple par classe (jeu d\'entraînement)</div>', unsafe_allow_html=True)
    ex = one_example_per_class()
    if not ex:
        st.info("Fichier `data/train_data.pkl` introuvable. Place les pickles dans `data/`.")
        return
    cols = st.columns(5)
    for c in range(5):
        with cols[c]:
            border_color = CLASS_COLORS[c]
            if c in ex:
                buf = io.BytesIO()
                Image.fromarray(ex[c]).save(buf, format="PNG")
                st.image(ex[c], caption=CLASS_SHORT[c], use_container_width=True)
                st.markdown(
                    f"<div style='text-align:center; color:{border_color}; font-weight:600;'>"
                    f"{CLASS_LABELS[c]}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption(f"Classe {c} : aucun exemple")

    st.divider()
    st.caption(
        "💡 Note : les 5 classes représentent un niveau croissant de qualité d'image rétinienne. "
        "Le modèle apprend à distinguer ces niveaux à partir de 1 080 images d'entraînement."
    )


def render_batch_tab(model, checkpoint: dict) -> None:
    imagenet_norm = bool(checkpoint.get("use_imagenet_norm", False))
    st.markdown('<div class="card-title">📦 Inférence en lot</div>', unsafe_allow_html=True)
    files = st.file_uploader(
        "Téléversez plusieurs images",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
        key="batch_uploader",
    )
    if not files:
        st.markdown(
            """
            <div class="card" style="text-align:center; padding:2rem 1rem;">
              <div style="font-size:2.5rem;">📁</div>
              <div class="card-title" style="margin-top:0.6rem;">En attente de fichiers</div>
              <p style="color:#6B7280; margin:0;">
                Sélectionnez plusieurs images pour générer un rapport CSV.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    rows = []
    progress = st.progress(0.0, text="Inférence en cours...")
    for i, f in enumerate(files, 1):
        pil = Image.open(f).convert("RGB")
        pred, probs = predict(model, np.array(pil), imagenet_norm)
        rows.append({
            "filename": f.name,
            "prediction": pred,
            "label": CLASS_LABELS[pred],
            "confidence_max": float(probs.max()),
            "uncertain": float(probs.max()) < UNCERTAINTY_THRESHOLD,
            **{f"p_class_{c}": float(probs[c]) for c in range(len(probs))},
        })
        progress.progress(i / len(files), text=f"Inférence... ({i}/{len(files)})")
    progress.empty()
    df = pd.DataFrame(rows)

    cols = st.columns(4)
    cols[0].metric("📁 Images traitées", len(df))
    cols[1].metric("🎯 Confiance moyenne", f"{df['confidence_max'].mean() * 100:.1f}%")
    cols[2].metric("⚠️ Incertains", int(df["uncertain"].sum()))
    cols[3].metric("🏷️ Classes prédites", df["prediction"].nunique())

    # Distribution
    dist = df["prediction"].value_counts().sort_index().reindex(range(5), fill_value=0)
    fig = go.Figure(
        go.Bar(
            x=[CLASS_SHORT[c] for c in dist.index],
            y=dist.values,
            marker_color=[CLASS_COLORS[c] for c in dist.index],
            text=dist.values,
            textposition="outside",
        )
    )
    fig.update_layout(**plotly_layout(height=260, title="Distribution des prédictions"))
    fig.update_yaxes(title_text="# images")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "📥 Télécharger le CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_about_tab() -> None:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(
            """
### 📋 Le projet

**Tâche** : classification multi-classes (5 niveaux de qualité) d'images rétiniennes 28×28×3.

**Données** : 1 080 images d'entraînement, 400 images de test (Kaggle, format Pickle).

**Pipeline DL**
1. Normalisation \\([0, 1]\\) + conversion en tenseurs PyTorch (option **ImageNet** pour ResNet18 TL).
2. Augmentation forte (rotation, recadrage, flip, jitter, affine, etc.) ×5–×15.
3. **CrossEntropyLoss** (label smoothing optionnel), **Adam/SGD**, **weight decay**, **ReduceLROnPlateau**, **early stopping** sur la val loss.

**Modèles comparés**
- 🟦 MLP baseline (sans prior spatial)
- 🟪 CNN maison (4 conv + tête FC)
- 🟧 SmallResNet (skip connections, scratch)
- 🟩 **ResNet18 Transfer Learning** (ImageNet → 5 classes)

**Métriques** : accuracy, **F1 macro**, **ROC AUC multiclasse OvR**, matrice de confusion.

**Bonus** : cette application Streamlit (UI/UX pro + Grad-CAM).
            """
        )
    with c2:
        st.markdown('<div class="card-title">📐 Architecture CNN</div>', unsafe_allow_html=True)
        architecture = ASSETS_DIR / "architecture.webp"
        if architecture.exists():
            st.image(str(architecture), use_container_width=True)
        else:
            st.info("Image `assets/architecture.webp` introuvable.")
        st.markdown('<div class="card-title" style="margin-top:1rem;">🧪 Évaluation rapide</div>', unsafe_allow_html=True)
        st.code(
            "python scripts/compare_models.py \\\n  --epochs 8 --augment 3 --lr-plateau",
            language="bash",
        )


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer">
          IFT 3395/6390 — Projet Deep Learning · Retinal Image Classifier
          &nbsp;·&nbsp; Built with ❤️ + Streamlit + PyTorch
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    inject_css()
    render_hero()

    model_name = render_sidebar()
    model, checkpoint = load_model(model_name)
    history = load_history(model_name)

    if model is None or checkpoint is None:
        st.error(
            f"❌ Aucun poids trouvé pour le modèle « **{model_name}** ». "
            "Entraînez-le d'abord avec :\n\n"
            f"```\npython train_and_save.py --model {model_name}\n```"
        )
        st.divider()
        render_about_tab()
        render_footer()
        return

    render_metrics_row(checkpoint)

    tab_pred, tab_hist, tab_ex, tab_batch, tab_about = st.tabs(
        [
            "🔮 Prédiction",
            "📊 Apprentissage",
            "🖼️ Exemples",
            "📦 Inférence en lot",
            "ℹ️ À propos",
        ]
    )
    with tab_pred:
        render_inference_tab(model, checkpoint)
    with tab_hist:
        st.markdown('<div class="card-title">📊 Courbes d\'apprentissage</div>', unsafe_allow_html=True)
        render_history_panel(history)
    with tab_ex:
        render_examples_tab()
    with tab_batch:
        render_batch_tab(model, checkpoint)
    with tab_about:
        render_about_tab()

    render_footer()


if __name__ == "__main__":
    main()
