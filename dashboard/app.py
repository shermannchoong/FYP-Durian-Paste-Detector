import os, time, io, datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import onnxruntime as ort
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib as mpl

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image

import yaml

def load_css(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("style.css")

mpl.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.12,
})

# =========================
# Plotly Styling Helper
# =========================
def _style_fig(fig, title:str):
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=14),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    return fig

# =========================
# Constants & Defaults
# =========================
PAGE_TITLE = "Durian Paste Dirt Detector (Faster R-CNN)"
DEFAULT_ONNX = str(Path(__file__).resolve().parents[1] / "models" / "model_final.onnx")
DEFAULT_CFG  = str(Path(__file__).resolve().parents[1] / "models" / "faster_rcnn_config.yaml")  

DEFAULT_CLASSES = ["dirt_large", "dirt_small", "seed"]  # overridden if cfg has names

# Optional: remap model class names to desired display names
# Example requested mapping: seed -> dirt_small, dirt_small -> dirt_large
LABEL_REMAP = {"seed": "dirt_small", "dirt_small": "dirt_large"}

FACE_CASCADE_PATH = str(Path(__file__).resolve().parents[1] / "assets" / "haarcascade_frontalface_default.xml")

CLEAN_MAX_COUNT = 4

# =========================
# Page chrome
# =========================
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# Optional CSS (dashboard/style.css)
def inject_local_css():
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

inject_local_css()

# Hide the very first caption on the page (the pre-login usage hint)
st.markdown(
    "<style>div[data-testid='stCaptionContainer']:first-of-type{display:none!important}</style>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="page-hero">
      <h1>Durian Paste Dirt Detector</h1>
      <p class="sub">Final Year Project — <strong>Shermann Choong</strong></p>
      <p class="hint">Upload → click Detect Dirt → view detections, counts, graphs, and export.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
# =========================
# AUTH (login / register) — drop-in block
# =========================
import datetime as _dt
import hashlib
from pathlib import Path
import pandas as pd
import streamlit as st


LOGIN_LOG = Path("login_log.csv")

# --- Simple user DB helpers (CSV) ---
def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def _read_user_db() -> pd.DataFrame:
    if LOGIN_LOG.exists():
        try:
            df = pd.read_csv(LOGIN_LOG)
        except Exception:
            df = pd.DataFrame(columns=["user", "password", "time"])
    else:
        df = pd.DataFrame(columns=["user", "password", "time"])
    for c in ["user", "password", "time"]:
        if c not in df.columns:
            df[c] = ""
    return df[["user", "password", "time"]]

def _save_user_db(df: pd.DataFrame) -> None:
    df = df[["user", "password", "time"]]
    df.to_csv(LOGIN_LOG, index=False)

def _user_exists(username: str) -> bool:
    df = _read_user_db()
    return username in set(df["user"].astype(str))

def _register_user(username: str, password: str) -> bool:
    if not username or not password:
        return False
    df = _read_user_db()
    if username in set(df["user"].astype(str)):
        return False
    row = pd.DataFrame([
        {
            "user": username,
            "password": _hash_pw(password),
            "time": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    ])
    df = pd.concat([df, row], ignore_index=True)
    _save_user_db(df)
    return True

def _verify_user(username: str, password: str) -> bool:
    df = _read_user_db()
    if df.empty:
        return False
    cand = df[df["user"].astype(str) == username]
    if cand.empty:
        return False
    return cand.iloc[0]["password"] == _hash_pw(password)

def login_ui_styled() -> bool:
    # --- Init session ---
    if "auth" not in st.session_state:
        st.session_state.auth = {"user": None}

    # --- Already logged in ---
    if st.session_state.auth["user"]:
        left, right = st.columns([3, 1])
        left.success(f"Logged in as **{st.session_state.auth['user']}**")
        if right.button("Logout"):
            st.session_state.auth = {"user": None}
            st.rerun()
        return True

    # --- Sidebar panel (Navigation / Login) ---
    with st.sidebar:
        st.markdown(
            """
            <div class="nav-panel">
            <div class="nav-title">Navigation</div>

            <!-- Give Login its own class so we can style it -->
            <div class="nav-entry nav-login"> Login</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Centered black login card ---
    _, body, _ = st.columns([1, 1.2, 1])
    with body:
        st.markdown('<div class="auth-title">Login</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Enter your credentials to continue.</div>', unsafe_allow_html=True)

        with st.form("login_form_only", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Your unique username")
            password = st.text_input("Password", type="password", placeholder="Your password")
            submit   = st.form_submit_button("Sign in")

        if submit:
            if username.strip() and password.strip():
                # Save login state
                st.session_state.auth = {"user": username.strip()}

                # Append to CSV log
                row = pd.DataFrame([{
                    "user": username.strip(),
                    "time": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }])
                if LOGIN_LOG.exists():
                    df = pd.read_csv(LOGIN_LOG)
                    df = pd.concat([df, row], ignore_index=True)
                else:
                    df = row
                df.to_csv(LOGIN_LOG, index=False)

                st.rerun()
            else:
                st.error(" Please enter both username and password.")

        st.markdown('</div>', unsafe_allow_html=True)

    return False

#--------
# New login with account creation (uses login_log.csv as user store)
def login_ui_accounts() -> bool:
    if "auth" not in st.session_state:
        st.session_state.auth = {"user": None}

    if st.session_state.auth["user"]:
        left, right = st.columns([3, 1])
        left.success(f"Logged in as **{st.session_state.auth['user']}**")
        if right.button("Logout"):
            st.session_state.auth = {"user": None}
            st.rerun()
        return True

    # Hide the sidebar on the login screen
    st.markdown("<style>[data-testid='stSidebar']{display:none !important}</style>", unsafe_allow_html=True)

    _, body, _ = st.columns([0.5, 2.0, 0.5])
    with body:
        st.markdown('<div class="auth-title">Welcome</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Login or create an account to continue.</div>', unsafe_allow_html=True)
        #st.markdown('<div class="auth-panel">', unsafe_allow_html=True)

        tab_login, tab_create = st.tabs(["Login", "Create account"])

        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="Your unique username")
                password = st.text_input("Password", type="password", placeholder="Your password")
                submit   = st.form_submit_button("Sign in")
            if submit:
                if username.strip() and password.strip():
                    if _verify_user(username.strip(), password):
                        st.session_state.auth = {"user": username.strip()}
                        st.rerun()
                    else:
                        st.error("No registered user. Please create an account.")
                else:
                    st.error("Please enter both username and password.")

        with tab_create:
            with st.form("create_form", clear_on_submit=True):
                new_user = st.text_input("New username")
                new_pw   = st.text_input("New password", type="password")
                new_pw2  = st.text_input("Confirm password", type="password")
                create   = st.form_submit_button("Create account")
            if create:
                if not new_user.strip() or not new_pw:
                    st.error("Username and password are required.")
                elif new_pw != new_pw2:
                    st.error("Passwords do not match.")
                elif _user_exists(new_user.strip()):
                    st.error("Username already exists. Choose another.")
                else:
                    if _register_user(new_user.strip(), new_pw):
                        st.success("Account created. You can now log in.")
                    else:
                        st.error("Could not create account. Try again.")

        st.markdown('</div>', unsafe_allow_html=True)

    return False


# ---- Gate the rest of the app (everything below only shows after login) ----
if not login_ui_accounts():
    st.stop()

# =========================
# Login History (optional)
# =========================
if st.session_state.auth["user"]:
    st.subheader("Login History")
    if LOGIN_LOG.exists():
        df = pd.read_csv(LOGIN_LOG)
        keep = [c for c in ["user", "time"] if c in df.columns]
        st.dataframe(df[keep].tail(10), use_container_width=True)
    else:
        st.info("No login history yet.")

# =========================
# Helpers (batch logging/reset) - defined early for sidebar use
# =========================
def ensure_log():
    """Initialize the batch log if needed."""
    if "summary_log" not in st.session_state:
        st.session_state.summary_log = pd.DataFrame(
            columns=["Image ID", "Total dirt", "Dirt (large)", "Dirt (small)", "Dirt Coverage (%)"]
        )

def reset_batch():
    """Clear the batch dataframe and reset the image counter back to 1."""
    ensure_log()
    st.session_state.summary_log = st.session_state.summary_log.iloc[0:0]
    st.session_state.img_idx = 1
    st.rerun()

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    # --- Detection Controls ---
    st.header("Detection Controls")
    global_thr = st.slider("Global confidence threshold", 0.0, 1.0, 0.30, 0.01)
    enabled_classes = st.multiselect("Show classes", DEFAULT_CLASSES, default=DEFAULT_CLASSES)

    st.markdown("---")
    # --- Box Colors ---
    st.header("Box Colors")
    default_col = {"dirt_large": "#FF6B6B", "dirt_small": "#3ABFF8", "seed": "#22C55E"}
    per_class_col = {}
    for cls in DEFAULT_CLASSES:
        per_class_col[cls] = st.color_picker(cls, default_col.get(cls, "#888888"))

    st.markdown("---")
    # --- Drawing & Limits ---
    st.header("Drawing & Limits")
    show_scores = st.checkbox("Show scores on labels", value=True)
    max_dets = st.slider("Max detections to draw", 10, 1000, 300, 10)

    st.markdown("---")
    # Sidebar reset (above Model)
    if st.button("Reset Batch Log", key="reset_batch_side"):
        reset_batch()

    st.markdown("---")
    # --- Model (moved to bottom) ---
    st.header("Model")
    onnx_path = st.text_input("ONNX model (.onnx)", value=DEFAULT_ONNX)
    cfg_path  = st.text_input("Config (.yaml, optional)", value=DEFAULT_CFG)
    device_opt = st.selectbox("Execution Provider", ["CPUExecutionProvider"], index=0)



# =========================
# Helpers (UI)
# =========================
def card(title_emoji_text: str):
    st.markdown(
        f"""
        <div style="background:#D1D5DB; border-radius:12px; padding:12px 14px; margin:8px 0;">
          <h3 style="margin:0 0 6px 0; color:#22C55E; font-weight:bold;">
            {title_emoji_text}
          </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

def hex_to_rgb255(hex_color: str) -> Tuple[int, int, int]:
    """Return color as BGR for OpenCV drawing."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)

def next_image_id() -> str:
    """001, 002, ... (increments each time a detection runs)."""
    if "img_idx" not in st.session_state:
        st.session_state.img_idx = 1
    img_id = f"{st.session_state.img_idx:03d}"
    st.session_state.img_idx += 1
    return img_id



# =========================
# Read cfg (if available)
# =========================
def read_cfg(yaml_path: str) -> Dict:
    cfg = {}
    p = Path(yaml_path)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            pass
    return cfg

# defaults similar to Detectron2 Faster R-CNN
def cfg_defaults() -> Dict:
    return {
        "INPUT": {
            "FORMAT": "BGR",
            "MIN_SIZE_TEST": 800,
            "MAX_SIZE_TEST": 1333,
            "PIXEL_MEAN": [103.530, 116.280, 123.675],  # BGR
            "PIXEL_STD": [1.0, 1.0, 1.0],
        }
    }

# =========================
# ONNX runtime (cached)
# =========================
@st.cache_resource(show_spinner=True)
def load_onnx_session(path: str, provider: str = "CPUExecutionProvider") -> ort.InferenceSession:
    if not Path(path).exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")
    sess_opts = ort.SessionOptions()
    sess = ort.InferenceSession(path, sess_options=sess_opts, providers=[provider])
    return sess

# =========================
# Pre/Post-processing
# =========================
def read_image(file) -> np.ndarray:
    if isinstance(file, (str, Path)):
        img = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image at {file}")
        return img
    data = file.read()
    file.seek(0)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image.")
    return img

def apply_resize_keep_aspect(img: np.ndarray, min_size: int, max_size: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = float(min_size) / min(h, w)
    if max(h, w) * scale > max_size:
        scale = float(max_size) / max(h, w)
    new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
    if scale != 1.0:
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img
    return img_resized, scale

def preprocess_for_onnx(img_bgr: np.ndarray, cfg_all: Dict) -> Tuple[np.ndarray, Dict]:
    cfg = cfg_defaults()
    user_cfg = read_cfg(cfg_path) if cfg_path else {}
    for k1 in ("INPUT",):
        if k1 in user_cfg:
            cfg[k1].update({kk: user_cfg[k1].get(kk, cfg[k1].get(kk)) for kk in cfg[k1].keys()})

    fmt      = cfg["INPUT"]["FORMAT"]
    mean     = np.array(cfg["INPUT"]["PIXEL_MEAN"], dtype=np.float32)
    std      = np.array(cfg["INPUT"]["PIXEL_STD"], dtype=np.float32)
    min_size = int(cfg["INPUT"]["MIN_SIZE_TEST"])
    max_size = int(cfg["INPUT"]["MAX_SIZE_TEST"])

    img_resized, scale = apply_resize_keep_aspect(img_bgr, min_size, max_size)
    if fmt.upper() == "RGB":
        img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    else:
        img = img_resized  # BGR

    img = img.astype(np.float32)
    if mean.shape[0] == 3:
        img -= mean
    if std.shape[0] == 3:
        img /= std

    chw = np.transpose(img, (2, 0, 1))   # (3,H,W) float32
    inp = chw

    meta = {"scale": scale, "input_format": fmt.upper()}
    return inp, meta

def postprocess_outputs(outputs: Dict[str, np.ndarray], scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (boxes[N,4], scores[N], classes[N])."""

    def find_key(keys: List[str]) -> Optional[str]:
        for k in keys:
            if k in outputs:
                return k
        for k in outputs:
            if any(p in k for p in keys):
                return k
        return None

    boxes_key  = find_key(["boxes", "pred_boxes", "bbox", "dets"])
    scores_key = find_key(["scores", "score"])
    cls_key    = find_key(["labels", "classes", "pred_classes", "class_ids"])

    # Case 1: separate outputs
    if boxes_key and scores_key and cls_key:
        boxes      = np.asarray(outputs[boxes_key]).reshape(-1, 4).astype(np.float32)
        scores_raw = np.asarray(outputs[scores_key]).reshape(-1)
        clses_raw  = np.asarray(outputs[cls_key]).reshape(-1)

        def prob_like(x: np.ndarray) -> bool:
            if x.size == 0:
                return False
            x = x.astype(np.float32)
            frac = np.mean((x >= 0.0) & (x <= 1.0))
            return frac >= 0.8

        if not prob_like(scores_raw) and prob_like(clses_raw):
            clses  = np.rint(scores_raw).astype(np.int32)
            scores = clses_raw.astype(np.float32)
        else:
            scores = scores_raw.astype(np.float32)
            clses  = np.rint(clses_raw).astype(np.int32)

        if scale != 0:
            boxes = boxes / scale
        return boxes, scores, clses

    # Case 2: single Nx6 tensor
    for name, val in outputs.items():
        arr = np.asarray(val)
        if arr.ndim == 2 and arr.shape[1] == 6:
            arr = arr.astype(np.float32)
            boxes = arr[:, :4]
            col4, col5 = arr[:, 4], arr[:, 5]

            def prob_like(x: np.ndarray) -> bool:
                if x.size == 0:
                    return False
                within = np.mean((x >= 0.0) & (x <= 1.0))
                return within >= 0.8

            if prob_like(col4) and not prob_like(col5):
                scores = col4.astype(np.float32)
                clses  = col5.astype(np.int32)
            elif prob_like(col5) and not prob_like(col4):
                clses  = col4.astype(np.int32)
                scores = col5.astype(np.float32)
            else:
                clses  = col4.astype(np.int32)
                scores = col5.astype(np.float32)

            if scale != 0:
                boxes = boxes / scale
            return boxes, scores, clses

    raise RuntimeError(f"Unexpected ONNX outputs: {list(outputs.keys())}")

def filter_by_threshold_and_class(
    boxes, scores, clses, classes: List[str], enabled: List[str], thr: float
):
    keep = []
    for i, (sc, cid) in enumerate(zip(scores, clses)):
        orig = classes[cid] if 0 <= cid < len(classes) else str(cid)
        disp = LABEL_REMAP.get(orig, orig)
        if (disp in enabled) and (sc >= thr):
            keep.append(i)
    if len(keep) == 0:
        return (
            np.empty((0, 4), np.float32),
            np.empty((0,), np.float32),
            np.empty((0,), np.int32),
        )
    return boxes[keep], scores[keep], clses[keep]

def draw_detections(
    img_bgr: np.ndarray,
    boxes,
    scores,
    clses,
    classes: List[str],
    color_map: Dict[str, str],
    show_scores: bool,
    max_draw: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    vis = img_bgr.copy()
    n = min(len(boxes), max_draw)
    counts: Dict[str, int] = {}

    for i in range(n):
        x1, y1, x2, y2 = boxes[i]
        cls_id = int(clses[i])
        orig = classes[cls_id] if 0 <= cls_id < len(classes) else f"id:{cls_id}"
        name = LABEL_REMAP.get(orig, orig)
        counts[name] = counts.get(name, 0) + 1

        color = hex_to_rgb255(color_map.get(name, color_map.get(orig, "#888888")))
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        label = f"{name} {scores[i]:.2f}" if show_scores else name
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (int(x1), int(y1) - th - 6), (int(x1) + tw + 4, int(y1)), color, -1)
        cv2.putText(vis, label, (int(x1) + 2, int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return vis, counts

def instances_to_dataframe(boxes, scores, clses, class_names: List[str]) -> pd.DataFrame:
    if len(boxes) == 0:
        return pd.DataFrame(
            columns=["class_id", "class_name", "display_name", "score",
                     "x1", "y1", "x2", "y2", "width", "height", "area"]
        )
    rows = []
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, clses):
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        name = class_names[c] if 0 <= c < len(class_names) else str(c)
        disp = LABEL_REMAP.get(name, name)
        rows.append({
            "class_id": int(c),
            "class_name": name,
            "display_name": disp,
            "score": float(s),
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "width": int(w), "height": int(h), "area": float(w * h),
        })
    return pd.DataFrame(rows)

def cleanliness_percent_from_boxes(boxes: np.ndarray, w: int, h: int) -> float:
    if boxes is None or len(boxes) == 0:
        return 100.0
    area_sum = 0.0
    for (x1, y1, x2, y2) in boxes:
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area_sum += bw * bh
    clean_pct = 100.0 * (1.0 - area_sum / (w * h + 1e-9))
    return float(max(0.0, min(100.0, clean_pct)))

# -------- image guard (faces + size) --------
def load_face_detector():
    p = Path(FACE_CASCADE_PATH)
    if p.exists():
        return cv2.CascadeClassifier(str(p))
    return None

face_cascade = load_face_detector()

def reject_non_durian(img_bgr: np.ndarray) -> Tuple[bool, str]:
    h, w = img_bgr.shape[:2]
    if min(h, w) < 224:
        return True, "Image too small. Minimum side must be ≥224px."
    if face_cascade is not None:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        if len(faces) > 0:
            return True, "Human face detected. Please upload a durian paste image."
    return False, ""


# -------- PDF (UPDATED to Dirt Coverage) --------
def save_pdf_report(
    pdf_path: str,
    orig_bgr: np.ndarray,
    vis_bgr: np.ndarray,
    counts: Dict[str, int],
    dirt_coverage_pct: float,
    thr: float,
    user: str,
    image_name: Optional[str] = None,
):
    """
    Create a 1-page PDF report with original & annotated images, per-class counts, and dirt coverage. No 'role' field.
    """
    # Convert to RGB for ReportLab
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    vis_rgb  = cv2.cvtColor(vis_bgr,  cv2.COLOR_BGR2RGB)

    # Canvas
    c = canvas.Canvas(pdf_path, pagesize=A4)
    W, H = A4
    margin = 30

    # Header
    title = "Durian Paste Dirt Detector Report"
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, H - 40, title)
    c.setFont("Helvetica", 10)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hdr_line_2 = f"Generated: {ts} | Global threshold: {thr:.2f}"
    if image_name:
        hdr_line_2 += f" | Image: {image_name}"
    c.drawString(margin, H - 55, hdr_line_2)

    # Summary (counts + coverage)
    y = H - 85
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Summary")
    y -= 16
    c.setFont("Helvetica", 11)

    dl = int(counts.get("dirt_large", 0))
    ds = int(counts.get("dirt_small", 0))
    total = dl + ds
    lines = [
        f"- Total dirt: {total}",
        f"- dirt_large: {dl}",
        f"- dirt_small: {ds}",
        f"- Dirt Coverage: {dirt_coverage_pct:.2f}%",
    ]
    if total == 0:
        lines.append("- Status: CLEAN (no detections retained)")

    for line in lines:
        c.drawString(margin + 10, y, line)
        y -= 14

    # Helper to draw images with aspect fit
    def draw_img(img_rgb, x, y, max_w, max_h):
        # Ensure uint8
        if img_rgb.dtype != np.uint8:
            img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

        h0, w0 = img_rgb.shape[:2]
        scale = min(max_w / max(1, w0), max_h / max(1, h0))
        draw_w = max(1, int(w0 * scale))
        draw_h = max(1, int(h0 * scale))

        # Convert NumPy -> PIL -> in-memory PNG buffer 
        pil_img = Image.fromarray(img_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)

        c.drawImage(ImageReader(buf), x, y, width=draw_w, height=draw_h)

    # Images side-by-side
    left_x  = margin
    right_x = W / 2 + 10
    img_y   = H / 2.2    
    max_h   = H / 3  
    max_w   = W / 2 - (margin + 25)

    c.setFont("Helvetica-Bold", 11)
    c.drawString(left_x,  img_y + max_h + 5, "Original")
    c.drawString(right_x, img_y + max_h + 5, "Detections")

    draw_img(orig_rgb, left_x,  img_y, max_w, max_h)
    draw_img(vis_rgb,  right_x, img_y, max_w, max_h)

    # Footer
    c.setFont("Helvetica-Oblique", 8)
    c.drawRightString(W - margin, 20, f"User: {user}")
    c.showPage()
    c.save()

# =========================
# Load ONNX session
# =========================
try:
    sess = load_onnx_session(onnx_path, device_opt)
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# Read cfg and classes
cfg_all = read_cfg(cfg_path)
classes = DEFAULT_CLASSES.copy()

# Inputs/outputs info
input_name   = sess.get_inputs()[0].name
output_info  = sess.get_outputs()
output_names = [o.name for o in output_info]


# =========================
# Main UI
# =========================
st.markdown("### Upload image")
uploaded = st.file_uploader("PNG/JPG up to ~20MB", type=["png", "jpg", "jpeg"])

if uploaded:
    img_bgr = read_image(uploaded)

    # guard
    blocked, reason = reject_non_durian(img_bgr)
    if blocked:
        st.error(reason)
        st.stop()

    h, w = img_bgr.shape[:2]
    col1, col2 = st.columns(2)

    with col1:
        card(" Original")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    run_infer = st.button(" Detect Dirt")

    if run_infer:
        with st.spinner("Running inference..."):
            t0 = time.time()
            inp, meta = preprocess_for_onnx(img_bgr, cfg_all)

            # If your model expects batch, do: inp = np.expand_dims(inp, 0)
            out_vals = sess.run(None, {input_name: inp})
            out_map  = {out.name.lower(): val for out, val in zip(output_info, out_vals)}
            boxes, scores, clses = postprocess_outputs(out_map, scale=meta.get("scale", 1.0))

            latency_ms = (time.time() - t0) * 1000.0

            # Debug info
            try:
                _u = np.unique(clses)
                _smin = float(np.min(scores)) if scores is not None and len(scores) else 0.0
                _smax = float(np.max(scores)) if scores is not None and len(scores) else 0.0
                st.caption(f"Debug: class ids unique={_u.tolist()} | score_range=({_smin:.2f},{_smax:.2f})")
            except Exception:
                pass

        # filter & draw
        boxes_f, scores_f, clses_f = filter_by_threshold_and_class(
            boxes, scores, clses, classes, enabled_classes, global_thr
        )

        # ---------- Compute Area-based Dirt Coverage ----------
        df = instances_to_dataframe(boxes_f, scores_f, clses_f, classes)
        is_dirt = df["display_name"].isin(["dirt_small", "dirt_large"])
        dirt_area = float(df.loc[is_dirt, "area"].sum())
        img_area = float(w * h)
        dirt_coverage_pct = (dirt_area / img_area * 100.0) if img_area > 0 else 0.0
        # ------------------------------------------------------

        
        dirt_count = int(is_dirt.sum())
        if dirt_count <= CLEAN_MAX_COUNT:
            # zero-out detections so the whole app shows 0 (image, counts, tables, PDF)
            boxes_f = boxes_f[:0]
            scores_f = scores_f[:0]
            clses_f = clses_f[:0]
            dirt_coverage_pct = 0.0
            df = df.iloc[0:0]  # keep df consistent too

         # --- Collect kept detection scores for histogram (after filters + CLEAN rule) ---
        if "hist_scores_kept" not in st.session_state:
            st.session_state["hist_scores_kept"] = []

        st.session_state["hist_scores_kept"].extend([float(s) for s in np.asarray(scores_f).ravel()])

        # continue with drawing
        vis_bgr, counts = draw_detections(
            img_bgr, boxes_f, scores_f, clses_f, classes, per_class_col, show_scores, max_dets
        )   

        with col2:
            card("Detections")
            st.image(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB), use_container_width=True,
                    caption=f"Latency: {latency_ms:.1f} ms | Provider: {device_opt}")

            # Metrics (counts + area-based coverage)
            c_large = int(counts.get('dirt_large', 0))
            c_small = int(counts.get('dirt_small', 0))
            m1, m2, m3 = st.columns(3)
            m1.metric("dirt_large", c_large)
            m2.metric("dirt_small", c_small)
            m3.metric("Dirt Coverage (%)", f"{dirt_coverage_pct:.2f}")

            out_name = f"{Path(uploaded.name).stem}_pred.png"
            cv2.imwrite(out_name, vis_bgr)

        # ===== Tabs =====
        tabs = ["Summary", "Analytics", "Report"]
        tab_objs = st.tabs(tabs)

        # --- Summary ---
        with tab_objs[0]:
            card("Detection Summary")

            # Prepare per-image summary row
            image_id = next_image_id()
            total_dirt = int(counts.get('dirt_large', 0)) + int(counts.get('dirt_small', 0))
            summary_row = {
                "Image ID": image_id,
                "Total dirt": total_dirt,
                "Dirt (large)": int(counts.get('dirt_large', 0)),
                "Dirt (small)": int(counts.get('dirt_small', 0)),
                "Dirt Coverage (%)": round(dirt_coverage_pct, 2),
            }

            # 1) Per-image summary (top)
            st.markdown("**Dirt Coverage Summary (per image)**")
            st.dataframe(pd.DataFrame([summary_row]), use_container_width=True)

            # 2) Batch Log (accumulates all images)
            ensure_log()
            st.session_state.summary_log = pd.concat(
                [st.session_state.summary_log, pd.DataFrame([summary_row])],
                ignore_index=True
            )
            st.markdown("**Batch Log (all images)**")
            st.dataframe(st.session_state.summary_log, use_container_width=True)

            # Downloads / maintenance
            csv_bytes = st.session_state.summary_log.to_csv(index=False).encode("utf-8")
            st.download_button(" Download log as CSV", data=csv_bytes,
                               file_name="dirt_detection_log.csv", mime="text/csv")
            # Reset moved to sidebar only
            # --- Per-image mini chart: Dirt count vs Coverage (dual-axis)
            st.markdown("**Current image — Dirt breakdown & coverage**")

            # counts from your app state
            c_large = int(counts.get('dirt_large', 0))
            c_small = int(counts.get('dirt_small', 0))
            cov = float(dirt_coverage_pct)  # 0..100
            clean_pct = max(0.0, 100.0 - cov)

            col_left, col_right = st.columns([3, 2])

            # ── Left: bar chart for dirt counts
            with col_left:
                labels = ["dirt_large", "dirt_small"]
                values = [c_large, c_small]
                fig_cur = px.bar(
                    x=labels, y=values, text=[str(v) for v in values],
                    labels={"x": "Class", "y": "Count"}
                )
                fig_cur.update_traces(textposition="outside")
                _style_fig(fig_cur, "Current Image — Dirt Breakdown")
                fig_cur.add_annotation(
                    text=f"Dirt Coverage: {cov:.2f}%",
                    showarrow=False, xref="paper", yref="paper", x=0.5, y=-0.22,
                    font=dict(color="gray")
                )
                st.plotly_chart(fig_cur, use_container_width=True)

            # ── Right: donut showing coverage vs clean
            with col_right:
                df_cov = pd.DataFrame({
                    "type": ["Dirt area", "Clean area"],
                    "pct":  [cov, clean_pct]
                })
                fig_cov = px.pie(
                    df_cov, names="type", values="pct", hole=0.55
                )
                fig_cov.update_traces(textinfo="label+percent", pull=[0.08, 0])
                _style_fig(fig_cov, "Area Composition")
                st.plotly_chart(fig_cov, use_container_width=True)

            # Optional: quick KPI row under the charts
            k1, k2, k3 = st.columns(3)
            k1.metric("dirt_large", c_large)
            k2.metric("dirt_small", c_small)
            k3.metric("Dirt Coverage (%)", f"{cov:.2f}")

            

        # --- Analytics ---
        with tab_objs[1]:
            card("Visual Analytics")

            # ========== A) Batch totals (all processed images) ==========
            st.markdown("**Batch Dirt Counts (all images)**")
            log = st.session_state.get("summary_log")
            if log is not None and not log.empty:
                totals = {
                    "dirt_large": log["Dirt (large)"].sum(),
                    "dirt_small": log["Dirt (small)"].sum()
                }
                df_totals = pd.DataFrame({"class": list(totals.keys()), "count": list(totals.values())})

                fig_bar_batch = px.bar(
                    df_totals, x="class", y="count", text="count",
                    labels={"class": "Class", "count": "Total Count"}
                )
                fig_bar_batch.update_traces(textposition="outside")
                _style_fig(fig_bar_batch, "Batch Dirt Counts (all images)")
                st.plotly_chart(fig_bar_batch, use_container_width=True)
            else:
                st.info("No batch data yet. Upload a few images to see totals.")

            # ========== B) Current image distribution ==========
            if counts:
                st.markdown("**Pie chart (all images)**")

                log = st.session_state.get("summary_log")
                if log is not None and not log.empty:
                    totals = {
                        "dirt_large": pd.to_numeric(log["Dirt (large)"], errors="coerce").fillna(0).sum(),
                        "dirt_small": pd.to_numeric(log["Dirt (small)"], errors="coerce").fillna(0).sum(),
                    }
                    df_totals = pd.DataFrame({
                        "class": list(totals.keys()),
                        "count": list(totals.values())
                    })

                    fig_pie_batch = px.pie(
                        df_totals,
                        names="class",
                        values="count",
                        hole=0.6
                    )
                    fig_pie_batch.update_traces(
                        textposition="inside",
                        texttemplate="%{label}\n%{percent:.1%} (n=%{value})",
                        hovertemplate="%{label}: %{value} detections<br>%{percent}",
                        pull=[0.06 if c == df_totals['class'].iloc[0] else 0 for c in df_totals['class']]
                    )
                    fig_pie_batch.update_layout(
                        showlegend=False,
                        uniformtext_minsize=12,
                        uniformtext_mode="hide",
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    # Start at top for consistency
                    fig_pie_batch.update_traces(rotation=90)

                    # Optional: match your sidebar colors (if keys exist)
                    color_map = {
                        "dirt_large": per_class_col.get("dirt_large", "#FF6B6B"),
                        "dirt_small": per_class_col.get("dirt_small", "#3ABFF8"),
                    }
                    fig_pie_batch.update_traces(marker=dict(colors=[color_map[c] for c in df_totals["class"]]))

                    # Add total in the center
                    total_n = int(df_totals["count"].sum())
                    fig_pie_batch.add_annotation(
                        text=f"Total\n{total_n}",
                        showarrow=False,
                        font=dict(size=16),
                        x=0.5, y=0.5
                    )

                    _style_fig(fig_pie_batch, "Dirt Distribution (All Images)")
                    st.plotly_chart(fig_pie_batch, use_container_width=True)
                else:
                    st.info("No batch data yet. Upload a few images to see totals.")

            # ========== C) Confidence Histogram (percent + threshold line) ==========
            st.markdown("**Confidence Score Histogram (Kept detections)**")

            scores_for_plot = st.session_state.get("hist_scores_kept", [])

            if scores_for_plot and len(scores_for_plot):
                df_scores = pd.DataFrame({"score": scores_for_plot})
                fig_hist = px.histogram(
                    df_scores,
                    x="score",
                    nbins=12,
                    histnorm="percent",
                    range_x=[0.2, 1.0],   # focus on 0.2–1.0 for clarity
                )
                fig_hist.update_traces(
                    hovertemplate="Confidence: %{x:.2f}<br>Percentage: %{y:.1f}%",
                    texttemplate="%{y:.1f}%",
                    textposition="outside",
                    cliponaxis=False,
                )
                fig_hist.update_xaxes(title="Confidence score", range=[0.2, 1.0])
                fig_hist.update_yaxes(title="Percentage (%)", ticksuffix="%")
                fig_hist.add_vline(
                    x=float(global_thr),
                    line_dash="dash",
                    line_width=2,
                    opacity=0.9,
                    annotation_text=f"Threshold {global_thr:.2f}",
                    annotation_position="top",
                )
                _style_fig(fig_hist, "Detection Confidence (Kept Detections Only)")
                st.plotly_chart(fig_hist, use_container_width=True)

                # KPIs under the chart
                arr = np.asarray(scores_for_plot, dtype=float)
                thr = float(global_thr)
                total = arr.size
                below = int((arr < thr).sum())
                kept  = total - below
                st.caption(
                    f"Samples: {total} | < threshold: {below} | ≥ threshold: {kept} ({(kept/total*100):.1f}%)"
                )
            else:
                st.info("No scores to plot yet — upload and run detection.")

            # ========== D) Scatter: Dirt Coverage vs Total Dirt (batch) ==========
            st.markdown("**Dirt Coverage vs Total Dirt (all images)**")
            needed = {"Total dirt", "Dirt Coverage (%)"}
            if log is not None and not log.empty and needed.issubset(log.columns):
                df_log = log.copy()
                df_log["Total dirt"] = pd.to_numeric(df_log["Total dirt"], errors="coerce")
                df_log["Dirt Coverage (%)"] = pd.to_numeric(df_log["Dirt Coverage (%)"], errors="coerce")
                df_log = df_log.dropna(subset=["Total dirt", "Dirt Coverage (%)"])

                if len(df_log) >= 1:
                    trend = "ols" if len(df_log) >= 3 else None
                    fig_sc = px.scatter(
                        df_log, x="Total dirt", y="Dirt Coverage (%)", trendline=trend
                    )
                    _style_fig(fig_sc, "Relationship across processed images")
                    st.plotly_chart(fig_sc, use_container_width=True)

                    if len(df_log) >= 2:
                        x = df_log["Total dirt"].to_numpy(dtype=float)
                        y = df_log["Dirt Coverage (%)"].to_numpy(dtype=float)
                        r = float(np.corrcoef(x, y)[0, 1])
                        st.caption(f"Correlation (r) between count and coverage: {r:.2f}")
                    else:
                        st.caption("Only 1 point available — correlation needs at least 2 points.")
                else:
                    st.info("No valid rows after cleaning the log.")
            else:
                st.info("No logged images yet — upload a few to see this chart.")

        # --- Report (pass Dirt Coverage %) ---
        with tab_objs[2]:
            card("Download Report (PDF)")
            pdf_name = f"{Path(uploaded.name).stem}_report.pdf"
            save_pdf_report(
                pdf_name,
                img_bgr, vis_bgr, counts,
                dirt_coverage_pct,
                global_thr,
                user=st.session_state.auth["user"],
                image_name=uploaded.name
            )
            with open(pdf_name, "rb") as f:
                st.download_button("Download PDF report", data=f,
                                   file_name=pdf_name, mime="application/pdf")

else:
    st.info("Upload an image to begin.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div class='app-footer'>© 2025 Shermann Choong · Asia Pacific University</div>",
    unsafe_allow_html=True
)
