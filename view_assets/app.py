"""Streamlit app to browse and view images, audio, and videos from pipeline S3 output buckets."""

import base64
import glob
import io
import yaml
import boto3
import streamlit as st

CONFIG_DIR = "config"

# Read account/region from .env or environment
from dotenv import load_dotenv
import os
load_dotenv()
ACCOUNT = os.environ.get("AWS_ACCOUNT_ID")
REGION = os.environ.get("REGION", "us-east-1")


def _shared_prefix() -> str:
    """Read shared_prefix from cicd config."""
    try:
        with open(f"{CONFIG_DIR}/cicd/cicd.yaml") as f:
            return yaml.safe_load(f).get("shared_prefix", "")
    except Exception:
        return ""


SHARED_PREFIX = _shared_prefix()

# Supported media types and their extensions
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
AUDIO_EXTENSIONS = (".mp3", ".wav", ".ogg", ".flac", ".aac")
ALL_EXTENSIONS = VIDEO_EXTENSIONS + IMAGE_EXTENSIONS + AUDIO_EXTENSIONS

s3 = boto3.client("s3", region_name=REGION)


def detect_media_type(key: str) -> str:
    lower = key.lower()
    if lower.endswith(VIDEO_EXTENSIONS):
        return "video"
    if lower.endswith(IMAGE_EXTENSIONS):
        return "image"
    if lower.endswith(AUDIO_EXTENSIONS):
        return "audio"
    return "unknown"


def list_construct_ids() -> list[str]:
    """Read construct_id from each config YAML file."""
    ids = []
    for path in sorted(glob.glob(f"{CONFIG_DIR}/pipeline/config*.yaml")):
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            if cfg and "construct_id" in cfg:
                ids.append(cfg["construct_id"])
        except Exception:
            continue
    return sorted(set(ids))


@st.cache_data(ttl=60)
def list_buckets(construct_id: str) -> list[str]:
    """List output buckets for a given construct ID."""
    prefix = f"{ACCOUNT}-{REGION}-{SHARED_PREFIX}{construct_id}-"
    resp = s3.list_buckets()
    return sorted(
        b["Name"] for b in resp["Buckets"]
        if b["Name"].startswith(prefix) and b["Name"].endswith("-output-bucket")
    )


@st.cache_data(ttl=60)
def list_assets(bucket: str, media_filter: str) -> list[str]:
    """List all matching media keys in a bucket."""
    if media_filter == "video":
        exts = VIDEO_EXTENSIONS
    elif media_filter == "image":
        exts = IMAGE_EXTENSIONS
    elif media_filter == "audio":
        exts = AUDIO_EXTENSIONS
    else:
        exts = ALL_EXTENSIONS

    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(exts):
                keys.append(obj["Key"])
    return sorted(keys)


@st.cache_data(ttl=300)
def download_asset(bucket: str, key: str) -> bytes:
    """Download an asset from S3 into memory."""
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    return buf.getvalue()


def render_video(data: bytes, key: str):
    ext = key.rsplit(".", 1)[-1].lower()
    mime = {"mp4": "video/mp4", "webm": "video/webm", "mkv": "video/x-matroska"}.get(ext, "video/mp4")
    b64 = base64.b64encode(data).decode()
    st.components.v1.html(
        f'<video controls style="max-width:100%">'
        f'<source src="data:{mime};base64,{b64}" type="{mime}">'
        f"</video>",
        height=720,
    )


def render_image(data: bytes, key: str):
    st.image(data, use_container_width=True)


def render_audio(data: bytes, key: str):
    ext = key.rsplit(".", 1)[-1].lower()
    mime = {"mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg", "flac": "audio/flac", "aac": "audio/aac"}.get(ext, "audio/mpeg")
    st.audio(data, format=mime)


# --- UI ---

st.set_page_config(page_title="Asset Viewer", layout="wide")
st.title("🎨 Pipeline Asset Viewer")

# Sidebar: construct ID and bucket selection
construct_ids = list_construct_ids()
if not construct_ids:
    st.error("No output buckets found.")
    st.stop()

selected_construct = st.sidebar.selectbox("Construct ID", construct_ids)

buckets = list_buckets(selected_construct)
if not buckets:
    st.error("No output buckets found for this construct.")
    st.stop()

step_prefix = f"{ACCOUNT}-{REGION}-{SHARED_PREFIX}{selected_construct}-"
selected_bucket = st.sidebar.selectbox(
    "Step", buckets,
    format_func=lambda b: b[len(step_prefix):-len("-output-bucket")]
)

# List all assets first to extract execution IDs
all_assets = list_assets(selected_bucket, "all")

# Execution ID filter (right below step)
exec_ids = sorted({k.split("/")[0] for k in all_assets if "/" in k})
selected_exec = st.sidebar.selectbox("Execution ID", ["All"] + exec_ids)

# Media type filter
media_filter = st.sidebar.radio("Media type", ["all", "video", "image", "audio"])

# Apply filters
assets = list_assets(selected_bucket, media_filter)

if selected_exec != "All":
    assets = [a for a in assets if a.startswith(f"{selected_exec}/")]

if not assets:
    st.info(f"No {media_filter} assets found.")
    st.stop()

st.sidebar.markdown(f"**{len(assets)}** assets found")

# Track current index
if "asset_idx" not in st.session_state:
    st.session_state.asset_idx = 0
st.session_state.asset_idx = max(0, min(st.session_state.asset_idx, len(assets) - 1))
idx = st.session_state.asset_idx

# Dropdown
selected_key = st.selectbox("Asset", assets, index=idx, format_func=lambda k: k.split("/")[-1])
if assets.index(selected_key) != idx:
    st.session_state.asset_idx = assets.index(selected_key)
    st.rerun()

# Prev / counter / Next
col_prev, col_counter, col_next = st.columns([1, 2, 1])
with col_prev:
    if st.button("⬅ Prev", disabled=idx == 0):
        st.session_state.asset_idx -= 1
        st.rerun()
with col_counter:
    st.markdown(f"**{idx + 1} / {len(assets)}**")
with col_next:
    if st.button("Next ➡", disabled=idx == len(assets) - 1):
        st.session_state.asset_idx += 1
        st.rerun()

st.caption(f"`s3://{selected_bucket}/{selected_key}`")

# Render the asset
data = download_asset(selected_bucket, selected_key)
mtype = detect_media_type(selected_key)

if mtype == "video":
    render_video(data, selected_key)
elif mtype == "image":
    render_image(data, selected_key)
elif mtype == "audio":
    render_audio(data, selected_key)
else:
    st.warning("Unsupported media type.")
