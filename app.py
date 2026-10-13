
import io
from pathlib import Path

import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from archs import create_model
from options.options import parse


# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="DarkIR | Low-Light Image Restoration",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "options" / "test" / "LOLBlur.yml"
CHECKPOINT_PATH = (
    ROOT / "models" / "training_lolblur"
    / "best_official_format_fixed.pt"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource(show_spinner="Loading trained DarkIR model...")
def load_darkir():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration not found: {CONFIG_PATH}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    opt = parse(str(CONFIG_PATH))
    model, _, _ = create_model(opt["network"], rank=0)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "params" in checkpoint:
        weights = checkpoint["params"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        weights = checkpoint["model_state_dict"]
    else:
        weights = checkpoint

    # Handle DataParallel / DistributedDataParallel checkpoints.
    model_keys = set(model.state_dict())
    weight_keys = list(weights.keys())

    if weight_keys and all(k.startswith("module.") for k in weight_keys):
        weights = {k[len("module."):]: v for k, v in weights.items()}
    elif model_keys and all(k.startswith("module.") for k in model_keys):
        weights = {"module." + k: v for k, v in weights.items()}

    model.load_state_dict(weights, strict=True)
    model = model.to(DEVICE).eval()

    return model


# =========================================================
# IMAGE RESTORATION
# =========================================================
def restore_image(image, model):
    image = image.convert("RGB")
    original_width, original_height = image.size

    tensor = to_tensor(image).unsqueeze(0).to(DEVICE)

    # DarkIR input dimensions must be padded to multiples of 8.
    _, _, height, width = tensor.shape
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8

    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(
            tensor, (0, pad_w, 0, pad_h), mode="constant", value=0
        )

    with torch.inference_mode():
        output = model(tensor)

        if isinstance(output, (tuple, list)):
            output = output[0]

        # Remove padding without changing the original dimensions.
        output = output[..., :height, :width]
        output = output.clamp(0, 1)

    restored = to_pil_image(output.squeeze(0).cpu())

    # Preserve original dimensions.
    if restored.size != (original_width, original_height):
        restored = restored.resize(
            (original_width, original_height),
            Image.Resampling.LANCZOS,
        )

    return restored


def image_to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("🌙 DarkIR")
    st.caption("Low-Light Image Restoration")

    st.divider()

    st.markdown("### About the project")
    st.write(
        "This application demonstrates a trained DarkIR model "
        "for restoring low-light and degraded images."
    )

    st.markdown("### Model information")
    st.write("**Architecture:** DarkIR-m")
    st.write("**Parameters:** Approximately 3.32 million")
    st.write(f"**Device:** {DEVICE}")

    st.divider()

    st.markdown("### How to use")
    st.markdown(
        """
        1. Upload a low-light image.
        2. Click **Restore Image**.
        3. Compare the input and restored result.
        4. Download the restored image.
        """
    )


# =========================================================
# MAIN PAGE
# =========================================================
st.title("🌙 DarkIR: Low-Light Image Restoration")
st.markdown(
    "### Recover visibility and detail from degraded images"
)
st.write(
    "Upload a low-light or degraded photograph and process it "
    "with your trained DarkIR deep learning model."
)

st.info(
    "This is an image-restoration demonstration. Results depend "
    "on image content and the types of degradation present."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    help="Supported formats: PNG, JPG, JPEG, BMP and WEBP.",
)

if uploaded_file is not None:
    try:
        original = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Unable to open the uploaded image: {exc}")
        st.stop()

    st.markdown("#### Input image")

    input_col1, input_col2, input_col3 = st.columns(3)
    input_col1.metric("Width", f"{original.width} px")
    input_col2.metric("Height", f"{original.height} px")
    input_col3.metric("File size", f"{uploaded_file.size / 1024:.1f} KB")

    st.image(
        original,
        caption="Uploaded image",
        use_container_width=True,
    )

    if st.button("✨ Restore Image", type="primary", use_container_width=True):
        try:
            with st.spinner("DarkIR is restoring your image..."):
                model = load_darkir()
                restored = restore_image(original, model)

            st.session_state["darkir_original"] = original
            st.session_state["darkir_restored"] = restored
            st.session_state["darkir_filename"] = uploaded_file.name

            st.success("Image restoration completed!")

        except Exception as exc:
            st.error(f"Restoration failed: {exc}")
            st.exception(exc)

    if "darkir_restored" in st.session_state:
        original_result = st.session_state["darkir_original"]
        restored_result = st.session_state["darkir_restored"]

        st.divider()
        st.markdown("## Restoration Results")

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("### Original")
            st.image(
                original_result,
                caption="Before restoration",
                use_container_width=True,
            )

        with right:
            st.markdown("### DarkIR Restored")
            st.image(
                restored_result,
                caption="After restoration",
                use_container_width=True,
            )

        st.divider()

        st.markdown("### Download your result")

        download_name = (
            Path(st.session_state["darkir_filename"]).stem
            + "_darkir_restored.png"
        )

        st.download_button(
            label="⬇️ Download Restored Image (PNG)",
            data=image_to_png_bytes(restored_result),
            file_name=download_name,
            mime="image/png",
            type="primary",
            use_container_width=True,
        )

        st.caption(
            "The output is saved as PNG to preserve the restored image "
            "without introducing additional JPEG compression."
        )

else:
    st.markdown("### What this demo offers")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 🖼️ Image upload")
        st.write("Upload your own supported image formats.")

    with c2:
        st.markdown("#### 🧠 Deep learning")
        st.write("Run inference using your trained DarkIR checkpoint.")

    with c3:
        st.markdown("#### 📥 Download")
        st.write("Save the restored image as a PNG file.")

    st.markdown(
        """
        ---
        **Project:** CNN-Based Low-Light Image Restoration and Deblurring Using DarkIR

        **Research reference:** [DarkIR — CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Feijoo_DarkIR_Robust_Low-Light_Image_Restoration_CVPR_2025_paper.html)
        """
    )

st.divider()
st.caption(
    "Academic project demonstration • DarkIR • "
    "Outputs are model predictions and may not be artifact-free."
)
