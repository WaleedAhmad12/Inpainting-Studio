from PIL import Image
import numpy as np
import gradio as gr
import cv2
from config import MASK_DIFF_THRESHOLD,MASK_DILATE_KERNEL,SD_IMAGE_SIZE,SD_MODEL_ID
import torch
from diffusers import StableDiffusionInpaintPipeline
_pipeline: StableDiffusionInpaintPipeline | None = None

CURRENT_IMAGE: Image.Image | None = None


def get_current_image():
    return CURRENT_IMAGE




def url_load_image(url):
    global CURRENT_IMAGE
    from PIL import Image
    import requests
    from io import BytesIO

    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        CURRENT_IMAGE = img
        return img
    except Exception as e:
        print(f"Error loading image from URL: {e}")
        return None


def file_load_image(file):
    global CURRENT_IMAGE
    try:
        img = file.convert("RGB")
        CURRENT_IMAGE = img
        return img
    except Exception as e:
        print(f"Error loading image from file: {e}")
        return None


def enhance_prompt(prompt: str) -> str:
    if not prompt or not prompt.strip():
        return ""
    
    prompt = prompt.strip()
    
    # Inpainting-specific enhancements
    inpainting_context = [
        "seamless blending",
        "matching scene lighting",
        "correct perspective",
        "consistent style",
        "high quality details"
    ]
    
    # Quality and realism enhancements
    quality_keywords = [
        "photorealistic",
        "highly detailed",
        "professional quality",
        "natural lighting",
        "sharp focus"
    ]
    
    # Build the enhanced prompt
    enhanced = ", ".join(quality_keywords) + ", " + prompt
    enhanced = enhanced + ", " + ", ".join(inpainting_context)
    
    return enhanced


def pil_to_numpy(img):
    
    if img is None:
        return None

    # If it's already a PIL Image, convert to numpy
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"), dtype=np.uint8)

    # If it's already a numpy array, make sure it's uint8 RGB
    if isinstance(img, np.ndarray):
        arr = img.astype(np.uint8)

        # Grayscale (2D) → add colour channels
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

        # RGBA (4 channels) → drop the alpha channel
        if arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)

        # Already RGB
        return arr

    return None 



def draw_red_overlay(original_np: np.ndarray, mask_np: np.ndarray) -> Image.Image:
    from config import MASK_PREVIEW_RED, MASK_PREVIEW_ALPHA

    preview = original_np.copy().astype(np.float32)
    mask_bool = mask_np.astype(bool)

    red = np.array(MASK_PREVIEW_RED, dtype=np.float32)
    preview[mask_bool] = (
        preview[mask_bool] * MASK_PREVIEW_ALPHA +
        red * (1 - MASK_PREVIEW_ALPHA)
    )

    preview = np.clip(preview, 0, 255).astype(np.uint8)
    return Image.fromarray(preview)


def cleanup_mask(mask: np.ndarray, min_area: int = 40) -> np.ndarray:
    if mask is None:
        return mask

    cleaned = mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    filtered = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered[labels == i] = 255

    return filtered



def extract_mask_from_drawing(drawn_img, original_img):
    # Safety checks
    if drawn_img is None or original_img is None:
        return None, None

    # Gradio's ImageEditor gives us a dict. The actual image is under "composite".
    if isinstance(drawn_img, dict):
        composite = drawn_img.get("composite")
    else:
        composite = drawn_img   # Sometimes it's already an image

    if composite is None:
        return None, None

    # Convert both images to numpy arrays so OpenCV can work with them
    composite_np = pil_to_numpy(composite)
    original_np  = pil_to_numpy(original_img)

    if composite_np is None or original_np is None:
        return None, None

    # If sizes don't match (can happen with some uploads), resize composite to match
    if composite_np.shape[:2] != original_np.shape[:2]:
        h, w = original_np.shape[:2]
        composite_np = cv2.resize(composite_np, (w, h), interpolation=cv2.INTER_NEAREST)

    # ── Step 1: Find where pixels changed ────────────────────
    # absdiff gives us the absolute difference between each pixel
    # Where the user painted, pixels will be very different from the original
    diff = cv2.absdiff(composite_np, original_np)

    # Convert the 3-channel diff to single-channel (grayscale)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    # Any pixel that changed more than THRESHOLD becomes white in the mask
    _, mask = cv2.threshold(diff_gray, MASK_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # ── Step 2: Expand mask edges slightly ───────────────────
    # "Dilate" = grow the mask outward by a few pixels.
    # This makes sure we cover the full stroke, not just the center.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MASK_DILATE_KERNEL, MASK_DILATE_KERNEL)
    )
    mask = cv2.dilate(mask, kernel, iterations=1)

    # ── Return both the preview and the clean B&W mask ───────
    red_preview = draw_red_overlay(original_np, mask)
    bw_mask     = Image.fromarray(mask)

    return red_preview, bw_mask


def normalize_mask_input(mask_like, original_img):
    if mask_like is None:
        return None

    if isinstance(mask_like, dict):
        _preview, bw_mask = extract_mask_from_drawing(mask_like, original_img)
        return bw_mask

    if isinstance(mask_like, Image.Image):
        return mask_like.convert("L")

    if isinstance(mask_like, np.ndarray):
        if mask_like.ndim == 3:
            return Image.fromarray(mask_like).convert("L")
        return Image.fromarray(mask_like)

    return None



def get_sd_pipeline():
    global _pipeline

    if _pipeline is None:
        # Detect if a GPU is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32

        print(f"[SD] Loading {SD_MODEL_ID} on {device} …")
        print("[SD] This may take a few minutes on first run (downloading model).")

        _pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=dtype,
        ).to(device)

        # enable_attention_slicing uses less VRAM on GPU (slightly slower but prevents OOM errors)
        _pipeline.enable_attention_slicing()

        print("[SD] Model ready.")

    return _pipeline




def resize_for_sd(image: Image.Image, mask: Image.Image, size: int = 512):
    image_resized = image.convert("RGB").resize((size, size), Image.LANCZOS)
    mask_resized  = mask.convert("L").resize((size, size), Image.NEAREST)
    return image_resized, mask_resized




def _run_stable_diffusion(
    original_img: Image.Image,
    bw_mask:      Image.Image,
    prompt:       str,
    steps:        int,
    guidance:     float,
    strength:     float,
) -> Image.Image | None:

    # Load the SD pipeline (from cache if already loaded)
    pipeline = get_sd_pipeline()

    # Resize image and mask to 512×512 (SD's required input size)
    image_512, mask_512 = resize_for_sd(
        original_img if isinstance(original_img, Image.Image) else Image.fromarray(original_img),
        bw_mask      if isinstance(bw_mask,      Image.Image) else Image.fromarray(bw_mask),
        size=SD_IMAGE_SIZE,
    )

    # Run Stable Diffusion!
    # The pipeline returns an object with an .images list; we take the first result.
    output = pipeline(
        prompt             = prompt,
        image              = image_512,
        mask_image         = mask_512,
        num_inference_steps= int(steps),
        guidance_scale     = float(guidance),
        strength           = float(strength),
    )

    result_image = output.images[0]
    return result_image





def inpainting(bw_mask, prompt, steps, guidance, strength):

    if not prompt or not prompt.strip():
        gr.Warning("Please enter a prompt describing what to put in the masked area.")
        return None

    original_img = get_current_image()

    bw_mask = normalize_mask_input(bw_mask, original_img)

    if bw_mask is None:
        gr.Warning("Please create or generate a mask first.")
        return None


    return _run_stable_diffusion(original_img, bw_mask, prompt, steps, guidance, strength)




def get_mask_and_inpaint(method, manual, auto, prompt, steps, guidance, strength):
    mask = manual if method == "Manual Masking" else auto
    return inpainting(mask, prompt, steps, guidance, strength)
