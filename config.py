import os

# ── Where to save downloaded models ──────────
os.environ["HF_HOME"] = "./models/huggingface"

# ── Which models to use ───────────────────────
YOLO_MODEL_NAME = "yolo_model/yolo11n-seg.pt"

# Stable Diffusion model from HuggingFace Hub
SD_MODEL_ID = "runwayml/stable-diffusion-inpainting"


# ── Stable Diffusion slider defaults ─────────

SD_DEFAULT_STEPS    = 50     # how many diffusion steps (more = slower but better)
SD_DEFAULT_GUIDANCE = 7.5    # how closely to follow the prompt (higher = stricter)
SD_DEFAULT_STRENGTH = 0.8    # how much to change the masked area (1.0 = full replace)
SD_IMAGE_SIZE       = 512    # SD works best at 512×512 pixels



# ── Manual brush mask settings ────────────────

MASK_DIFF_THRESHOLD = 15
MASK_DILATE_KERNEL = 7



# ── YOLO overlay colours ──────────────────────
# How transparent the coloured overlay is (0.0 = full colour, 1.0 = invisible)
DETECTION_ALPHA = 0.40

# Red colour used to highlight the selected mask area
MASK_PREVIEW_RED   = (220, 30, 30)
MASK_PREVIEW_ALPHA = 0.45   # original image weight in the red preview


# ── Manual brush mask settings ────────────────
# Minimum pixel difference to count as "brush stroke" vs original image
MASK_DIFF_THRESHOLD = 15

# How many pixels to expand/grow the mask edges outward
MASK_DILATE_KERNEL = 7


# ── Stable Diffusion slider defaults ─────────
# These are the starting values shown in the UI sliders
SD_DEFAULT_STEPS    = 50     # how many diffusion steps (more = slower but better)
SD_DEFAULT_GUIDANCE = 7.5    # how closely to follow the prompt (higher = stricter)
SD_DEFAULT_STRENGTH = 0.8    # how much to change the masked area (1.0 = full replace)
SD_DEFAULT_BRUSH_SIZE = 10    # how much to change the Brush Size
SD_IMAGE_SIZE       = 512    # SD works best at 512×512 pixels






# Colours used to highlight each detected object (cycles through this list)
PALETTE = [
    (255,  56,  56), (255, 157, 151), (255, 112,  31), (255, 178,  29),
    (207, 210,  49), ( 72, 249,  10), (146, 204,  23), ( 61, 219, 134),
    ( 26, 147,  52), (  0, 212, 187), ( 44, 153, 168), (  0, 194, 255),
    ( 52,  69, 147), (100, 115, 255), (  0,  24, 236), (132,  56, 255),
    ( 82,   0, 133), (203,  56, 255), (255, 149, 200), (255,  55, 199),
]
