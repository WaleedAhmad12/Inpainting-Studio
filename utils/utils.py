from PIL import Image
import numpy as np
import gradio as gr
import cv2
import os
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


## Rule-based enhancer (previous behaviour) kept as a fallback
def _rule_enhance_prompt(prompt: str) -> str:
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


# Groq API integration with fallback to rule-based enhancer
# The function below is the public entrypoint used by the UI and pipeline.
def enhance_prompt(prompt: str) -> str:
    # Prefer environment variable for API key, fall back to hard-coded key if present
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "REDACTED_GROQ_API_KEY")

    if not prompt or not prompt.strip():
        return _rule_enhance_prompt(prompt)

    # Attempt to use Groq SDK synchronously. Any error falls back to rule-based enhancer.
    try:
        import groq

        print("[PROMPT] groq module imported; version:", getattr(groq, "__version__", "unknown"))

        # Try multiple constructor names to support different SDK versions
        client = None
        constructor_candidates = ["Client", "GroqClient", "Groq", "ClientV1"]
        for name in constructor_candidates:
            ctor = getattr(groq, name, None)
            if ctor and callable(ctor):
                try:
                    client = ctor(api_key=GROQ_API_KEY)
                    print(f"[PROMPT] Constructed Groq client using {name}")
                    break
                except Exception as e:
                    print(f"[PROMPT] Groq constructor {name} failed: {repr(e)}")

        # Try factory helpers
        if client is None:
            for fn in ("from_api_key", "connect", "create_client"):
                factory = getattr(groq, fn, None)
                if callable(factory):
                    try:
                        client = factory(GROQ_API_KEY)
                        print(f"[PROMPT] Constructed Groq client using factory {fn}")
                        break
                    except Exception as e:
                        print(f"[PROMPT] Groq factory {fn} failed: {repr(e)}")

        if client is None:
            raise RuntimeError("Unable to construct Groq client; no known constructors succeeded")

        system_prompt = """
    You are a Stable Diffusion INPAINTING prompt engineer.

    Your job is to rewrite user prompts for image inpainting (object replacement or modification), NOT full image generation.

    Rules:
    - ONLY describe the object or content that should appear in the masked area
    - DO NOT describe the full scene
    - DO NOT add background, environment, or storytelling
    - Keep it focused and concise
    - Ensure the result blends naturally with the existing image
    - Include: lighting match, perspective match, shadows, realism

    Examples:

    User: change book to laptop
    Output: a realistic modern laptop, matching the scene lighting, perspective and shadows, seamless blend, high detail

    User: make shirt red
    Output: a red colored shirt with natural fabric texture, consistent lighting and shading, realistic

    User: replace person with robot
    Output: a humanoid robot, metallic texture, matching pose, lighting and perspective, realistic, seamless integration

    Now rewrite the user prompt.
    """

        base_payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1024,
            "temperature": 0.2,
        }

        # Allow overriding the model via env var `GROQ_MODEL` for flexibility.
        env_model = os.environ.get("GROQ_MODEL")
        model_candidates = []
        if env_model:
            model_candidates.append(env_model)
        # User-requested preferred model – try this first if available
        model_candidates.append("llama-3.3-70b-versatile")
        # Try a small list of candidate model names in case one is decommissioned.
        model_candidates.extend([
            "llama3-70b-8192",
            "llama3-70b",
            "llama3-13b",
            "llama3-7b",
        ])

        method_attempts = [
            ("generate", lambda c, p: c.generate(**p)),
            ("run", lambda c, p: c.run(**p)),
            ("chat", lambda c, p: c.chat(p)),
            ("chat.create", lambda c, p: getattr(c, "chat").create(**p)),
            ("chat.completions.create", lambda c, p: getattr(getattr(c, "chat", None), "completions").create(**p)),
            ("completions.create", lambda c, p: getattr(c, "completions").create(**p)),
            ("completions", lambda c, p: c.completions.create(**p) if hasattr(c, "completions") else c.completions(p)),
        ]

        resp = None
        final_exc = None

        for model_name in model_candidates:
            payload = dict(base_payload)
            payload["model"] = model_name
            print(f"[PROMPT] Attempting Groq model: {model_name}")

            last_exc = None
            for name, fn in method_attempts:
                try:
                    resp = fn(client, payload)
                    print(f"[PROMPT] Groq method succeeded: {name} (model={model_name})")
                    break
                except Exception as e:
                    last_exc = e
                    msg = repr(e)
                    print(f"[PROMPT] Groq method {name} failed for model {model_name}: {msg}")
                    # If the error explicitly says the model is decommissioned, stop trying
                    if "decommissioned" in msg or "model_decommissioned" in msg:
                        # mark this so outer loop will try the next model
                        break

            if resp is not None:
                break

            # If Groq reports the model is decommissioned, try the next candidate.
            err_text = repr(last_exc) if last_exc is not None else ""
            if "decommissioned" in err_text or "model_decommissioned" in err_text:
                print(f"[PROMPT] Model {model_name} reported decommissioned; trying next candidate")
                final_exc = last_exc
                continue
            # otherwise stop trying models and propagate the last error
            final_exc = last_exc
            break

        if resp is None:
            raise RuntimeError(f"All Groq method attempts failed: {repr(final_exc)}")

        # Parse response defensively and extract only the assistant prompt text
        enhanced_text = None

        # 1) ChatCompletion-like objects with `choices`
        try:
            if hasattr(resp, "choices"):
                choices = resp.choices
                if isinstance(choices, (list, tuple)) and len(choices) > 0:
                    first = choices[0]
                    # Try common attributes on the choice
                    if hasattr(first, "message") and hasattr(first.message, "content"):
                        enhanced_text = first.message.content
                    elif hasattr(first, "text"):
                        enhanced_text = first.text
                    elif isinstance(first, dict):
                        enhanced_text = first.get("text") or (first.get("message") or {}).get("content")

            # 2) Top-level message fields
            if not enhanced_text and hasattr(resp, "message") and hasattr(resp.message, "content"):
                enhanced_text = resp.message.content
        except Exception:
            enhanced_text = None

        # 3) Dict-like responses
        if not enhanced_text and isinstance(resp, dict):
            enhanced_text = resp.get("text") or resp.get("output")
            if not enhanced_text and "choices" in resp:
                choices = resp.get("choices")
                if isinstance(choices, list) and len(choices) > 0:
                    first = choices[0]
                    if isinstance(first, dict):
                        enhanced_text = first.get("text") or (first.get("message") or {}).get("content")

        # 4) Fallback: use the string representation only as a last resort
        if not enhanced_text:
            try:
                s = str(resp)
                enhanced_text = s.strip()
            except Exception:
                enhanced_text = None

        if enhanced_text and isinstance(enhanced_text, str) and enhanced_text.strip():
            cleaned = enhanced_text.strip()
            try:
                # Strip common Groq assistant prefaces while keeping the full prompt intact.
                lowered = cleaned.lower()
                if lowered.startswith("here is") or lowered.startswith("here's") or lowered.startswith("here’s"):
                    parts = cleaned.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        cleaned = parts[1].strip()
                    else:
                        colon_index = cleaned.find(":")
                        if colon_index != -1 and colon_index + 1 < len(cleaned):
                            cleaned = cleaned[colon_index + 1 :].strip()
            except Exception:
                pass

            # Trim surrounding quotes if still present
            if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
                cleaned = cleaned[1:-1].strip()

            print("[PROMPT] Using Groq AI enhancer")
            return cleaned
        else:
            raise RuntimeError("Groq returned empty or unparseable response")

    except Exception as e:
        # Any exception (import error, API error, timeout, parsing) → fallback
        print("[PROMPT] Groq failed, using fallback enhancer; error:", repr(e))
        try:
            return _rule_enhance_prompt(prompt)
        except Exception:
            # As a last resort, return the original prompt
            return prompt.strip()


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

    if original_img is None or bw_mask is None:
        gr.Warning("Please load an image and create a mask before inpainting.")
        return None

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

    if original_img is None:
        gr.Warning("Please load an image first.")
        return None

    bw_mask = normalize_mask_input(bw_mask, original_img)

    if bw_mask is None:
        gr.Warning("Please create or generate a mask first.")
        return None


    return _run_stable_diffusion(original_img, bw_mask, prompt, steps, guidance, strength)




def get_mask_and_inpaint(method, manual, auto, prompt, steps, guidance, strength):
    mask = manual if method == "Manual Masking" else auto
    return inpainting(mask, prompt, steps, guidance, strength)
