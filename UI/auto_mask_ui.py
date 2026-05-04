import gradio as gr
from ultralytics import YOLO

from config import SD_DEFAULT_STEPS, SD_DEFAULT_GUIDANCE, SD_DEFAULT_STRENGTH, YOLO_MODEL_NAME, DETECTION_ALPHA, PALETTE
from utils.utils import get_current_image, pil_to_numpy, draw_red_overlay, cleanup_mask
import numpy as np
from PIL import Image, ImageDraw
import cv2

_yolo_model: YOLO | None = None

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        print(f"[YOLO] Loading model: {YOLO_MODEL_NAME}")
        _yolo_model = YOLO(YOLO_MODEL_NAME)
        print("[YOLO] Model ready.")
    return _yolo_model


DETECTION_RESULTS: dict = {}

def run_detection():
    global DETECTION_RESULTS

    # 1. Load image
    orig = pil_to_numpy(get_current_image())
    h, w = orig.shape[:2]

    # 2. Run YOLO
    model = get_yolo_model()
    result = model(orig, verbose=False)[0]

    labels = []
    masks_bin = []
    class_counts = {}

    # 3. Process detections
    for cls_id, seg_mask in zip(result.boxes.cls, result.masks.data):
        name = model.names[int(cls_id)]

        # count objects
        class_counts[name] = class_counts.get(name, 0) + 1
        label = f"{name} {class_counts[name]}"
        labels.append(label)

        # resize + binarize mask
        mask = cv2.resize(seg_mask.cpu().numpy(), (w, h))
        mask = (mask > 0.5).astype(np.uint8) * 255
        masks_bin.append(mask)

    # 4. Create overlay
    overlay = orig.astype(np.float32)
    label_meta = []

    for i, (label, mask) in enumerate(zip(labels, masks_bin)):
        color = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
        mask_bool = mask.astype(bool)

        # apply color
        overlay[mask_bool] = overlay[mask_bool] * 0.6 + color * 0.4

        # find center
        ys, xs = np.where(mask_bool)
        if len(xs):
            cx, cy = int(xs.mean()), int(ys.mean())
            label_meta.append((cx, cy, label, tuple(color.astype(int))))

    # 5. Convert to image
    annotated = np.clip(overlay, 0, 255).astype(np.uint8)
    annotated_pil = Image.fromarray(annotated)

    # 6. Draw labels
    draw = ImageDraw.Draw(annotated_pil)
    for cx, cy, label, color in label_meta:
        draw.text((cx, cy), label, fill=color)

    # 7. Save results
    DETECTION_RESULTS = {
        "labels": labels,
        "masks_bin": masks_bin,
        "orig_np": orig,
    }

    return annotated_pil, gr.update(choices=labels)

def combine_selected_masks(selected_labels):
    data = DETECTION_RESULTS
    orig_np = data["orig_np"]
    h, w = orig_np.shape[:2]
    all_labels = data["labels"]
    all_masks = data["masks_bin"]

    combined = np.zeros((h, w), dtype=np.uint8)

    for label, mask in zip(all_labels, all_masks):
        if label in selected_labels:
            combined = cv2.bitwise_or(combined, mask)

    return cleanup_mask(combined)


def preview_selected_objects(selected_labels: list[str]):

    if not DETECTION_RESULTS:
        return gr.update(value=None), gr.update(visible=False)

    combined_mask = combine_selected_masks(selected_labels)
    overlay = draw_red_overlay(DETECTION_RESULTS["orig_np"], combined_mask)
    return gr.update(value=overlay), gr.update(visible=True)





def create_mask_from_selection(selected_labels: list[str]):
    # Defensive checks
    if not DETECTION_RESULTS:
        return gr.update(value=None), gr.update(value=None)

    # Ensure selected_labels is a list
    if selected_labels is None:
        selected_labels = []
    if isinstance(selected_labels, str):
        selected_labels = [selected_labels]

    # Generate combined mask from selected labels
    combined_mask = combine_selected_masks(selected_labels)

    combined_mask = cleanup_mask(combined_mask)

    # Create overlay preview (red overlay)
    overlay = draw_red_overlay(DETECTION_RESULTS["orig_np"], combined_mask)

    # Create B&W mask
    bw_mask = Image.fromarray(combined_mask)

    # Return explicit gr.update for both overlay preview and B&W image
    return gr.update(value=overlay), gr.update(value=bw_mask)



def auto_masking_ui():
    with gr.Column():
        detect_btn = gr.Button("🔍 Detect Objects ▶", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Detected Objects** — each one gets a different colour")
                detection_preview = gr.Image(label="", height=320, interactive=False)

                gr.Markdown("**Tick the objects you want to inpaint:**")
                object_checkboxes = gr.CheckboxGroup(
                    choices=[],
                    label="",
                    value=[],
                )
            
            with gr.Column(scale=1):
                gr.Markdown("**Selection Preview** — red = will be inpainted")
                overlay_preview = gr.Image(label="", height=225, interactive=False)

                # Keep the mask panel visible so updates render immediately
                with gr.Column(visible=True) as bw_mask_row:
                    gr.Markdown("**B&W Mask** — white = area to fill")
                    bw_mask_image = gr.Image(label="", height=225, interactive=False)
        
        detect_btn.click(
            fn=run_detection,
            inputs=[],
            outputs=[detection_preview, object_checkboxes],
        )

        object_checkboxes.change(
            fn=create_mask_from_selection,
            inputs=[object_checkboxes],
            outputs=[overlay_preview, bw_mask_image],
        )
    return bw_mask_image