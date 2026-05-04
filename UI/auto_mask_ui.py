import gradio as gr
from ultralytics import YOLO

from config import SD_DEFAULT_STEPS, SD_DEFAULT_GUIDANCE, SD_DEFAULT_STRENGTH, YOLO_MODEL_NAME, DETECTION_ALPHA, PALETTE
from utils.utils import get_current_image, pil_to_numpy,draw_red_overlay
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

    return annotated_pil, gr.update(choices=labels), gr.update(visible=False), gr.update(visible=False)

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

    return combined





def preview_selected_objects(selected_labels: list[str]):
    data = DETECTION_RESULTS
    combined_mask = combine_selected_masks(selected_labels)
    overlay = draw_red_overlay(data["orig_np"], combined_mask)
    return overlay, gr.update(visible=True)


def confirm_and_get_mask(selected_labels: list[str]):
    combined_mask = combine_selected_masks(selected_labels)
    bw_mask = Image.fromarray(combined_mask)
    return bw_mask, gr.update(visible=True)


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
                overlay_preview = gr.Image(label="", height=250, interactive=False)

                confirm_btn = gr.Button(
                    "✓ Confirm Selection",
                    variant="primary",
                    visible=False,
                )

                with gr.Column(visible=False) as bw_mask_row:
                    gr.Markdown("**B&W Mask** — white = area to fill")
                    bw_mask_image = gr.Image(label="", height=200, interactive=False)
        
        detect_btn.click(
            fn=run_detection,
            inputs=[],
            outputs=[detection_preview, object_checkboxes, confirm_btn, bw_mask_row],
        )

        object_checkboxes.change(
            fn=preview_selected_objects,
            inputs=[object_checkboxes],
            outputs=[overlay_preview, confirm_btn],
        )

        confirm_btn.click(
            fn=confirm_and_get_mask,
            inputs=[object_checkboxes],
            outputs=[bw_mask_image, bw_mask_row],
        )
    return bw_mask_image