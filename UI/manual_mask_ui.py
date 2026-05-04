import gradio as gr
import numpy as np
import cv2
from PIL import Image
from config import MASK_DIFF_THRESHOLD, MASK_DILATE_KERNEL, SD_DEFAULT_STEPS, SD_DEFAULT_BRUSH_SIZE
from utils.utils import get_current_image, pil_to_numpy




def generate_mask(canvas_image):

    original_img = get_current_image()

    if canvas_image is None:
        return None

    if isinstance(canvas_image, dict):
        composite = canvas_image.get("composite")
    else:
        composite = canvas_image

    if composite is None:
        return None
    
    composite_np = pil_to_numpy(composite)
    original_np  = pil_to_numpy(original_img)

    if composite_np is None or original_np is None:
        return None 

    if composite_np.shape[:2] != original_np.shape[:2]:
        h, w = original_np.shape[:2]
        composite_np = cv2.resize(composite_np, (w, h), interpolation=cv2.INTER_NEAREST)

    diff = cv2.absdiff(composite_np, original_np)

    # Convert the 3-channel diff to single-channel (grayscale)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    _, mask = cv2.threshold(diff_gray, MASK_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MASK_DILATE_KERNEL, MASK_DILATE_KERNEL)
    )
    mask = cv2.dilate(mask, kernel, iterations=1)

    bw_mask= Image.fromarray(mask)

    return bw_mask



def manual_masking_ui():
    
    with gr.Group():
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Draw Mask")
                Brush_slider = gr.Slider(1, 100, value=SD_DEFAULT_BRUSH_SIZE, step=1, label="Brush Size")
                canvas = gr.ImageEditor(
                    label="",
                    type="pil",
                    height=380,
                    brush=gr.Brush(
                        colors=["#FF000080"],
                        color_mode="fixed",
                        default_size=SD_DEFAULT_BRUSH_SIZE,
                    ),
                    eraser=gr.Eraser(default_size=20),
                )
                
            with gr.Column():
                gr.Markdown("## Black & White Mask Preview")
                bw_mask_preview = gr.Image(
                    label="",
                    height=380,
                    interactive=False,  # read-only display
                )

    canvas.change(
        fn=generate_mask,
        inputs=canvas,
        outputs=bw_mask_preview,
    )
    
    # Update brush size dynamically when slider changes
    def update_brush_size(size):
        return gr.update(
            brush=gr.Brush(
                colors=["#FF000080"],
                color_mode="fixed",
                default_size=int(size),
            ),
        )
    
    Brush_slider.change(
        fn=update_brush_size,
        inputs=[Brush_slider],
        outputs=[canvas],
    )
    
    return canvas, bw_mask_preview