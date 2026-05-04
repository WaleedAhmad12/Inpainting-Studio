import gradio as gr
from utils.utils import url_load_image, file_load_image, inpainting, get_mask_and_inpaint, enhance_prompt
from UI.manual_mask_ui import manual_masking_ui
from UI.auto_mask_ui import auto_masking_ui
from config import SD_DEFAULT_GUIDANCE, SD_DEFAULT_STEPS, SD_DEFAULT_STRENGTH


with gr.Blocks(title="Inpaint Studio") as demo:

    gr.Markdown("# 🎨 Inpaint Studio")



    # ──────────────────────────────────────────
    #  STEP 1: Load an image
    # ──────────────────────────────────────────

    gr.Markdown("## Load Image")

    with gr.Group():

        with gr.Row():
            url_input = gr.Textbox(
                    placeholder="Paste an image URL here and press Enter…",
                    label="Image URL",
                    scale=2,
                )
            
            file_upload = gr.Image(
                    label="Or upload a file",
                    type="pil",
                    sources=["upload"],
                    scale=1,
                    height=120,
                )

        image_preview = gr.Image(
            label="Image Preview",
            type="pil",
            interactive=False,
            height=280
            )


    url_input.submit(fn=url_load_image, inputs=url_input, outputs=image_preview)
    url_input.blur(fn=url_load_image, inputs=url_input, outputs=image_preview)

    file_upload.change(fn=file_load_image, inputs=file_upload, outputs=image_preview)







    # ──────────────────────────────────────────
    #  STEP 2: Choose masking method
    # ──────────────────────────────────────────

    gr.Markdown("## Masking")
        
    with gr.Group():
        method_selector = gr.Radio(
            choices=["Manual Masking", "Auto Masking (YOLO)"],
            value="Manual Masking",
            label="Masking Method",
        )


    with gr.Group(visible=True) as manual_group:
        manual_canvas, manual_mask = manual_masking_ui()
    
    with gr.Group(visible=False) as auto_group:
        auto_mask = auto_masking_ui()
    
    # Update visibility when radio button changes
    def toggle_masking_method(method):
        return gr.update(visible=method == "Manual Masking"), gr.update(visible=method == "Auto Masking (YOLO)")
    
    method_selector.change(
        fn=toggle_masking_method,
        inputs=method_selector,
        outputs=[manual_group, auto_group]
    )
    
    # When image is loaded, set it as the canvas background
    image_preview.change(
        fn=lambda img: img,
        inputs=image_preview,
        outputs=manual_canvas
    )



    # ──────────────────────────────────────────
    #  STEP 3: Inpainting
    # ──────────────────────────────────────────


    gr.Markdown("## Inpainting")                    
    
    prompt_box = gr.Textbox(
        placeholder="Describe what should appear in the masked area…  e.g. 'a blue sky with clouds'",
        label="Prompt",
        lines=2,
    )

    enhance_btn = gr.Button("✨ Enhance Prompt", size="lg")

    # Event handler for prompt enhancement - updates the same textbox
    enhance_btn.click(
        fn=enhance_prompt,
        inputs=prompt_box,
        outputs=prompt_box,
    )

    with gr.Row():
        steps_slider    = gr.Slider(10, 100, value=SD_DEFAULT_STEPS,    step=1,    label="Steps (quality)")
        guidance_slider = gr.Slider(1,   20, value=SD_DEFAULT_GUIDANCE, step=0.5,  label="Guidance Scale (prompt strictness)")
        strength_slider = gr.Slider(0.1,  1, value=SD_DEFAULT_STRENGTH, step=0.05, label="Strength (how much to change)")

    run_inpaint_btn = gr.Button("🎨 Run Inpainting", variant="primary", size="lg")

    result_image = gr.Image(label="Result", height=380, interactive=False)

    run_inpaint_btn.click(
        fn=get_mask_and_inpaint,
        inputs=[method_selector, manual_mask, auto_mask, prompt_box, steps_slider, guidance_slider, strength_slider],
        outputs=result_image,
    )
 





if __name__ == "__main__":
    demo.launch()