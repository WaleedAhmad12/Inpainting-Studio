---
title: Inpaint Studio
emoji: 🎨
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.14.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Inpaint Studio

A Gradio-based image inpainting app that supports both manual brush masking and YOLO-powered auto masking.

## Features

- Load images from a URL or file upload
- Draw a manual mask with a brush editor
- Detect objects with YOLO and build an auto mask from selected objects
- Enhance prompts in-place for better Stable Diffusion inpainting results
- Run Stable Diffusion inpainting on the selected mask

## Requirements

- Python 3.10+
- A virtual environment is recommended
- You can see requirements.txt


## Setup

Create  a virtual environment:


```powershell
python -m venv .venv
```


If you are using the project virtual environment on Windows, activate it first:

```powershell
.venv\Scripts\Activate.ps1
```

## Run

Start the app:

```bash
python app.py
```

The app will open at the local Gradio URL printed in the terminal.


## Project Structure

- `app.py` - Gradio UI and app launch
- `UI/manual_mask_ui.py` - manual brush masking UI
- `UI/auto_mask_ui.py` - YOLO auto masking UI
- `utils/utils.py` - image loading, prompt enhancement, mask helpers, and diffusion pipeline logic
- `config.py` - model and UI settings
- `requirements.txt` - Python dependencies

## Architecture

```
Image Load 
    ↓
Choose Mask (Manual or Auto)
    ↓
Generate Mask
    ↓
Enhance Prompt
    ↓
Stable Diffusion Inpaint
    ↓
Output Image
```


REDACTED_GROQ_API_KEY