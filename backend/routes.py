from __future__ import annotations

import base64
from html import escape
from io import BytesIO

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

IMAGE_FILE = File(...)
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from backend.backend_types import EditForm, EditResponse
from backend.pipeline import PipelineStore

router = APIRouter()
store = PipelineStore()
edit_form = Depends(EditForm.as_form)
image_file = IMAGE_FILE


@router.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


async def read_uploaded_image(upload: UploadFile) -> Image.Image:
    try:
        return Image.open(BytesIO(await upload.read())).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(400, "Uploaded file is not a readable image") from error


async def run_edit(upload: UploadFile, form: EditForm) -> EditResponse:
    if not form.source_prompt.strip() or not form.target_prompt.strip():
        raise HTTPException(400, "Both prompts are required")
    try:
        image = await read_uploaded_image(upload)
        output = await run_in_threadpool(store.edit, image, form)
        return EditResponse(
            original_image=encode_image(image),
            image=encode_image(output),
            model=form.model,
            source_prompt=form.source_prompt,
            target_prompt=form.target_prompt,
        )
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        raise HTTPException(503, str(error)) from error


def encode_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_result(response: EditResponse) -> str:
    source_prompt = escape(response.source_prompt)
    target_prompt = escape(response.target_prompt)
    return f"""
<div class="comparison">
  <figure><img src="{response.original_image}" alt="Original image">
    <figcaption><strong>Original</strong><span>{source_prompt}</span></figcaption>
  </figure>
  <figure><img src="{response.image}" alt="Edited image">
    <figcaption><strong>Edited</strong><span>{target_prompt}</span></figcaption>
  </figure>
</div>
"""


@router.post("/edit", response_class=HTMLResponse)
async def edit_page(
    form: EditForm = edit_form,
    image: UploadFile = image_file,
) -> HTMLResponse:
    return HTMLResponse(render_result(await run_edit(image, form)))


@router.post("/api/edit", response_model=EditResponse)
async def edit_api(
    form: EditForm = edit_form,
    image: UploadFile = image_file,
) -> EditResponse:
    return await run_edit(image, form)


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script src="https://unpkg.com/htmx.org@2.0.4"></script>
  <title>ChordEdit</title>
  <style>
    body { font: 16px system-ui, sans-serif; margin: 2rem auto; max-width: 56rem; padding: 0 1rem; }
    form { display: grid; gap: .8rem; max-width: 36rem; }
    label { display: grid; gap: .3rem; }
    input, select, button { font: inherit; padding: .5rem; }
    button { cursor: pointer; }
    #status { color: #555; }
    #result { margin-top: 1rem; }
    .comparison { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }
    figure { margin: 0; }
    figure img { display: block; width: 100%; height: auto; }
    figcaption { display: grid; gap: .35rem; padding-top: .5rem; }
    figcaption span { white-space: pre-wrap; }
    @media (max-width: 700px) { .comparison { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <h1>ChordEdit</h1>
  <form action="/edit" method="post" enctype="multipart/form-data"
        hx-post="/edit" hx-target="#result" hx-swap="innerHTML"
        hx-encoding="multipart/form-data" hx-indicator="#status">
    <label>Image <input name="image" type="file" accept="image/*" required></label>
    <label>Model <select name="model"><option value="unet">UNet</option><option value="dit">DiT</option></select></label>
    <label>Source prompt <input name="source_prompt" required></label>
    <label>Target prompt <input name="target_prompt" required></label>
    <label>Seed <input name="seed" type="number" value="67"></label>
    <label>Noise samples <input name="noise_samples" type="number" min="1" max="16" value="1"></label>
    <label>Step scale <input name="step_scale" type="number" min="0.1" max="5" step="0.1" value="1"></label>
    <label>Start time <input name="t_start" type="number" min="0.01" max="1" step="0.01" value="0.9"></label>
    <label>End time <input name="t_end" type="number" min="0" max="0.99" step="0.01" value="0.3"></label>
    <label>Time delta <input name="t_delta" type="number" min="0" max="0.5" step="0.01" value="0.15"></label>
    <button type="submit">Edit</button>
  </form>
  <p id="status">The first 512px edit downloads and loads SD-Turbo; CPU inference may take a few minutes.</p>
  <div id="result"></div>
</body>
</html>
"""
