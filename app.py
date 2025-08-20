# app.py - Offline Qwen Image Editor (no API calls)

import gradio as gr
import torch
import numpy as np
import random
from diffusers import QwenImageEditPipeline

# Optional simple prompt cleaner
def simple_clean(prompt: str):
    # Just strip whitespace or you can add any local rewriter
    return prompt.strip()

# Determine device and dtype
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Qwen Image Edit model
pipe = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=dtype
).to(device)

MAX_SEED = np.iinfo(np.int32).max

def infer(
    image,
    prompt,
    seed=42,
    randomize_seed=False,
    true_guidance_scale=4.0,
    num_inference_steps=30,
    rewrite_prompt=True
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)

    if rewrite_prompt:
        prompt = simple_clean(prompt)

    negative_prompt = " "

    images = pipe(
        image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        true_cfg_scale=true_guidance_scale,
        num_images_per_prompt=1
    ).images

    return images[0], seed

# Gradio UI
def run_edit(img, pr, sd):
    return infer(img, pr, sd)

with gr.Blocks(title="Offline Qwen Image Edit") as demo:
    with gr.Row():
        input_image = gr.Image(type="pil", label="Input Image")
        result_image = gr.Image(type="pil", label="Edited Result")

    prompt_box = gr.Textbox(label="Prompt / Edit Instruction", placeholder="Describe the image edit")
    seed_slider = gr.Slider(0, MAX_SEED, value=42, label="Seed")
    run_btn = gr.Button("Edit Image")

    run_btn.click(
        fn=lambda i, p, s: run_edit(i, p, s),
        inputs=[input_image, prompt_box, seed_slider],
        outputs=[result_image, seed_slider]
    )

if __name__ == "__main__":
    demo.launch()
