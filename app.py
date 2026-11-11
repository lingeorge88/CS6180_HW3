"""
Name: George Lin

HW3 — Vision-Language Image Assistant (CS 6180).

Run:  source .venv/bin/activate && python app.py   ->  http://127.0.0.1:7860
"""

import os
import time
from itertools import islice  # take the first n rows of a streamed dataset

import gradio as gr
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText

# Settings

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"  # pretrained VLM (HW-required)
DATASET_ID = "intro/flickr8k"  # image-caption dataset (HW-required)
CAPTION_PROMPT = "Describe this image in one sentence."  # default caption prompt
N_GALLERY = 20  # dataset images shown in the app
THEME_FILE = "mario_theme.json"  # local copy of the hub theme

# Load the model ONCE at startup (HW step 3). Every button click reuses these
# objects, so a second image/question works without restarting Python.

# processor = image preprocessing (resize, patches, normalize -> pixel_values)
#           + tokenizer (text <-> token ids, chat template tokens)
processor = AutoProcessor.from_pretrained(MODEL_ID)
# device_map="auto" lets accelerate place the model on CUDA / Apple MPS / CPU.
# .eval() = inference mode (no dropout, no training).
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, device_map="auto").eval()
print(f"Model loaded on {model.device} ({model.dtype})")


def ask_image(image, question, max_new_tokens=70):
    """Run one image + question through the VLM. Returns (answer, seconds)."""
    # Guards: return a friendly message instead of crashing on empty input.
    if image is None:
        return "Please upload or select an image first", 0.0
    if not question or not question.strip():
        return "Please enter a question", 0.0
    image = image.convert("RGB")  # uploads may be RGBA/grayscale; model expects RGB

    # Chat-format message: one user turn containing the image and the question.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question.strip()},
            ],
        }
    ]

    # One call does it all: insert chat tokens + image placeholder tokens,
    # tokenize the text (input_ids), and preprocess the image (pixel_values).
    # add_generation_prompt=True appends "Assistant:" so the model answers next.
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(
        model.device
    )  # move tensors to the same device as the model

    start = time.perf_counter()  # time only the generation step
    with torch.inference_mode():  # no gradient tracking -> faster, less memory
        # do_sample=False = greedy decoding -> same answer every run (reproducible)
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    # generate() returns prompt + answer; slice off the prompt to keep only new tokens.
    tokens = output[0][inputs["input_ids"].shape[-1] :]
    # Token ids -> text; skip_special_tokens drops markers like <end_of_utterance>.
    answer = processor.decode(tokens, skip_special_tokens=True).strip()
    return answer, round(time.perf_counter() - start, 2)


def load_flickr_subset(n=60, split="test"):
    """Stream the first n Flickr8k rows -> list of {id, image, captions}."""
    # streaming=True reads rows one at a time instead of downloading the full dataset.
    ds = load_dataset(DATASET_ID, split=split, streaming=True)
    samples = []
    for i, row in enumerate(islice(ds, n)):
        # This mirror stores 5 human captions per image in caption_0 ... caption_4.
        captions = [row[f"caption_{k}"].strip() for k in range(5)]
        samples.append(
            {"id": i, "image": row["image"].convert("RGB"), "captions": captions}
        )
    return samples  # after the loop, so all n samples are returned


# Cache the gallery images (and their reference captions) once at startup.
SAMPLES = load_flickr_subset(N_GALLERY)


# Callbacks: Gradio passes the listed input components' values as arguments,
# and puts the returned values into the listed output components (in order).
def on_generate(image, question, max_tokens):
    answer, secs = ask_image(image, question, int(max_tokens))  # slider gives a float
    return answer, f"{secs} s"


def on_caption(image, max_tokens):
    answer, secs = ask_image(image, CAPTION_PROMPT, int(max_tokens))
    # 3rd value writes the caption prompt into the question box so the user sees it.
    return answer, f"{secs} s", CAPTION_PROMPT


def on_select_sample(evt: gr.SelectData):
    # gr.SelectData tells us which gallery tile was clicked (evt.index).
    sample = SAMPLES[evt.index]
    # image -> input box, first human caption -> reference box, clear old answer/time.
    return sample["image"], sample["captions"][0], "", ""


def on_upload():
    # User's own image has no human reference; also clear the old answer/time.
    return "(your own image - no reference caption)", "", ""


# Layout: image input, question box, Generate button, answer,
# inference time, default caption button, plus extras (slider, examples, gallery).

with gr.Blocks(title="VLM Image Assistant") as demo:
    gr.Markdown(
        "# Vision-Language Image Assistant\n"
        "Upload an image or click a Flickr8k sample, ask a question, and press **Generate**. "
        "Model: SmolVLM-256M-Instruct."
    )

    with gr.Row():  # left column = inputs, right column = outputs
        with gr.Column():
            image_in = gr.Image(
                type="pil", label="Image"
            )  # type="pil" -> PIL image to callbacks
            question_in = gr.Textbox(label="Question", value="What is happening?")
            max_tokens = gr.Slider(10, 150, value=70, step=5, label="Max new tokens")
            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary")
                caption_btn = gr.Button("Caption (one sentence)")
        with gr.Column():
            answer_out = gr.Textbox(label="AI answer", lines=4)
            time_out = gr.Textbox(label="Inference time")
            reference_out = gr.Textbox(label="Human reference caption")

    # Clickable preset questions: each inner list = one example, one value per input.
    gr.Examples(
        examples=[
            ["What is happening?"],
            ["How many people are there?"],
            ["What colors stand out?"],
            ["Where was this photo taken?"],
        ],
        inputs=question_in,
    )

    # 20 dataset images in a 5-column grid; height="auto" shows all rows without scrolling.
    gallery = gr.Gallery(
        value=[s["image"] for s in SAMPLES],
        label="Flickr8k samples (click one)",
        columns=5,
        height="auto",
        object_fit="cover",
        allow_preview=False,  # click selects the image instead of opening a lightbox
    )

    # Event wiring: component.event(function, inputs, outputs).
    generate_btn.click(
        on_generate, [image_in, question_in, max_tokens], [answer_out, time_out]
    )
    question_in.submit(  # pressing Enter in the question box also generates
        on_generate, [image_in, question_in, max_tokens], [answer_out, time_out]
    )
    caption_btn.click(
        on_caption, [image_in, max_tokens], [answer_out, time_out, question_in]
    )
    gallery.select(
        on_select_sample, None, [image_in, reference_out, answer_out, time_out]
    )
    image_in.upload(
        on_upload, None, [reference_out, answer_out, time_out]
    )  # user uploads only


# UI theme
if not os.path.exists(THEME_FILE):
    gr.Theme.from_hub("hmb/super-mario").dump(THEME_FILE)
MARIO_THEME = gr.Theme.load(THEME_FILE)

if __name__ == "__main__":  # start the server only when run directly (not on import)
    demo.launch(theme=MARIO_THEME)
