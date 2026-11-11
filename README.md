# HW3 — Vision-Language Image Assistant (CS 6180)

A Gradio web app that answers English questions about an image, and writes one-sentence captions, using the pretrained **SmolVLM-256M-Instruct** vision-language model. There is no training or fine-tuning.

## Files
| File | Purpose |
|---|---|
| `app.py` | Model loading, `ask_image()` inference, Flickr8k loader, and the Gradio app |
| `experiment.ipynb` | 5 dataset examples, 10-image held-out evaluation, prompt experiment, analysis |
| `mario_theme.json` | Local copy of the Gradio UI theme (auto-downloaded on first run) |
| `requirements.txt` | Pinned package versions |

## Setup
Requires Python 3.10+. Tested with Python 3.13 on an Apple M3 Pro (MPS).
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # or: pip install torch transformers accelerate datasets pillow gradio matplotlib pandas ipykernel
python -m ipykernel install --user --name cs6180-hw3 --display-name "Python (cs6180-hw3)"
```

## Run
```bash
python app.py                       # open http://127.0.0.1:7860
jupyter notebook experiment.ipynb   # select kernel "Python (cs6180-hw3)"
```
The first run downloads the model (about 500 MB) and streams the Flickr8k samples into `~/.cache/huggingface`.
The app picks the device automatically: CUDA, then Apple MPS, then CPU.

## API adjustments vs. starter code
- With transformers ≥ 5 and no dtype given, the model loads in the checkpoint's bfloat16 (MPS/CUDA).
- `do_sample=False` (greedy decoding) is used so that evaluation results are reproducible.

## Results
- Evaluation table: `eval_results.csv` (and section 5 of the notebook)
- Demo video: https://youtu.be/Z9XNi0SV9Lo

## Credits
- **Model:** [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct), Hugging Face TB Research, Apache-2.0.
- **Dataset:** Flickr8k, from M. Hodosh, P. Young, J. Hockenmaier, "Framing Image Description as a Ranking Task," *JAIR* 2013, via the [`intro/flickr8k`](https://huggingface.co/datasets/intro/flickr8k) Hugging Face mirror.
- **UI theme:** [`hmb/super-mario`](https://huggingface.co/spaces/hmb/super-mario) Gradio community theme.
- **Libraries:** PyTorch, Hugging Face Transformers / Datasets / Accelerate, Gradio, Pillow.
- Inference pattern adapted from the course's `VLM_Seven_Steps_Commented.ipynb` (Step 7) and the HW3 starter code.
