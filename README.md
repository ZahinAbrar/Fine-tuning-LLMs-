# QLoRA Fine‑Tuning of Gemma on Google Colab
Parameter‑efficient fine‑tuning of **Gemma‑2B** using **QLoRA** (4‑bit base + LoRA adapters). Stable on **Colab T4/L4** GPUs. Includes training, inference (attach vs. merge), evaluation tips, and troubleshooting for common gotchas.

> **Why this exists**: Adapt a general LLM to a domain/task **without big GPUs**, while keeping the pipeline **simple, reproducible, and cheap**.

---

## Features
- **QLoRA**: 4‑bit (nf4) base + small **LoRA** adapters on attention projections.
- **Low‑VRAM recipe** that avoids OOM on T4/L4.
- **Two inference modes**: attach adapters at runtime, or merge‑and‑unload to a single checkpoint.
- Clean **instruction‑following data schema** (single text field).
- **Troubleshooting** for Kaggle/Hugging Face auth, KerasNLP vs. HF mixups, and OOM.

---

## Stack
- **Core**: `transformers`, `trl` (SFT), `peft` (LoRA/QLoRA), `bitsandbytes`, `datasets`, `accelerate`
- **Optional**: `keras-nlp` (alternate stack; don’t mix variable names with HF code)
- **Hardware**: Colab **T4 (16 GB)** or **L4 (24 GB)**; also runs on A100

---

## Quickstart (Colab, Hugging Face path — recommended)

> This path avoids Kaggle friction and is simplest on Colab.

### 1) Install
```bash
!pip -q install "transformers>=4.43" peft bitsandbytes accelerate datasets trl huggingface_hub
```

### 2) Log in to Hugging Face
Create a token at https://huggingface.co/settings/tokens (read permission).
```python
from huggingface_hub import login
login()  # paste your HF token when prompted
```

### 3) Train with QLoRA (SFT on a tiny sample for demo)
```python
import torch, os, gc
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

os.environ["WANDB_DISABLED"] = "true"   # keep memory down
gc.collect(); torch.cuda.empty_cache()

model_id = "google/gemma-2-2b-it"   # small, instruction-tuned Gemma

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,  # fp16 is safer on T4
)

tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tok.model_max_length = 256          # ↓ shorter context = big VRAM savings
tok.padding_side = "right"
tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb,
    device_map="auto"
)
base.gradient_checkpointing_enable()

# LoRA on attention projections
lora = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(base, lora)

# Example dataset (replace with your own domain data)
ds = load_dataset("tatsu-lab/alpaca", split="train[:1%]")

def fmt(ex):
    instr = ex["instruction"]
    inp   = ex.get("input","")
    out   = ex["output"]
    ex["text"] = f"### Instruction:\n{instr}\n### Input:\n{inp}\n### Response:\n{out}"
    return ex

ds = ds.map(fmt)

args = TrainingArguments(
    output_dir="out",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",
    report_to=[],
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=ds,
    dataset_text_field="text",
    args=args,
    max_seq_length=256
)

trainer.train()
model.save_pretrained("lora_adapter")  # small artifact
```

---

## Inference

### A) Attach adapters at runtime (recommended for Colab)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_id = "google/gemma-2-2b-it"
tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
tok.pad_token = tok.eos_token
tok.padding_side = "left"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
base = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, "lora_adapter").eval()

def build_prompt(instruction, inp=""):
    return f"### Instruction:\n{instruction}\n### Input:\n{inp}\n### Response:\n"

prompt = build_prompt("Summarize in one sentence.", "QLoRA trains small adapters over a 4-bit base model.")
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=150, temperature=0.2, top_p=0.9, do_sample=True)
print(tok.decode(out[0], skip_special_tokens=True))
```

### B) Merge‑and‑unload (single checkpoint export)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", torch_dtype=torch.float16, device_map="auto")
peft_model = PeftModel.from_pretrained(base, "lora_adapter")
merged = peft_model.merge_and_unload()

tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", use_fast=True)
merged.save_pretrained("merged_model")
tok.save_pretrained("merged_model")
```

---

## Data schema
Use a single text field per example for SFT:
```
### Instruction:
<instruction>
### Input:
<optional or empty>
### Response:
<gold answer>
```

**Tip:** Keep this schema for both training and inference prompts to reduce distribution shift.

---

## Safe defaults (T4‑friendly)
- Model: `google/gemma-2-2b-it`
- Quantization: 4‑bit **nf4** + double‑quant (`bitsandbytes`)
- Seq length: **256** (128 if you still OOM)
- Batch: **1**, Gradient Accum **16–32**
- LoRA: **r=4–8**, `q/k/v/o_proj` targets, dropout ≈ 0.05
- Gradient checkpointing: **on**
- Compute: **fp16** (bf16 if A100)

---

## Evaluation (starter)
- **Training signal:** cross‑entropy / **perplexity** on a small hold‑out split.
- **Task metrics (pick per task):**
  - Summarization: **ROUGE‑L**, ROUGE‑1/2 F1
  - QA: **Exact‑Match (EM)** / F1
  - Classification/IE: **F1** (micro/macro)
- **Hallucinations:** context‑bound spot checks on a sampled set (report rate ± CI).
- **Latency/Throughput (serving):** TTFT p95, tokens/s, p99 tail.

---

## System design notes (for deployment)
- **Serving engine:** vLLM/TGI with 4‑bit base; **on‑demand adapter attach** per tenant/use case.
- **SLOs:** TTFT ≤ 600 ms (p95), streaming enabled; tokens/s target per GPU.
- **Scaling:** small batch windows (≤25 ms), KV‑cache, autoscale on queue depth.
- **Observability:** TTFT, tokens/s, p95/p99, error rate, cache hit rate, cost/1K tokens.
- **Safety:** evidence‑required generation for RAG; refusal on low retrieval score; redaction.

---

## Troubleshooting
- **`SecretNotFoundError` / “not authorized”**: missing token or license.  
  - HF: `from huggingface_hub import login; login()` and accept model license page.
  - Kaggle Models: use `kagglehub.model_download("…/…/…/3")` with version **in the handle** and accept terms; CLI must be up to date.
- **`NameError: gemma_lm is not defined`**: you mixed **KerasNLP** code (`gemma_lm`) with **HF** variables (`model`, `tokenizer`). Use one stack consistently.
- **OOM / runtime crash**: drop `tok.model_max_length` to 256→128, set batch=1, increase grad accumulation, LoRA rank ≤ 8, enable grad checkpointing, use fp16, reduce `max_new_tokens` for tests.

---

## Notes on licenses & data
- Gemma requires accepting its license (HF or Kaggle). Respect the model’s usage terms.
- Ensure any domain data is **de‑identified** and permitted for fine‑tuning.

---

## Repo Structure (suggested)
```
.
├── train_qlora.py        # optional: move the training cell here
├── infer_attach.py       # attach‑adapter inference
├── infer_merge.py        # merged checkpoint inference
├── data/                 # your domain dataset (or HF script)
├── lora_adapter/         # saved adapters (output)
├── merged_model/         # merged model (optional output)
└── README.md
```

---

## FAQ
**Q: Why QLoRA instead of full fine‑tune?**  
A: 90% of the value at ~1% of the cost/VRAM; fits on T4/L4; easier validation/rollback via small adapters.

**Q: Can I use KerasNLP instead?**  
A: Yes—use `keras_nlp.models.GemmaCausalLM.from_preset(...)` and keep variables (`gemma_lm`) separate from HF code. Don’t mix them in one notebook.

**Q: How do I serve this?**  
A: Prefer attaching adapters to a 4‑bit base in an inference server (vLLM/TGI) for fast swaps; or ship a merged checkpoint if infra prefers single artifacts.

---

## Credits
- Built with the open‑source Hugging Face ecosystem and Google’s Gemma models.
- Inspired by the QLoRA paper and TRL SFT training patterns.
