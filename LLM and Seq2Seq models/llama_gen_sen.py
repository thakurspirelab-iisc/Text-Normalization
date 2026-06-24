import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import json
import re
import ast
import os

# =======================
# CONFIG
# =======================
MODEL_NAME     = "meta-llama/Llama-3.2-1B-Instruct"
LORA_PATH      = "/raid/home/rizwank/Normalization/model_building/lora_llama_kan_updated/epoch_2"
INPUT_FILE     = "/raid/home/rizwank/kannada_train.txt"   # same format as training
OUTPUT_FILE    = "/raid/home/rizwank/kannada_inference_out.jsonl"
MAX_NEW_TOKENS = 128
BATCH_SIZE     = 8        # adjust based on VRAM; each item is short
DEVICE         = "cuda:5"

INSTRUCTION = (
    "You are a text normalization system. "
    "Convert the unnormalized sentence into a natural spoken hindi sentence. "
    "Normalize all numbers into spoken hindi words. "
    "Keep the meaning unchanged."
)
INSTRUCTION_kan = (
    "You are a text normalization system. "
    "Convert the unnormalized sentence into a natural spoken Kannada sentence. "
    "Normalize all numbers into spoken Kannada words. "
    "Keep the meaning unchanged."
)

# =======================
# HELPERS  (identical to training)
# =======================

def clean_tagged_text(text):
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"^(?:\s*[0-9]\.\s*)+", "", text)
    text = re.sub(r"<[A-Z_]+>", "", text)
    text = re.sub(r"<([^>]+)>", r"\1", text)
    return text.strip()

def parse_line(line):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(line)
        except Exception:
            return None

def build_prompt(input_text: str) -> str:
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{INSTRUCTION}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{input_text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

def extract_response(generated: str) -> str:
    """Pull out only the assistant turn from the decoded text."""
    # The assistant header is the last one in the string
    marker = "assistant"
    if marker in generated:
        return generated.split(marker)[-1].strip()
    return generated.strip()

# =======================
# LOAD MODEL
# =======================
print("=" * 60)
print(" LOADING MODEL FOR INFERENCE")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token  = tokenizer.eos_token
tokenizer.padding_side = "left"          # left-pad for batch generation

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model = model.to(DEVICE)
model.eval()
print(f" LoRA loaded from: {LORA_PATH}")

# =======================
# READ INPUT
# =======================
records = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = parse_line(line)
        if obj is None:
            continue
        if "translated_tagged" not in obj:
            continue
        records.append(obj)

print(f" Loaded {len(records)} records from {INPUT_FILE}")

# =======================
# BATCH INFERENCE
# =======================
print("\n Running inference ...")

def run_batch(batch_records):
    prompts = [build_prompt(clean_tagged_text(r["translated_tagged"])) for r in batch_records]

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        add_special_tokens=False,
    ).to(DEVICE)

    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (strip the prompt)
    prompt_lens = enc["input_ids"].shape[1]
    responses = []
    for i, ids in enumerate(out_ids):
        new_ids = ids[prompt_lens:]          # everything after the prompt
        decoded = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        responses.append(decoded)
    return responses

os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Batches"):
        batch = records[i : i + BATCH_SIZE]
        responses = run_batch(batch)

        for rec, resp in zip(batch, responses):
            out_rec = dict(rec)           # copy all original keys
            out_rec["llama_op"] = resp    # add new key
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

print(f" Done! Output written to: {OUTPUT_FILE}")
print(f" Total records processed: {len(records)}")

# =======================
# QUICK SPOT-CHECK
# =======================
print(" Spot-check (first 5 outputs):")
with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    for _ in range(5):
        line = f.readline()
        if not line:
            break
        obj = json.loads(line)
        src = clean_tagged_text(obj.get("translated_tagged", ""))
        out = obj.get("llama_op", "")
        print(f"  IN : {src[:80]}")
        print(f"  OUT: {out[:80]}")
        print()