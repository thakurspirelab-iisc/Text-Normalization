```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import ast
import re
import string
from tqdm import tqdm

BASE_MODEL = "google/gemma-3-1b-it"
LORA_PATH = "/path/to/epoch_2"

INPUT_FILE = "/path/to/test.txt"
OUTPUT_FILE = "/path/to/test_gemma_output.txt"

DEVICE = "cuda:0"

INSTRUCTION = (
    "You are a text normalization system. "
    "Convert the unnormalised sentence into a natural spoken Hindi sentence. "
    "Normalize all numbers into spoken hindi words. "
    "Keep the meaning unchanged."
)
INSTRUCTION_kan= (
    "You are a text normalization system. "
    "Convert the unnormalised sentence into a natural spoken Kannada sentence. "
    "Normalize all numbers into spoken hindi words. "
    "Keep the meaning unchanged."
)

def parse_line(line):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(line)
        except Exception:
            return None


def clean_tagged_text(text):
    if text is None:
        return ""

    text = str(text)

    text = re.sub(r"^(?:\s*[0-9]\.\s*)+", "", text)

    text = re.sub(r"<[A-Z]+>", "", text)

    text = re.sub(r"<([^>]+)>", r"\1", text)

    return text.strip()


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map=None,
)

print("Loading LoRA...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

model = model.to(DEVICE)
model.eval()

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in tqdm(fin):

        if not line.strip():
            continue

        obj = parse_line(line)

        if obj is None:
            continue

        if "tagged_output" not in obj:
            continue

        tagged_text = obj["tagged_output"]

        input_text = clean_tagged_text(tagged_text)

        prompt = (
            "### Instruction:\n"
            f"{INSTRUCTION}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        obj["gemma_1b_op"] = generated

        fout.write(
            json.dumps(obj, ensure_ascii=False) + "\n"
        )

print("Inference complete.")
print("Saved:", OUTPUT_FILE)
```
