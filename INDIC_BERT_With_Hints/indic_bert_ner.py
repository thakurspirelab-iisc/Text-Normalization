import ast
import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, Trainer, TrainingArguments,
    TrainerCallback
)
import evaluate
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
# -----------------------------
# Configuration
# -----------------------------
args = {
    "train_batch_size": 32,
   # "eval_batch_size": 32,
    "num_train_epochs": 2,
    "fp16": False,
    "warmup_ratio": 0.1,
    "lr": 3e-5,
    "eval_strategy":'no',
  #  "eval_steps": 5000,
    "save_steps": 5000,
    "output_name": "eng_hin_tel_kan_gemini_full_4",
    "model_name": "ai4bharat/IndicBERTv2-MLM-Sam-TLM",
    
}

class_to_id = {
    "O": 0,
    "B-DATE": 1, "I-DATE": 2,
    "B-CARDINAL": 3, "I-CARDINAL": 4,
    "B-FRACTION": 5, "I-FRACTION": 6,
    "B-MONEY": 7, "I-MONEY": 8,
    "B-TELEPHONE": 9, "I-TELEPHONE": 10,
    "B-MEASURE": 11, "I-MEASURE": 12,
    "B-TIME": 13, "I-TIME": 14,
    "B-DECIMAL": 15, "I-DECIMAL": 16,
}
id_to_class = {v: k for k, v in class_to_id.items()}

# -----------------------------
# Custom Callback for Loss in Progress Bar
# -----------------------------
class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Loss will automatically show in the progress bar via logs
        pass

# -----------------------------
# Tokenize and tag function
# -----------------------------
import json
import ast

def parse_line(line):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(line)
        except Exception:
            return None

def tokenize_and_tag(sentence, bio_classes, tokenizer):
    import re

    pattern = r'<(.*?)><(.*?)>'
    tokens = []
    labels = []
    pos = 0

    for match in re.finditer(pattern, sentence):
        start, end = match.span()
        prefix = sentence[pos:start]
        entity_text = match.group(1)
        entity_label = match.group(2).upper()

        # Non-entity text
        if prefix.strip():
            prefix_tokens = tokenizer.tokenize(prefix)
            tokens.extend(prefix_tokens)
            labels.extend(["O"] * len(prefix_tokens))

        # Entity tokens
        ent_tokens = tokenizer.tokenize(entity_text)
        if len(ent_tokens) > 0:
            tokens.append(ent_tokens[0])
            labels.append(f"B-{entity_label}")
            for tok in ent_tokens[1:]:
                tokens.append(tok)
                labels.append(f"I-{entity_label}")

        pos = end

    # Remaining text
    suffix = sentence[pos:]
    if suffix.strip():
        suffix_tokens = tokenizer.tokenize(suffix)
        tokens.extend(suffix_tokens)
        labels.extend(["O"] * len(suffix_tokens))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    label_ids = [bio_classes.get(tag, 0) for tag in labels]

    return tokens, input_ids, labels, label_ids

# -----------------------------
# PyTorch Dataset
# -----------------------------
class TNDatasetNER_org(Dataset):
    def __init__(self, txt_path, tokenizer, label2id, max_len=None):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.sentences = []

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = ast.literal_eval(line)
                self.sentences.append(ex["tagged_output"])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        # Tokenize and tag
        tokens, input_ids, labels, label_ids = tokenize_and_tag(
            sentence, self.label2id, self.tokenizer
        )

        # Encode tokens
        encoding = self.tokenizer.encode_plus(
            tokens,
            is_split_into_words=True,
            truncation=self.max_len is not None,
            max_length=self.max_len if self.max_len else None,
            return_tensors="pt",
            add_special_tokens=True
        )

        # Align labels with special tokens
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(label_ids[word_idx])

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }
import torch
from torch.utils.data import Dataset

class TNDatasetNER(Dataset):
    def __init__(self, txt_path, tokenizer, label2id, max_len=None):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.sentences = []

        with open(txt_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                obj = parse_line(line)
                if obj is None:
                    print(f"[WARN] Skipping unparsable line {idx}")
                    continue

                if "translated_tagged" not in obj:
                    print(f"[WARN] Missing 'translated_tagged' at line {idx}")
                    continue

                sent = obj["translated_tagged"]
                if not sent or not isinstance(sent, str):
                    continue

                self.sentences.append(sent)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        tokens, input_ids, labels, label_ids = tokenize_and_tag(
            sentence, self.label2id, self.tokenizer
        )

        encoding = self.tokenizer.encode_plus(
            tokens,
            is_split_into_words=True,
            truncation=self.max_len is not None,
            max_length=self.max_len if self.max_len else None,
            return_tensors="pt",
            add_special_tokens=True
        )

        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(label_ids[word_idx])

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }

# -----------------------------
# Load tokenizer and model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
label_list = list(class_to_id.keys())
model = AutoModelForTokenClassification.from_pretrained(
    args["model_name"], num_labels=len(label_list)
)
data_collator = DataCollatorForTokenClassification(tokenizer)

# -----------------------------
# Load train/val datasets
# -----------------------------
train_txt_path = "/home/sakshamt/SPIRE_TN/train_kannada.txt"
#val_txt_path = "/home/sakshamt/SPIRE_TN/valid.txt"

train_dataset = TNDatasetNER(train_txt_path, tokenizer, class_to_id)
#val_dataset = TNDatasetNER(val_txt_path, tokenizer, class_to_id)

# -----------------------------
# Metrics
# -----------------------------
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id_to_class[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_class[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# -----------------------------
# TrainingArguments
# -----------------------------
#training_args_org = TrainingArguments(
 #   output_dir=f"outputs/{args['output_name']}",
  #  learning_rate=args["lr"],
   # per_device_train_batch_size=args["train_batch_size"],
    #per_device_eval_batch_size=args["eval_batch_size"],
   # num_train_epochs=args["num_train_epochs"],
   # save_strategy="steps",
   # save_steps=args["save_steps"],
    #eval_steps=args["eval_steps"],
   # do_eval=False,
    #evaluation_strategy="steps",
   # weight_decay=0.01,
   # fp16=args["fp16"],
   # logging_steps=1,  # Log every step to see loss continuously
   # logging_first_step=True,  # Log the first step
   # load_best_model_at_end=True,
   # metric_for_best_model="f1",
   # disable_tqdm=False,  # Keep progress bar enabled
   # logging_nan_inf_filter=False,  # Show all losses including edge cases
   # save_total_limit=2,  # Keep only 2 checkpoints to save space
   # save_only_model=True  # Save only model weights, not optimizer states
#)

training_args = TrainingArguments(
    output_dir=f"outputs/{args['output_name']}",
    learning_rate=args["lr"],
    per_device_train_batch_size=args["train_batch_size"],
    num_train_epochs=args["num_train_epochs"],

    save_strategy="steps",
    save_steps=args["save_steps"],

    evaluation_strategy="no",
    do_eval=False,

    load_best_model_at_end=False,  # ← MUST be False

    weight_decay=0.01,
    fp16=args["fp16"],

    logging_steps=1,
    logging_first_step=True,

    save_total_limit=2,
    save_only_model=True,
    disable_tqdm=False,
    report_to="none"
)


# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    #eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[LossLoggingCallback()]  # Add custom callback
)

# -----------------------------
# Train
# -----------------------------
trainer.train()
#trainer.evaluate()

# -----------------------------
# Save only model weights to specified directory
# -----------------------------
save_dir = "/home/sakshamt/SPIRE_TN/TN_Models/Indic_Bert/ner_weights_kan"
os.makedirs(save_dir, exist_ok=True)

# Save the model and tokenizer
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"\nModel weights saved to: {save_dir}")