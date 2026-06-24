print("========================================= running =======================================")
from transformers import AutoTokenizer
from indic_numtowords import num2words
import re
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, BertModel, AutoConfig, TrainingArguments, Trainer
import json
import torch
from typing import List, Dict, Optional
from utils_decoder import  get_span_token_indices,get_hint_from_span
import re
import os
from tqdm import tqdm
import ast
# -------------------------

file_path = "/home/sakshamt/SPIRE_TN/TN_Models/Indic_Bert/decoder_training_outputs/op_kan.txt"  # replace with your file
lang_code = "kan"
model_name = "ai4bharat/IndicBERTv2-MLM-Sam-TLM"
tokenizer = AutoTokenizer.from_pretrained(model_name)
 #Language mapping
language_mapping = {"en": 0, "hi": 1, "te": 2, "kan": 3}

# Entity type mapping
entity_type_mapping = {
    "CARDINAL": 0,
    "DATE": 1,
    "FRACTION": 2,
    "MONEY": 3,
    "TELEPHONE": 4,
    "MEASURE": 5,
    "TIME": 6,
    "DECIMAL": 7
}

import json
import torch
import re
import random
from typing import List, Dict, Optional
from copy import deepcopy

# Language mapping
language_mapping = {"en": 0, "hi": 1, "te": 2, "kan": 3}

# Entity type mapping
entity_type_mapping = {
    "CARDINAL": 0,
    "DATE": 1,
    "FRACTION": 2,
    "MONEY": 3,
    "TELEPHONE": 4,
    "MEASURE": 5,
    "TIME": 6,
    "DECIMAL": 7
}

REGEX_TAGGING = r'<([^>]+)><(DATE|DECIMAL|CARDINAL|MONEY|FRACTION|MEASURE|TIME|TELEPHONE)>'

# -----------------------------
# Space Augmentation Functions
# -----------------------------
# -----------------------------
def parse_line(line: str) -> Optional[dict]:
    """
    Parse a line from file.
    Tries JSON first, then Python-style literal.
    """
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(line)
        except Exception:
            return None

# -----------------------------
# Space Augmentation Functions
# -----------------------------
def augment_span_spacing(span: str, entity_type: str) -> List[str]:
    variants = [span]  # Always include original
    
    # Pattern 1: Add/remove spaces around special chars (/, -, ., \)
    if any(char in span for char in ['/', '-', '.', '\\']):
        spaced = span
        for char in ['/', '-', '.', '\\']:
            spaced = re.sub(f'\\{char}', f' {char} ', spaced)
        spaced = re.sub(r'\s+', ' ', spaced).strip()
        if spaced != span:
            variants.append(spaced)
        
        no_space = re.sub(r'\s*([/\-\.\\\s])\s*', r'\1', span)
        if no_space != span:
            variants.append(no_space)
        
        if '/' in span:
            variant1 = re.sub(r'/', r' /', span)
            variant1 = re.sub(r'\s+', ' ', variant1).strip()
            if variant1 not in variants:
                variants.append(variant1)
            variant2 = re.sub(r'/', r'/ ', span)
            variant2 = re.sub(r'\s+', ' ', variant2).strip()
            if variant2 not in variants:
                variants.append(variant2)
    
    # Pattern 2: Add/remove spaces between digits and words
    if re.search(r'\d[^\d\s]', span):
        with_space = re.sub(r'(\d)([^\d\s%/\-\.\\\,])', r'\1 \2', span)
        if with_space != span:
            variants.append(with_space)
    
    if re.search(r'\d\s+[^\d\s]', span):
        no_space = re.sub(r'(\d)\s+([^\d\s%/\-\.\\\,])', r'\1\2', span)
        if no_space != span:
            variants.append(no_space)
    
    # Pattern 3: Non-digit + digit
    if re.search(r'[^\d\s]\d', span):
        with_space = re.sub(r'([^\d\s%/\-\.\\\,])(\d)', r'\1 \2', span)
        if with_space != span:
            variants.append(with_space)
    
    if re.search(r'[^\d\s]\s+\d', span):
        no_space = re.sub(r'([^\d\s%/\-\.\\\,])\s+(\d)', r'\1\2', span)
        if no_space != span:
            variants.append(no_space)
    
    # Pattern 4: Spaces around %
    if '%' in span:
        with_space = re.sub(r'%', r' % ', span)
        with_space = re.sub(r'\s+', ' ', with_space).strip()
        if with_space != span:
            variants.append(with_space)
        
        no_space = re.sub(r'\s*%\s*', r'%', span)
        if no_space != span:
            variants.append(no_space)
    
    # Remove duplicates
    seen = set()
    unique_variants = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique_variants.append(v)
    
    return unique_variants

def augment_decoder_pairs_only(decoder_pairs: List[tuple]) -> List[tuple]:
    augmented_pairs = []
    for span, norm, entity_type in decoder_pairs:
        variants = augment_span_spacing(span, entity_type)
        selected_span = random.choice(variants)
        augmented_pairs.append((selected_span, norm, entity_type))
    return augmented_pairs

# -----------------------------
# Extract decoder pairs
# -----------------------------
def extract_decoder_pairs(tagged_sentence, normalized_sentence) -> Optional[List[tuple]]:
    """
    Safe extraction: converts non-string inputs to string
    """
    if not isinstance(tagged_sentence, str):
        tagged_sentence = str(tagged_sentence)
    if not isinstance(normalized_sentence, str):
        normalized_sentence = str(normalized_sentence)

    pattern = re.compile(REGEX_TAGGING)
    tagged_matches = pattern.findall(tagged_sentence)
    norm_matches = pattern.findall(normalized_sentence)

    if not tagged_matches or len(tagged_matches) != len(norm_matches):
        return None

    pairs = []
    for (tm, nm) in zip(tagged_matches, norm_matches):
        unnorm_text, tag1 = tm
        norm_text, tag2 = nm
        if tag1 != tag2:
            return None
        pairs.append((unnorm_text.strip(), norm_text.strip(), tag1.strip()))
    return pairs

# -----------------------------
# Load language file (safe for dict-like lines)
# -----------------------------
def load_language_file(file_path: str, lang_code: str = "hi", augment: bool = True, num_augmentations: int = 1) -> List[Dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            example = parse_line(line)
            if not example:
                continue

            tagged = str(example.get("translated_tagged", ""))
            normalized = str(example.get("normalized_output", ""))
            if not tagged or not normalized:
                continue

            decoder_pairs = extract_decoder_pairs(tagged, normalized)
            if not decoder_pairs:
                continue

            unnormalised = re.sub(r"<([^>]+)><[^>]+>", r"\1", tagged)

            original_sample = {
                "unnormalised": unnormalised,
                "tagged": tagged,
                "normalized": normalized,
                "decoder_pairs": decoder_pairs,
                "language": lang_code
            }
            data.append(original_sample)

            if augment:
                for _ in range(num_augmentations):
                    augmented_pairs = augment_decoder_pairs_only(decoder_pairs)
                    if augmented_pairs == decoder_pairs:
                        continue
                    data.append({
                        "unnormalised": unnormalised,
                        "tagged": tagged,
                        "normalized": normalized,
                        "decoder_pairs": augmented_pairs,
                        "language": lang_code
                    })
    print(f"✅ Loaded {len(data)} samples from {file_path}")
    return data

dataset = load_language_file(file_path, lang_code="kan", augment=True, num_augmentations=1)



class DecoderDataset_1(Dataset):
    """
    Dataset for decoder training.

    Input: list of examples where each example is a dict:
        {
            "unnormalised": full sentence,
            "tagged": sentence with tagged spans,
            "normalized": sentence with normalized spans,
            "decoder_pairs": list of (unnorm_span, normalized_text, entity_type),
            "language": language code ("hi", "en", etc.)
        }

    Output for each item:
        {
            "encoder_input_ids": token ids of full sentence,
            "decoder_input_ids": token ids of normalized target,
            "unnorm_span_ids": token ids of unnormalized span,
            "span_indices": token positions of span in full sentence,
            "hint_input_ids": token ids of hint text,
            "language_id": numeric language id,
            "entity_type_id": numeric entity type id
        }
    """
    def __init__(
        self,
        data,
        tokenizer,
        entity_type_mapping,
        language_mapping,
        max_encoder_length=128,
        max_decoder_length=32,
        max_hint_length=16,
        preprocessed=False,
        debug=False
    ):
        self.tokenizer = tokenizer
        self.entity_type_mapping = entity_type_mapping
        self.language_mapping = language_mapping
        self.max_encoder_length = max_encoder_length
        self.max_decoder_length = max_decoder_length
        self.max_hint_length = max_hint_length

        self.data = []

        progress_bar = tqdm(data, desc="Processing data", dynamic_ncols=True, disable=not debug)
        error_count = 0

        for sample in progress_bar:
            if not sample.get("decoder_pairs"):
                continue

            lang_id = torch.tensor(language_mapping[sample["language"]], dtype=torch.long)

            # Tokenize full sentence for encoder
            encoder_inputs = tokenizer(
                sample["unnormalised"],
                truncation=True,
                max_length=self.max_encoder_length,
                return_tensors="pt",
                return_offsets_mapping=True
            )
            #input_ids = encoder_inputs["input_ids"].squeeze(0)
            offset_mapping = encoder_inputs["offset_mapping"].squeeze(0).tolist()

            for decoder_pair in sample["decoder_pairs"]:
                if len(decoder_pair) != 3:
                    continue

                unnorm_span, norm_text, entity_type = decoder_pair

                
                # Tokenize unnormalized span
                unnorm_span_ids = tokenizer(
                    unnorm_span,
                    truncation=True,
                    max_length=self.max_encoder_length,
                    return_tensors="pt"
                )["input_ids"].squeeze(0)

                # Tokenize normalized target
                decoder_input_ids = tokenizer(
                    norm_text,
                    truncation=True,
                    max_length=self.max_decoder_length,
                    return_tensors="pt"
                )["input_ids"].squeeze(0)

                # Generate rule-based hint
                hint_text, error_flag = get_hint_from_span(unnorm_span, lang=sample["language"])
                if error_flag:
                    error_count += 1
                hint_input_ids = tokenizer(
                    hint_text,
                    truncation=True,
                    max_length=self.max_hint_length,
                    return_tensors="pt"
                )["input_ids"].squeeze(0)

                # Add to dataset
                self.data.append({
                    
                    "decoder_input_ids": decoder_input_ids,
                    "unnorm_span_ids": unnorm_span_ids,
                    
                    "hint_input_ids": hint_input_ids,
                    "language_id": lang_id,
                    "entity_type_id": torch.tensor(entity_type_mapping[entity_type], dtype=torch.long)
                })

            progress_bar.set_postfix(errors=error_count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
d_dataset=DecoderDataset_1(dataset,tokenizer,entity_type_mapping,language_mapping) 



class BertDecoderOnlyLM_org(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_languages,                # Number of language tokens.
        num_entity_types,             # Number of entity types.
        tokenizer,
        model_name = "google/muril-base-cased",
        max_len=512,
        sep_token_id=None,
        padding_token_id=None,
    ):
        super(BertDecoderOnlyLM_org, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tokenizer = tokenizer
        
        # Initialize configuration first
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.is_decoder = True
        self.config.add_cross_attention = False
        self.hidden_size = self.config.hidden_size  # Should be equal to n_embd ideally

        # Load the decoder model (BERT repurposed)
        self.decoder = BertModel.from_pretrained(model_name, config=self.config)
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        
        # Positional Embedding for the constructed prompt
        # self.pos_embedding = nn.Embedding(max_len, self.hidden_size)

        # --- Final Output Head ---
        self.ln_f = nn.LayerNorm(self.hidden_size)
        self.head = nn.Linear(self.hidden_size, vocab_size, bias=False)
        # Tie the head weight to the token embeddings from the decoder.
        self.head.weight = self.decoder.embeddings.word_embeddings.weight

        # --- Additional Conditioning Embeddings ---
        self.lang_embedding = nn.Embedding(num_languages, self.hidden_size)
        self.entity_embedding = nn.Embedding(num_entity_types, self.hidden_size)

        # Set SEP and PAD token IDs.
        self.sep_token_id = sep_token_id if sep_token_id is not None else 105 
        self.padding_token_id = padding_token_id if padding_token_id is not None else 0

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.padding_token_id)

    def forward(
        self,
        unnorm_span_ids,   # Tokens for the unnormalized span.
        hint_input_ids,      # Tokens for the hint (e.g., numbers converted to words).
        decoder_input_ids,   # Decoder input tokens (e.g., target sequence with BOS).
        entity_ids,          # Entity type indices.
        language_ids,        # Language indices.
        labels=None,          # Optional target tokens for loss computation.
        **kwargs
    ):
        """
        Forward pass for the decoder-only model.
        The input prompt is constructed as:
          [Language Token] + [Entity Token] + [CLS] + [Hint Span] + [SEP] + [CLS] + [Unnormalized Span] + [SEP] + [CLS] + [Decoder Input] + [SEP]
        The model outputs the normalized span.
        """
        # breakpoint()
        batch_size = unnorm_span_ids.size(0)
        
        # --- Conditioning Embeddings ---
        # Obtain language and entity embeddings in hidden_size.
        lang_emb = self.lang_embedding(language_ids)   # (B, hidden_size)
        if len(lang_emb.shape) == 2:
            lang_emb = lang_emb.unsqueeze(1)                # (B, 1, hidden_size)
        ent_emb = self.entity_embedding(entity_ids)       # (B, hidden_size)
        if len(ent_emb.shape) == 2:
            ent_emb = ent_emb.unsqueeze(1)                  # (B, 1, hidden_size)

        # --- Token Embeddings ---
        hint_emb = self.decoder.embeddings(hint_input_ids)    # (B, L_hint, hidden_size)
        span_emb = self.decoder.embeddings(unnorm_span_ids)   # (B, L_span, hidden_size)
        dec_emb = self.decoder.embeddings(decoder_input_ids)    # (B, L_dec, hidden_size)

        # --- Construct Input Prompt ---
        # Prompt: [Language Token] + [Entity Token] + [CLS] + [Hint Span] + [SEP] 
        # # hint contains [CLS] and [SEP] as it is from the tokenizer without excluding special tokens, so no need to explicitly add them.
        prompt_seq = torch.cat([lang_emb, ent_emb, hint_emb], dim=1)  # (B, L_prompt, hidden_size)
        # Full input: [Prompt Seq] + [CLS] + [Unnormalized Span] + [SEP] + [CLS] + [Decoder Input] + [SEP]
        # full_input = torch.cat([prompt_seq, sep_emb, span_emb, sep_emb, dec_emb], dim=1)  # (B, T, hidden_size)
        full_input = torch.cat([prompt_seq, span_emb, dec_emb], dim=1)  # (B, T, hidden_size)
        
        seq_len = full_input.size(1)
        # --- Positional Embeddings ---
        # positions = torch.arange(0, seq_len, device=full_input.device).unsqueeze(0)
        # pos_emb = self.pos_embedding(positions)  # (1, T, hidden_size)
        # full_input = full_input + pos_emb

        # --- Causal Mask ---
        # causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=full_input.device))

        # --- Transformer Decoder ---
        # decoder_output = self.decoder(inputs_embeds=full_input, attention_mask=causal_mask)[0]  # (B, T, hidden_size)
        decoder_output = self.decoder(inputs_embeds=full_input)[0]  # (B, T, hidden_size)

        # --- Final Output Head ---
        x_norm = self.ln_f(decoder_output)
        logits_all = self.head(x_norm)  # (B, T, vocab_size)

        # Compute context length: length of [Prompt Seq] + [SEP] + [Unnormalized Span] + [SEP]
        L_prompt = prompt_seq.size(1)
        L_span = span_emb.size(1)
        # context_len = L_prompt + 1 + L_span + 1
        context_len = L_prompt + L_span
        
        # breakpoint()
        # Extract logits corresponding to the decoder input portion.
        logits = logits_all[:, context_len:, :]
        
        if labels is not None:
            target = labels[:, 1:].contiguous()
            shifted_logits = logits[:, :-1, :]
            loss = self.loss_fct(shifted_logits.reshape(-1, shifted_logits.size(-1)), target.reshape(-1))
            return {"loss": loss, "logits": logits}
    
        return {"logits": logits}


def data_collator(batch):
    tensors_to_pad = ["unnorm_span_ids", "decoder_input_ids", "hint_input_ids"]#["unnorm_span_ids", "decoder_input_ids", "hint_input_ids", "encoder_input_ids"]
    return_item = {}
    for tensor_name in tensors_to_pad:
        max_len = max(item[tensor_name].size(0) for item in batch)
        padded_tensor = []
        for item in batch:
            tensor = item[tensor_name]
            pad_size = max_len - tensor.size(0)
            if pad_size > 0:
                pad = torch.full((pad_size,), tokenizer.pad_token_id, dtype=torch.long)
                tensor = torch.cat([tensor, pad], dim=0)
            padded_tensor.append(tensor)
        return_item[tensor_name] = torch.stack(padded_tensor, dim=0)

    #max_span_len = max(item["span_indices"].size(0) for item in batch)
    #padded_span_indices = []
    #for item in batch:
        #span = item["span_indices"]
       # pad_size = max_span_len - span.size(0)
      #  if pad_size > 0:
     #       pad = torch.full((pad_size,), -1, dtype=torch.long)
    #        span = torch.cat([span, pad], dim=0)
   #     padded_span_indices.append(span)
  #  span_indices_tensor = torch.stack(padded_span_indices, dim=0)

    entity_type_ids = torch.stack([item["entity_type_id"] for item in batch], dim=0)
    language_ids = torch.stack([item["language_id"] for item in batch], dim=0)

    return {
        **return_item,
        "entity_type_ids": entity_type_ids,
        "language_ids": language_ids,
        "labels": return_item["decoder_input_ids"]}
      #  "span_indices": span_indices_tensor,
    #}
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            unnorm_span_ids=inputs["unnorm_span_ids"],
            hint_input_ids=inputs["hint_input_ids"],
            decoder_input_ids=inputs["decoder_input_ids"],
            entity_ids=inputs["entity_type_ids"],
            language_ids=inputs["language_ids"],
            labels=inputs["labels"]
           # span_indices=inputs["span_indices"],
            #encoder_input_ids=inputs["encoder_input_ids"]
        )
        loss = outputs.get("loss")
        return (loss, outputs) if return_outputs else loss


# -------------------------------
# 9. Load dataset
# -------------------------------
#file_path = "/home/sakshamt/SPIRE_TN/valid.txt"
#lang_code = "hi"
#data = load_language_file(file_path, lang_code)
#dataset = DecoderDataset(data, tokenizer, entity_type_mapping)

# -------------------------------
# 10. Initialize model
# -------------------------------

# -------------------------------
# 1. Device
# -------------------------------
device = "cuda:1"  # or "cpu"

decoder_model = BertDecoderOnlyLM_org(
    vocab_size=len(tokenizer),
    num_languages=len(language_mapping),
    num_entity_types=len(entity_type_mapping),
    tokenizer=tokenizer,
    model_name="ai4bharat/IndicBERTv2-MLM-Sam-TLM"
).to(device)

# -------------------------------
# 2. Training Arguments - Train for 100 steps instead of epochs
# -------------------------------
training_args = TrainingArguments(
    output_dir="decoder_training_outputs",
    per_device_train_batch_size=16,
    
    num_train_epochs=4,  # Still need this but max_steps takes precedence
    save_total_limit=3,
    logging_steps=50,  # Log every 10 steps
    # Evaluate every 50 steps (if you have eval dataset)
    save_strategy="no",
    
    
    remove_unused_columns=False,
    report_to="none",  # Disable wandb/tensorboard if not needed
    gradient_accumulation_steps=1,  # No gradient accumulation
    warmup_steps=10,  # 10 steps of warmup
    lr_scheduler_type="linear",
    learning_rate=5e-5,
    weight_decay=0.01,
)

# -------------------------------
# 3. Trainer
# -------------------------------
trainer = CustomTrainer(
    model=decoder_model,
    args=training_args,
    train_dataset=d_dataset,
    data_collator=data_collator,
    # If you have validation dataset:
    # eval_dataset=val_dataset,
)

# -------------------------------
# 4. Start Training (will stop after 100 steps)
# -------------------------------
trainer.train()

# -------------------------------
# 5. Save Weights Manually to .pt


output_folder = "/home/sakshamt/SPIRE_TN/TN_Models/Indic_Bert/weights_kan"
os.makedirs(output_folder, exist_ok=True)

# Method 1: Simple save (will work now with fixed model)
torch.save(decoder_model.state_dict(), f"{output_folder}/IB_decoder_weights_kan_updated.pt")
print(f"Model weights saved to {output_folder}/IB_decoder_weights_kan.pt")
