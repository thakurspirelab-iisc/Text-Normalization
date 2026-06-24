import torch
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence 
import json
import random
from transformers import MT5Tokenizer, MT5ForConditionalGeneration,MT5Tokenizer
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import MT5EncoderModel, MT5Tokenizer
from TB_RB import normalize_sent
# rb:0,mt5:1,hints_IB 2
class LM_DATASET(Dataset):
    def __init__(self, path, tokenizer):
        self.X = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())

                # Extract text + WER values
                unnorm = obj["unnorm"]
                a = obj["cer_rb"]
                b = obj["cer_mt5"]
                c = obj["cer_hints_ib"]

                # Which model was best
                ##best_model = pick_lowest(a, b, c)

                # Extract outputs
                op_rb = obj["rb"]
                op_mt5 = obj["mt5"]
                op_h_ib = obj["hints_ib"]

                # Get language and make a tag
                lang = obj.get("language", "unknown")   # default if missing
                lang_tag = f"<|{lang}|>"

                # Store all tokenized (language tag added to every field)
                self.X.append({
                    "unnorm": tokenizer(unnorm)['input_ids'],
                   ## "best_model": best_model,
                    "rb": tokenizer(op_rb)['input_ids'],
                    "mt": tokenizer(op_mt5)['input_ids'],
                    "h_ib": tokenizer(op_h_ib)['input_ids'],
                    "cer_rb":a,
                    "cer_mt5":b,
                    "cer_hints_IB":c
                }) 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class MT5MultiInputClassifier(nn.Module):
    def __init__(self, mt5_model_name='google/mt5-small', output_dim=3):
        super().__init__()
        # Load MT5 encoder
        self.encoder = MT5EncoderModel.from_pretrained(mt5_model_name)
        self.hidden_size = self.encoder.config.d_model  # usually 512 or 768
        
        # Linear layer after concatenation
        self.fc = nn.Linear(self.hidden_size * 4, output_dim)  # 4 sequences concatenated

    def forward(self, unnorm, rb_op, mt_op, h_ib_op, attention_mask=None):
        """
        Each input: (batch, seq_len)
        attention_mask is optional (if padding 0 is used, can auto-generate)
        """
        # Encode unnorm
        emb_unnorm = self.encoder(unnorm, attention_mask=(unnorm != 0).long()).last_hidden_state
        pooled_unnorm = emb_unnorm.mean(dim=1)  # simple mean pooling

        # Encode rb_op
        emb_rb = self.encoder(rb_op, attention_mask=(rb_op != 0).long()).last_hidden_state
        pooled_rb = emb_rb.mean(dim=1)

        # Encode mt_op
        emb_mt = self.encoder(mt_op, attention_mask=(mt_op != 0).long()).last_hidden_state
        pooled_mt = emb_mt.mean(dim=1)

        # Encode h_ib_op
        emb_h_ib = self.encoder(h_ib_op, attention_mask=(h_ib_op != 0).long()).last_hidden_state
        pooled_h_ib = emb_h_ib.mean(dim=1)

        # Concatenate all embeddings
        concatenated = torch.cat([pooled_unnorm, pooled_rb, pooled_mt, pooled_h_ib], dim=1)

        # Final linear layer
        logits = self.fc(concatenated)
        return logits
DEVICE="cuda:0"
import os
def collate_fn(batch):
    def pad(seqs):
        tensors = [torch.tensor(s, dtype=torch.long) for s in seqs]
        return pad_sequence(tensors, batch_first=True, padding_value=0)

    unnorm = pad([b["unnorm"] for b in batch])
    rb     = pad([b["rb"]     for b in batch])
    mt     = pad([b["mt"]     for b in batch])
    h_ib   = pad([b["h_ib"]   for b in batch])

    # logit 0=rb, 1=mt5, 2=hints_IB — same order as CER columns
    cer = torch.tensor(
        [[b["cer_rb"], b["cer_mt5"], b["cer_hints_IB"]] for b in batch],
        dtype=torch.float32
    )
    return unnorm, rb, mt, h_ib, cer


# ── LOSS: mean( sum_i( softmax(logit_i) * CER_i ) ) ──────────────────────────
def cer_weighted_softmax_loss(logits, cer):
    probs    = torch.softmax(logits, dim=1)   # (B, 3)
    weighted = probs * cer                    # (B, 3)
    return weighted.sum(dim=1).mean()         # scalar


# ── TRAIN ONE EPOCH ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0.0
    for unnorm, rb, mt, h_ib, cer in tqdm(loader, desc="  Train", leave=False):
        unnorm, rb, mt, h_ib, cer = (
            unnorm.to(DEVICE), rb.to(DEVICE),
            mt.to(DEVICE), h_ib.to(DEVICE), cer.to(DEVICE)
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(unnorm, rb, mt, h_ib)
            loss   = cer_weighted_softmax_loss(logits, cer)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── VALIDATE ──────────────────────────────────────────────────────────────────
def validate(model, loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for unnorm, rb, mt, h_ib, cer in tqdm(loader, desc="  Val  ", leave=False):
            unnorm, rb, mt, h_ib, cer = (
                unnorm.to(DEVICE), rb.to(DEVICE),
                mt.to(DEVICE), h_ib.to(DEVICE), cer.to(DEVICE)
            )
            with torch.cuda.amp.autocast():
                logits = model(unnorm, rb, mt, h_ib)
                loss   = cer_weighted_softmax_loss(logits, cer)
            total_loss += loss.item()
    return total_loss / len(loader)

def trainer(model, t_dataloader, v_dataloader, opt,
            loss_fn=cer_weighted_softmax_loss,
            epochs=4,
            device='cpu',
            val_every=1,
            op_folder="/home/sakshamt/SPIRE_TN/TN_Models/LM_selector_kan_new_l_fn"):

    model.to(device)
    best_val_loss = float("inf")

    # create folder if it doesn't exist
    os.makedirs(op_folder, exist_ok=True)

    save_path = os.path.join(op_folder, "best_model.pt")

    for e in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        loop = tqdm(t_dataloader, desc=f"Epoch {e}/{epochs}", leave=False)

        for unnorm, rb, mt, h_ib, cer in loop:
            unnorm, rb, mt, h_ib, cer = (
                unnorm.to(device), rb.to(device),
                mt.to(device),     h_ib.to(device),
                cer.to(device)
            )

            opt.zero_grad()
            logits = model(unnorm, rb, mt, h_ib)
            loss   = loss_fn(logits, cer)

            loss.backward()
            opt.step()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / (loop.n + 1))

        train_loss = running_loss / len(t_dataloader)
        print(f"Epoch {e}: Train Loss = {train_loss:.4f}")

        # ---- VALIDATION ----
        #if v_dataloader and e % val_every == 0:
        model.eval()
        total_cer = 0.0
        total_samples = 0

        with torch.no_grad():
            for unnorm, rb, mt, h_ib, cer in v_dataloader:
                unnorm, rb, mt, h_ib, cer = (
                unnorm.to(device), rb.to(device),
                mt.to(device),     h_ib.to(device),
                cer.to(device)
            )

                logits = model(unnorm, rb, mt, h_ib)

            
                probs = torch.softmax(logits, dim=1)
                pred  = torch.argmax(probs, dim=1)   # (B,)

                # Pick CER corresponding to predicted class
                batch_indices = torch.arange(cer.size(0), device=device)
                selected_cer  = cer[batch_indices, pred]

                total_cer += selected_cer.sum().item()
                total_samples += cer.size(0)

            avg_val_cer = total_cer / total_samples
            print(f"Epoch {e}: Val Avg CER = {avg_val_cer:.6f}")

    
            if avg_val_cer < best_val_loss:
                best_val_loss = avg_val_cer
                torch.save(model.state_dict(), save_path)
                print(f"✓ Saved best model (Val CER: {best_val_loss:.6f})")
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

dataset_train=LM_DATASET("/home/sakshamt/SPIRE_TN/DATASET/LM_train_kan.txt",tokenizer)
dataset_valid=LM_DATASET("/home/sakshamt/SPIRE_TN/DATASET/LM_val_kan.txt",tokenizer)
#dataset_test=LM_DATASET_1("/home/sakshamt/SPIRE_TN/DATASET/LM_hindi_data/test_lm_hi.txt",tokenizer)
dataloader_train=DataLoader(dataset_train,collate_fn=collate_fn,batch_size=32)
dataloader_valid=DataLoader(dataset_valid,collate_fn=collate_fn,batch_size=32)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = MT5MultiInputClassifier(mt5_model_name='google/mt5-small', output_dim=3).to(device)
opt=torch.optim.Adam(model.parameters())
trainer(model,dataloader_train,v_dataloader=dataloader_valid,opt=opt)