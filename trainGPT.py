import os
import time
import gc
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast, AutoModelForCausalLM, get_scheduler
from torch.optim import AdamW
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator

# ---------------- Hyperparameters ----------------
CSV_FILE = "commit_training.csv"   # <-- updated to your new CSV with 'training_text' column
BATCH_SIZE = 20
MAX_LENGTH = 150
NUM_EPOCHS = 2
LR = 5e-5
ACCUMULATION_STEPS = 20
NUM_WORKERS = 6
PREFETCH_FACTOR = 2
GC_EVERY = 500
FREEZE_UPTO_LAST_N = 2
MIXED_PRECISION = "fp16"
COMPILE_MODEL = True
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL = 5000
# -------------------------------------------------

class CommitDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=MAX_LENGTH):
        self.tokenizer = tokenizer
        self.texts = dataframe['training_text'].tolist()
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # For causal LM, labels are same as input_ids but mask pad tokens
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def load_data_and_tokenizer(csv_file):
    print("[INFO] Loading dataset:", csv_file)
    df = pd.read_csv(csv_file)
    tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[INFO] Dataset size: {len(df)}")
    return df, tokenizer

def freeze_model_layers(model, keep_last_n=FREEZE_UPTO_LAST_N):
    try:
        blocks = model.transformer.h
    except Exception:
        blocks = getattr(model, "transformer", None)
        if blocks is None:
            print("[WARN] Could not find transformer blocks to freeze.")
            return
    num_blocks = len(blocks)
    freeze_until = max(0, num_blocks - keep_last_n)
    for i in range(num_blocks):
        block = blocks[i]
        requires_grad = (i >= freeze_until)
        for p in block.parameters():
            p.requires_grad = requires_grad
    for p in model.transformer.wte.parameters():
        p.requires_grad = True
    if hasattr(model, "lm_head"):
        for p in model.lm_head.parameters():
            p.requires_grad = True
    print(f"[INFO] Frozen {freeze_until} transformer blocks; kept last {keep_last_n} trainable.")

def save_checkpoint(accelerator, global_step):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_step_{global_step}")
    if accelerator.is_main_process:
        print(f"[INFO] Saving checkpoint at step {global_step} to {checkpoint_path}")
        accelerator.save_state(checkpoint_path)

def load_latest_checkpoint(accelerator):
    if not os.path.exists(CHECKPOINT_DIR):
        return 0
    checkpoints = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint_step_")]
    if not checkpoints:
        return 0
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]), reverse=True)
    latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[0])
    print(f"[INFO] Loading latest checkpoint from {latest_checkpoint}")
    accelerator.load_state(latest_checkpoint)
    global_step = int(checkpoints[0].split("_")[-1])
    return global_step

def train():
    try:
        accelerator = Accelerator(mixed_precision=MIXED_PRECISION)
    except Exception as e:
        print(f"[WARN] Mixed precision {MIXED_PRECISION} not supported, falling back to no mixed precision: {e}")
        accelerator = Accelerator(mixed_precision="no")

    df, tokenizer = load_data_and_tokenizer(CSV_FILE)
    dataset = CommitDataset(df, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR,
    )

    print("[INFO] Loading model...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.resize_token_embeddings(len(tokenizer))

    if FREEZE_UPTO_LAST_N and FREEZE_UPTO_LAST_N > 0:
        freeze_model_layers(model, keep_last_n=FREEZE_UPTO_LAST_N)

    if COMPILE_MODEL and hasattr(torch, "compile"):
        try:
            print("[INFO] Compiling model with torch.compile() ...")
            model = torch.compile(model)
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}. Continuing without compile.")

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    num_training_steps = NUM_EPOCHS * len(dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    global_step = load_latest_checkpoint(accelerator)

    print(f"[INFO] Starting training loop from global step {global_step}")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{NUM_EPOCHS} =====")
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}", ncols=120)
        model.train()

        for step, batch in pbar:
            with accelerator.autocast():
                outputs = model(**batch)
                loss = outputs.loss / ACCUMULATION_STEPS

            accelerator.backward(loss)

            if (step + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % CHECKPOINT_INTERVAL == 0:
                    save_checkpoint(accelerator, global_step)

            if (step + 1) % GC_EVERY == 0:
                gc.collect()

            try:
                display_loss = loss.item() * ACCUMULATION_STEPS
            except Exception:
                display_loss = None
            if display_loss is not None:
                pbar.set_postfix({"loss": f"{display_loss:.4f}"})

        gc.collect()

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        print("[INFO] Saving final model to distilgpt2-commit-generator")
        unwrapped_model.save_pretrained("distilgpt2-commit-generator", save_function=accelerator.save)
        tokenizer.save_pretrained("distilgpt2-commit-generator")

    elapsed = time.time() - start_time
    print(f"[INFO] Training finished. Global steps: {global_step}. Elapsed: {elapsed:.1f}s")

if __name__ == "__main__":
    train()

