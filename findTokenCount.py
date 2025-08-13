from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. Load your tokenizer
tokenizer = AutoTokenizer.from_pretrained("./Auto-Git-Handler-Hub/distilgpt2-commit-generator")
max_length = tokenizer.model_max_length  # typically 1024

# 2. Load your dataset
df = pd.read_csv("./commit_training.csv", header=None, names=["text"])

# 3. Function to tokenize a single row
def count_tokens(text):
    return len(tokenizer.encode(str(text), truncation=True, max_length=max_length))

# 4. Use ThreadPoolExecutor to parallelize token counting
total_tokens = 0
num_threads = 11  # adjust based on your CPU cores

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Submit all tasks
    futures = [executor.submit(count_tokens, text) for text in df["text"]]
    
    # Use tqdm to track progress
    for f in tqdm(as_completed(futures), total=len(futures), desc="Counting tokens", ncols=100):
        total_tokens += f.result()

print(f"\nTotal number of tokens in dataset: {total_tokens}")
print(f"Average tokens per entry: {total_tokens / len(df):.2f}")
