import pandas as pd
import re

# Load CSV
df = pd.read_csv("./commit_data.csv")

# Replace NaN with empty strings globally
df = df.fillna("")

# --- Cleaning commit messages ---
def clean_message(msg):
    if not isinstance(msg, str):
        return ""
    msg = re.sub(r'git-svn-id:.*', '', msg)  # remove svn metadata
    msg = re.sub(r'\s+', ' ', msg).strip()   # collapse spaces
    return msg

# --- Formatting diff text ---
def format_diff(additions, removals):
    adds = [f"+ {line.strip()}" for line in additions.splitlines() if line.strip()] if additions else []
    rems = [f"- {line.strip()}" for line in removals.splitlines() if line.strip()] if removals else []
    if not adds:
        adds = ["+ None"]
    if not rems:
        rems = ["- None"]
    return "\n".join(adds + rems)

df["clean_message"] = df["message"].apply(clean_message)
df["diff_text"] = df.apply(lambda row: format_diff(row["additions"], row["removals"]), axis=1)

# --- Create prompt + target ---
df["prompt"] = df["diff_text"].apply(lambda diff: f"Commit message for the following changes:\n{diff}\n---\n")
df["target"] = df["clean_message"]

# --- Training formats ---
# 1. CSV format (prompt, target)
df[["prompt", "target"]].to_csv("commit_prompts.csv", index=False)

# 2. Plain text format for GPT2 with EOS token
EOS_TOKEN = "<|endoftext|>"
df["training_text"] = df["prompt"] + df["target"] + " " + EOS_TOKEN
df["training_text"].to_csv("commit_training.txt", index=False, header=False)

# 3. Also create a CSV version with training_text
df[["training_text"]].to_csv("commit_training.csv", index=False)

