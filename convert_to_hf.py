# convert_to_hf.py
import os
import sys
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# ==============================
# 🔹 CONFIG (EDIT THESE)
# ==============================
MODEL_PT_PATH = "saved_models/checkpoint.pt"       # path to your .pt file
OUTPUT_DIR = "truthlens_v1"       # folder to save HF files
BASE_MODEL = "roberta-base"       # base architecture used during training
NUM_LABELS = 2                    # set to your number of classes

# ==============================
# 🔹 SAFETY CHECKS
# ==============================
if not os.path.exists(MODEL_PT_PATH):
    print(f"❌ model file not found: {MODEL_PT_PATH}")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔄 Loading base model (CPU)...")
model = RobertaForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=NUM_LABELS
)

print("🔄 Loading state_dict from .pt ...")
state = torch.load(MODEL_PT_PATH, map_location="cpu")

# Handle cases where checkpoint is wrapped (e.g., {'model_state_dict': ...})
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
elif isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]

# Some trainers save keys with 'module.' prefix (DataParallel)
clean_state = {}
for k, v in state.items():
    new_k = k.replace("module.", "") if k.startswith("module.") else k
    clean_state[new_k] = v

print("🔄 Loading weights into model...")
missing, unexpected = model.load_state_dict(clean_state, strict=False)

print("ℹ️ Missing keys:", len(missing))
print("ℹ️ Unexpected keys:", len(unexpected))

# If there are many missing/unexpected keys, architecture likely mismatched
if len(unexpected) > 0:
    print("⚠️ Warning: Unexpected keys found. Check if architecture matches.")

print("🔄 Loading tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained(BASE_MODEL)

print("💾 Saving in Hugging Face format...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Done! Files saved to:", OUTPUT_DIR)
print("\nExpected files:")
print("""
- config.json
- pytorch_model.bin
- tokenizer.json
- vocab.json
- merges.txt
- tokenizer_config.json
- special_tokens_map.json
""")

# ==============================
# 🔹 QUICK CPU TEST (optional)
# ==============================
def quick_test(text="This is a sample news headline."):
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    conf = float(torch.max(probs))
    print(f"\n🧪 Test Prediction → label: {pred}, confidence: {conf:.4f}")

# Run a quick test
quick_test()