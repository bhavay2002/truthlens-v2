import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def _build_idx_to_label(model) -> dict[int, str]:
    """Use config.id2label first; fallback to label2id if needed."""

    idx_to_label: dict[int, str] = {}

    id2label = getattr(model.config, "id2label", None) or {}
    for idx, label in id2label.items():
        idx_to_label[int(idx)] = str(label).strip().upper()

    if idx_to_label:
        return idx_to_label

    label2id = getattr(model.config, "label2id", None) or {}
    for label, idx in label2id.items():
        idx_to_label[int(idx)] = str(label).strip().upper()

    if not idx_to_label:
        idx_to_label = {0: "REAL", 1: "FAKE"}

    return idx_to_label


def _get_label_index(idx_to_label: dict[int, str], target: str) -> int | None:
    target = target.strip().upper()
    for idx, label in idx_to_label.items():
        if label == target:
            return idx
    return None


# Path to trained model
MODEL_PATH = "models/roberta_model"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()    

print("Model loaded successfully on", device)

# Replace these two strings to test your own example
title = "Mixed messages from Trump leave more questions than answers over war's end"
text = (
    "President Donald Trump and his administration have so far offered mixed "
    "messages and contradictory explanations on the joint US Israeli military "
    "campaign against Iran."
)

input_text = f"{title} {text}"

# Tokenize
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=512,
)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Prediction
with torch.no_grad():
    outputs = model(**inputs)

probs = F.softmax(outputs.logits, dim=1)[0]
pred_idx = int(torch.argmax(probs).item())

idx_to_label = _build_idx_to_label(model)
pred_label = idx_to_label.get(pred_idx, f"CLASS_{pred_idx}")

real_idx = _get_label_index(idx_to_label, "REAL")
fake_idx = _get_label_index(idx_to_label, "FAKE")

print("\nClass probabilities:")
for idx in sorted(idx_to_label.keys()):
    label = idx_to_label[idx]
    print(f"- {label:<8} ({idx}): {probs[idx].item() * 100:.2f}%")

if fake_idx is not None:
    print(f"\nFake News Probability: {probs[fake_idx].item() * 100:.2f}%")
else:
    print("\nFake News Probability: N/A (FAKE label not found in model config)")

if real_idx is not None:
    print(f"Real News Probability: {probs[real_idx].item() * 100:.2f}%")
else:
    print("Real News Probability: N/A (REAL label not found in model config)")

print(f"\nPredicted class index: {pred_idx}")
print(f"Prediction: {pred_label}")
