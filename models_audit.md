# Models Audit

## Findings
- ENCODER-DTYPE-BUG: fixed hardcoded `torch.float16` in `src/models/encoder/transformer_encoder.py` so AMP now respects `amp_dtype` (`bf16`/`fp16`).

## Status
- No additional production-grade issues found in the reviewed `src/models/` files.
