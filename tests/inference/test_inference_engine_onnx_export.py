import torch

from src.inference.inference_engine import InferenceEngine, InferenceConfig


class _DummyModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None):
        batch = input_ids.shape[0]
        return type("Out", (), {"logits": torch.randn(batch, 2)})


class _DummyTokenizer:
    def __call__(self, *args, **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }


def test_export_onnx_uses_expected_signature(monkeypatch, tmp_path):
    def _fake_load_model(self):
        self.model = _DummyModel()
        self.tokenizer = _DummyTokenizer()
        self.label_map = None

    monkeypatch.setattr(InferenceEngine, "_load_model", _fake_load_model)

    captured = {}

    def _fake_export(model, args, f, **kwargs):
        captured["args"] = args
        captured["f"] = f

    monkeypatch.setattr("src.inference.inference_engine.torch.onnx.export", _fake_export)

    engine = InferenceEngine(
        InferenceConfig(model_path=str(tmp_path), tokenizer_path=None, device="cpu")
    )
    out = tmp_path / "m.onnx"
    engine.export_onnx(str(out))

    assert len(captured["args"]) == 2
    assert str(captured["f"]).endswith("m.onnx")
