#!/usr/bin/env python3
"""
TruthLens AI — API Test Script
Usage:
    python api_test.py                  # interactive mode (prompts for text)
    python api_test.py --text "..."     # pass text directly
    python api_test.py --endpoint all   # run all endpoints (default)
    python api_test.py --endpoint analyze
    python api_test.py --endpoint explain
    python api_test.py --endpoint predict
    python api_test.py --endpoint report
    python api_test.py --endpoint health
"""

import argparse
import json
import sys
import textwrap
import time

try:
    import requests
except ImportError:
    print("ERROR: 'requests' library not found. Run: pip install requests")
    sys.exit(1)

BASE_URL = "http://localhost:5000"
SEPARATOR = "─" * 70


def _color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def green(t):  return _color(t, "32")
def yellow(t): return _color(t, "33")
def cyan(t):   return _color(t, "36")
def bold(t):   return _color(t, "1")
def red(t):    return _color(t, "31")
def dim(t):    return _color(t, "2")


def _section(title: str):
    print(f"\n{cyan(bold(title))}")
    print(dim(SEPARATOR))


def _ok(label: str, value):
    print(f"  {green('✓')} {bold(label)}: {value}")


def _warn(label: str, value):
    print(f"  {yellow('!')} {bold(label)}: {value}")


def _err(label: str, value):
    print(f"  {red('✗')} {bold(label)}: {value}")


def _post(path: str, body: dict, timeout: int = 60) -> dict:
    url = f"{BASE_URL}{path}"
    try:
        r = requests.post(url, json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        print(red(f"\nCannot connect to {BASE_URL}. Is the server running?"))
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        try:
            return {"_http_error": e.response.status_code, "detail": e.response.json()}
        except Exception:
            return {"_http_error": e.response.status_code, "detail": str(e)}
    except requests.exceptions.Timeout:
        return {"_timeout": True}


def _get(path: str) -> dict:
    url = f"{BASE_URL}{path}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        print(red(f"\nCannot connect to {BASE_URL}. Is the server running?"))
        sys.exit(1)
    except Exception as e:
        return {"error": str(e)}


def _bar(score: float, width: int = 30) -> str:
    filled = int(round(score * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.2f}"


def _preview(text: str, max_len: int = 80) -> str:
    t = text.strip().replace("\n", " ")
    return t if len(t) <= max_len else t[:max_len - 3] + "..."


def test_health():
    _section("HEALTH CHECK  GET /health")
    d = _get("/health")
    status = d.get("status", "unknown")
    fn = _ok if status == "healthy" else _warn
    fn("status", status)
    components = d.get("components", {})
    for k, v in components.items():
        st = v if isinstance(v, str) else v.get("status", v)
        (yellow if "degraded" in str(st) else green)(st)
        print(f"  {'  '}{dim(k)}: {st}")
    model_ready = d.get("model_files_complete", False)
    fn2 = _ok if model_ready else _warn
    fn2("model_files_complete", model_ready)
    return d


def test_predict(text: str):
    _section("PREDICT  POST /predict")
    t0 = time.time()
    d = _post("/predict", {"text": text})
    elapsed = time.time() - t0

    if "_http_error" in d or "detail" in d and "_http_error" not in d and d.get("detail"):
        err = d.get("detail") or d
        if isinstance(err, dict) and err.get("detail"):
            _err("error", err["detail"])
        else:
            _err("error", err)
        return d

    pred = d.get("prediction", "N/A")
    prob = d.get("fake_probability", d.get("probabilities", {}).get("fake", "N/A"))
    conf = d.get("confidence", "N/A")

    fn = _warn if str(pred).upper() == "FAKE" else _ok
    fn("prediction", bold(str(pred).upper()))
    if isinstance(prob, float):
        print(f"  {bold('fake_probability')}: {_bar(prob)}")
    if isinstance(conf, float):
        print(f"  {bold('confidence')}:       {_bar(conf)}")
    print(f"  {dim(f'elapsed: {elapsed:.2f}s')}")
    return d


def test_analyze(text: str):
    _section("ANALYZE  POST /analyze")
    t0 = time.time()
    d = _post("/analyze", {"text": text})
    elapsed = time.time() - t0

    if "_http_error" in d:
        _err("http_error", d["_http_error"])
        _err("detail", d.get("detail"))
        return d

    pred = d.get("prediction", "N/A")
    prob = d.get("fake_probability", 0.0)
    conf = d.get("confidence", 0.0)

    fn = _warn if str(pred).upper() == "FAKE" else _ok
    fn("prediction", bold(str(pred).upper()))
    print(f"  {bold('fake_probability')}: {_bar(float(prob))}")
    print(f"  {bold('confidence')}:       {_bar(float(conf))}")

    bias = d.get("bias", {})
    if bias:
        bs = float(bias.get("bias_score", 0))
        print(f"  {bold('bias_score')}:        {_bar(bs)}")
        mb = bias.get("media_bias", "")
        if mb:
            print(f"  {bold('media_bias')}:       {mb}")
        biased = bias.get("biased_tokens", [])
        if biased:
            print(f"  {bold('biased_tokens')}:    {', '.join(biased[:8])}")

    emotion = d.get("emotion", {})
    if emotion:
        dom = emotion.get("dominant_emotion", "N/A")
        scores = emotion.get("emotion_scores", {})
        print(f"  {bold('dominant_emotion')}: {dom}")
        if scores:
            top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  {bold('top emotions')}:     " + "  ".join(f"{k}={v:.3f}" for k, v in top3))

    narrative = d.get("narrative", {})
    if narrative:
        roles = narrative.get("roles", {})
        if roles:
            print(f"  {bold('narrative roles')}:  {list(roles.keys())[:5]}")

    expl = d.get("explainability", {})
    if expl:
        lime = expl.get("lime")
        if lime and isinstance(lime, dict):
            structured = lime.get("structured") or lime.get("important_features") or []
            if structured and isinstance(structured[0], dict):
                tokens = [t.get("token", t.get("word", "")) for t in structured[:6]]
            else:
                tokens = structured[:6]
            if tokens:
                print(f"  {bold('lime top tokens')}: {tokens}")

    cp = d.get("credibility_profile", {})
    if cp and isinstance(cp, dict):
        cs = cp.get("credibility_score")
        if cs is not None:
            print(f"  {bold('credibility_score')}: {_bar(float(cs))}")

    print(f"  {dim(f'elapsed: {elapsed:.2f}s')}")
    return d


def test_explain(text: str):
    _section("EXPLAIN  POST /explain")
    t0 = time.time()
    d = _post("/explain", {"text": text}, timeout=120)
    elapsed = time.time() - t0

    if "_http_error" in d:
        _err("http_error", d["_http_error"])
        _err("detail", d.get("detail"))
        return d

    if d.get("_timeout"):
        _warn("timeout", "request exceeded 120 s")
        return d

    src = d.get("predict_source", "N/A")
    pred_block = d.get("prediction", {})
    pred = pred_block.get("prediction", "N/A") if isinstance(pred_block, dict) else pred_block
    prob = pred_block.get("fake_probability", 0.0) if isinstance(pred_block, dict) else 0.0

    fn = _warn if str(pred).upper() == "FAKE" else _ok
    fn("prediction", bold(str(pred).upper()))
    print(f"  {bold('fake_probability')}: {_bar(float(prob))}")
    _ok("predict_source", src)

    expl = d.get("explainability", {})
    if expl:
        failures = expl.get("module_failures", [])
        if failures:
            _warn("module_failures", failures)
        else:
            _ok("module_failures", "none")

        qs = expl.get("explanation_quality_score")
        if qs is not None:
            print(f"  {bold('quality_score')}:     {_bar(float(qs))}")

        lime = expl.get("lime")
        if lime and isinstance(lime, dict):
            structured = lime.get("structured") or []
            if structured:
                tokens = [t.get("token", "") for t in structured[:6]]
                weights = [round(t.get("weight", 0), 3) for t in structured[:6]]
                print(f"  {bold('lime tokens')}:     {tokens}")
                print(f"  {bold('lime weights')}:    {weights}")

        agg = expl.get("aggregated")
        if agg and isinstance(agg, dict):
            structured = agg.get("structured") or []
            if structured:
                tokens = [t.get("token", "") for t in structured[:6]]
                print(f"  {bold('aggregated tokens')}: {tokens}")

        cm = expl.get("consistency_metrics")
        if cm and isinstance(cm, dict):
            for k, v in list(cm.items())[:4]:
                print(f"  {dim(k)}: {round(v, 4) if isinstance(v, float) else v}")

    agg_result = d.get("aggregation", {})
    if agg_result:
        scores = agg_result.get("scores", {})
        if scores:
            print()
            print(f"  {bold('── Aggregation Scores')}")
            for k, v in scores.items():
                if isinstance(v, (int, float)):
                    print(f"  {bold(k)}: {_bar(float(v))}")
                else:
                    print(f"  {bold(k)}: {v}")
        risks = agg_result.get("risks", {})
        if risks:
            print(f"  {bold('risks')}: {list(risks.keys())}")

    print(f"  {dim(f'elapsed: {elapsed:.2f}s')}")
    return d


def test_report(text: str):
    _section("REPORT  POST /report")
    t0 = time.time()
    d = _post("/report", {"text": text}, timeout=120)
    elapsed = time.time() - t0

    if "_http_error" in d:
        _err("http_error", d["_http_error"])
        _err("detail", d.get("detail"))
        return d

    if d.get("_timeout"):
        _warn("timeout", "request exceeded 120 s")
        return d

    credibility = d.get("credibility_score")
    if credibility is not None:
        print(f"  {bold('credibility_score')}: {_bar(float(credibility))}")

    verdict = d.get("verdict", d.get("overall_verdict", ""))
    if verdict:
        _ok("verdict", verdict)

    summary = d.get("summary", d.get("executive_summary", ""))
    if summary:
        wrapped = textwrap.fill(str(summary), width=66, initial_indent="    ", subsequent_indent="    ")
        print(f"  {bold('summary')}:\n{dim(wrapped)}")

    key_findings = d.get("key_findings", [])
    if key_findings:
        print(f"  {bold('key_findings')}:")
        for f in key_findings[:4]:
            print(f"    {dim('•')} {f}")

    rec = d.get("recommendations", [])
    if rec:
        print(f"  {bold('recommendations')}:")
        for r in (rec if isinstance(rec, list) else [rec])[:3]:
            print(f"    {dim('→')} {r}")

    print(f"  {dim(f'elapsed: {elapsed:.2f}s')}")
    return d


def run_all(text: str):
    print(f"\n{bold('TruthLens AI — API Test Runner')}")
    print(dim(SEPARATOR))
    print(f"  Input text: {cyan(_preview(text))}")

    test_health()
    test_predict(text)
    test_analyze(text)
    test_explain(text)
    test_report(text)

    print(f"\n{green(bold('Done.'))} All endpoints tested.\n")


ENDPOINT_MAP = {
    "health":  test_health,
    "predict": test_predict,
    "analyze": test_analyze,
    "explain": test_explain,
    "report":  test_report,
    "all":     run_all,
}

SAMPLE_TEXTS = [
    "Scientists discover new evidence supporting climate change based on peer-reviewed research from 50,000 participants.",
    "BREAKING: Government secretly adds mind control chemicals to tap water, leaked documents reveal shocking truth.",
    "Politicians spread fear and misinformation about the economy to manipulate vulnerable communities into panic.",
    "The stock market rose 1.2% today following better-than-expected jobs data released by the Labor Department.",
]


def main():
    parser = argparse.ArgumentParser(
        description="TruthLens AI API test script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--text", "-t", type=str, default=None,
                        help="Text to analyse (prompts if omitted)")
    parser.add_argument("--endpoint", "-e", type=str, default="all",
                        choices=list(ENDPOINT_MAP.keys()),
                        help="Which endpoint to test (default: all)")
    parser.add_argument("--sample", "-s", type=int, default=None,
                        choices=[1, 2, 3, 4],
                        help="Use a built-in sample text (1-4) instead of typing")
    args = parser.parse_args()

    if args.sample is not None:
        text = SAMPLE_TEXTS[args.sample - 1]
        print(f"\n{bold('Using sample text')} [{args.sample}]: {cyan(_preview(text))}")
    elif args.text:
        text = args.text
    else:
        print(f"\n{bold('TruthLens AI — API Test Runner')}")
        print(dim(SEPARATOR))
        print("Built-in samples:")
        for i, s in enumerate(SAMPLE_TEXTS, 1):
            print(f"  [{i}] {dim(_preview(s))}")
        print()
        choice = input("Enter 1-4 to use a sample, or paste your own text: ").strip()
        if choice in ("1", "2", "3", "4"):
            text = SAMPLE_TEXTS[int(choice) - 1]
        elif choice:
            text = choice
        else:
            text = SAMPLE_TEXTS[0]

    if not text.strip():
        print(red("Error: text cannot be empty."))
        sys.exit(1)

    fn = ENDPOINT_MAP[args.endpoint]
    if args.endpoint in ("health",):
        fn()
    elif args.endpoint == "all":
        fn(text)
    else:
        print(f"\n{bold('TruthLens AI — API Test Runner')}")
        print(dim(SEPARATOR))
        print(f"  Input text: {cyan(_preview(text))}")
        fn(text)
        print()


if __name__ == "__main__":
    main()
