# TruthLens AI — Frontend Design & Implementation Specification

---

## 1. System Summary

### What the model does
TruthLens AI is a multi-layer misinformation detection system. Given a piece of news text it runs 10+ NLP analysis pipelines in parallel — bias detection, emotion classification, narrative role extraction, rhetorical analysis, propaganda scoring, graph-based entity analysis, and LIME explainability — then aggregates all signals into a single credibility score and manipulation risk rating.

### What problem it solves
Readers and journalists cannot quickly assess whether an article is slanted, emotionally manipulative, or outright fake. TruthLens makes this judgment transparent and explainable by showing *which words* and *which linguistic patterns* drove the verdict.

### Target users
| User type | Goal |
|-----------|------|
| Journalists / fact-checkers | Quickly screen articles for red flags |
| Students / researchers | Understand bias and propaganda patterns |
| General public | Verify suspicious social-media claims |
| College demo audience | See an impressive live AI pipeline |

---

## 2. Input Handling

### Primary input form

| Field | Type | Rules |
|-------|------|-------|
| Article / claim text | `<textarea>` | Required · min 20 chars · max 5,000 chars |
| Analysis mode | Radio / tabs | `Quick` (analyze) · `Deep` (explain) · `Report` |

### Validation rules
```
- Cannot be empty
- Minimum 20 characters (show character counter)
- Maximum 5,000 characters (show remaining count + soft warning at 4,000)
- Strip leading/trailing whitespace before sending
- Detect and warn if input looks like a URL (suggest pasting the full article text instead)
```

### Example inputs (prefill buttons for demo)
```
[EXAMPLE 1 — Likely credible]
"Scientists discover new evidence supporting climate change based on
peer-reviewed research from 50,000 participants across 12 countries,
published in the journal Nature."

[EXAMPLE 2 — Sensational / likely fake]
"BREAKING: Government secretly adds mind-control chemicals to tap water!
Leaked documents reveal the shocking truth they don't want you to see!"

[EXAMPLE 3 — Politically charged]
"Politicians are spreading dangerous lies about vaccines to manipulate
vulnerable populations into fear and panic for electoral gain."
```

---

## 3. API Integration Layer

Base URL: `http://localhost:5000` (dev) · `https://<your-domain>` (prod)

---

### 3.1 `POST /analyze` — Full Analysis

**Request**
```json
{ "text": "string (20–5000 chars)" }
```

**Response shape** (abbreviated; all numeric fields are 0–1 floats)
```json
{
  "text": "preview string (100 chars)...",
  "prediction": "REAL | FAKE",
  "fake_probability": 0.73,
  "confidence": 0.91,

  "bias": {
    "bias_score": 0.0,
    "media_bias": "center | left | right | far-left | far-right",
    "biased_tokens": ["string"],
    "sentence_heatmap": [{ "sentence": "...", "bias_score": 0.0 }]
  },

  "emotion": {
    "dominant_emotion": "neutral | anger | fear | ...",
    "emotion_scores": { "neutral": 0.0, "anger": 0.0, "fear": 0.0, "...": 0.0 },
    "emotion_distribution": { "same keys as emotion_scores" }
  },

  "narrative": {
    "roles": { "hero_ratio": 0.0, "villain_ratio": 0.0, "victim_ratio": 0.0 },
    "conflict": { "conflict_intensity": 0.0, "polarization_ratio": 0.0 },
    "propagation": { "conflict_propagation_intensity": 0.0 },
    "temporal": { "urgency_language_ratio": 0.0 }
  },

  "framing": {
    "frame_conflict_score": 0.0,
    "frame_economic_score": 0.0,
    "frame_moral_score": 0.0,
    "frame_human_interest_score": 0.0,
    "frame_security_score": 0.0
  },

  "rhetoric": {
    "rhetorical_devices": {
      "rhetoric_exaggeration_score": 0.0,
      "rhetoric_fear_appeal_score": 0.0,
      "rhetoric_loaded_language_score": 0.0
    },
    "argument_structure": { "argument_complexity": 0.0 }
  },

  "propaganda_analysis": {
    "fear_propaganda_score": 0.0,
    "scapegoating_score": 0.0,
    "polarization_score": 0.0,
    "propaganda_intensity": 0.0
  },

  "credibility_profile": {
    "bias_score": 0.02,
    "bias": { "bias_score": 0.0 },
    "ideology": { "liberty_language_ratio": 0.11, "anti_elite_language_ratio": 0.11 },
    "metrics": { "bias_variance": 0.0, "ideology_entropy": 1.85 }
  },

  "graph_analysis": {
    "narrative_graph_metrics": {
      "graph_nodes": 4.0, "graph_edges": 6.0, "graph_density": 1.0, "graph_clustering": 1.0
    },
    "graph_features": { "narrative_graph_entropy": 1.79 },
    "temporal_graph": { "topic_shift_score": 0.0, "narrative_drift": 0.0 }
  },

  "explainability": {
    "lime": {
      "tokens": ["word1", "word2"],
      "importance": [0.36, 0.20],
      "structured": [{ "token": "word1", "importance": 0.36 }],
      "confidence": 0.99,
      "faithful": true
    },
    "emotion_explanation": {
      "tokens": ["word1"],
      "lexicon_intensity": [0.0],
      "fused_importance": [0.0],
      "sentence_scores": [{ "sentence": "...", "emotion_intensity": 0.0 }]
    }
  }
}
```

**Error handling**
| HTTP code | Cause | UI message |
|-----------|-------|------------|
| 422 | Text too short/missing | Show inline validation error |
| 503 | ML model not trained (uses fallback) | Show info banner: "Running in heuristic mode — predictions use lexicon scoring" |
| 500 | Internal server error | Show error card with retry button |
| Network error | Server down | Show "Cannot reach server" toast |

---

### 3.2 `POST /explain` — Deep Explainability + Aggregation

**Request** — same as `/analyze`

**Response shape**
```json
{
  "text": "preview...",
  "predict_source": "heuristic_fallback | model",
  "prediction": {
    "prediction": "REAL | FAKE",
    "fake_probability": 0.1,
    "confidence": 0.9,
    "label": "REAL"
  },

  "explainability": {
    "lime": {
      "structured": [{ "token": "dangerous", "importance": 3.6e-20 }],
      "confidence": 0.99,
      "faithful": true
    },
    "aggregated": {
      "structured": [{ "token": "dangerous", "importance": 3.6e-09 }],
      "method_weights": { "shap": 0.35, "ig": 0.25, "attn": 0.2, "lime": 0.1, "graph": 0.1 },
      "confidence_score": 1.0,
      "agreement_score": 0.0
    },
    "explanation_metrics": {
      "faithfulness": 0.0,
      "comprehensiveness": 0.0,
      "insertion_score": 0.7,
      "normalized": { "insertion_score": 0.85 },
      "overall_score": 0.57
    },
    "explanation_quality_score": 0.57,
    "module_failures": [],
    "metadata": {
      "latency_ms": { "lime": 21.4, "aggregation": 0.16 },
      "modules": { "lime": true, "graph_explainer": true }
    }
  },

  "aggregation": {
    "scores": {
      "manipulation_risk": 0.0,
      "credibility_score": 0.0,
      "final_score": 0.5
    },
    "risks": {
      "manipulation_risk": null,
      "credibility_level": null,
      "overall_truthlens_rating": null
    },
    "explanations": {
      "sections": {
        "bias": { "top_features": ["bias_score"], "section_score": 0.0 },
        "emotion": { "top_features": ["neutral", "fear"], "section_score": 0.0 }
      }
    },
    "analysis_modules": {
      "weights": { "bias": 0.4, "emotion": 0.3, "narrative": 0.2 }
    }
  }
}
```

---

### 3.3 `POST /report` — Summary Report

**Request** — same as `/analyze`

**Response shape**
```json
{
  "article_summary": {
    "title": null,
    "source": null,
    "word_count": 15,
    "analyzed_at": "2026-05-01T11-44-23Z"
  },
  "bias_analysis": {},
  "emotion_analysis": {},
  "narrative_structure": {},
  "credibility_score": null
}
```

---

### 3.4 `GET /health` — Server Status

**Response**
```json
{ "status": "healthy | degraded", "model_files_complete": false, "components": { ... } }
```

---

## 4. Output Visualization

### 4.1 Verdict Card (top of results)
```
┌─────────────────────────────────────────────────┐
│  ✅ LIKELY REAL          Confidence: 90%         │
│  Fake probability ████████░░░░░░░░░░  0.10       │
│  Final credibility ██████████████░░░░  0.70      │
│  Manipulation risk ██░░░░░░░░░░░░░░░  0.10       │
└─────────────────────────────────────────────────┘
```

- Background: green for REAL, red for FAKE, yellow for borderline (0.4–0.6)
- Three horizontal progress bars with animated fill on load
- Badge showing `predict_source` ("Heuristic mode" if no trained model)

---

### 4.2 Emotion Radar Chart
Display `emotion_scores` as a **radar (spider) chart**.

```
Axes: neutral · anger · fear · optimism · disapproval ·
      admiration · annoyance · curiosity · love · gratitude
```
Use Recharts `RadarChart`. Highlight the `dominant_emotion` axis in a contrasting color.

---

### 4.3 Bias Sentence Heatmap
Render `bias.sentence_heatmap` as a **color-highlighted paragraph**.

- Each sentence gets a background opacity proportional to its `bias_score`
- 0.0 → white · 1.0 → red
- Tooltip on hover shows the numeric score
- Biased tokens (`bias.biased_tokens`) get an underline + tooltip "Biased word"

```jsx
// Color scale
const getBiasColor = (score) =>
  `rgba(239, 68, 68, ${Math.min(score * 2, 0.8)})`;
```

---

### 4.4 LIME Token Importance Bar Chart
Display `explainability.lime.structured` as a **horizontal bar chart**.

- Bars sorted by importance descending
- Bar color: blue (positive contribution → toward FAKE) / grey (neutral)
- On hover: tooltip with exact importance value
- Title: "Which words most influenced the prediction?"

```
dangerous    ████████████████  0.036
Politicians  █████████         0.020
fear         █████████         0.020
spreading    ████              0.009
vaccines     ██                0.005
```

---

### 4.5 Aggregated Explainability Heatmap
Display `explainability.aggregated.structured` as **highlighted text** — render the original article with each word's color intensity driven by its aggregated importance score.

- Same color scale as bias heatmap but blue-to-purple palette
- Toggle button to switch between LIME view and Aggregated view

---

### 4.6 Framing Radar Chart
Display `framing` object keys as a radar chart with axes:
- Conflict · Economic · Moral · Human Interest · Security · Dominance

---

### 4.7 Propaganda Gauge Panel
Display `propaganda_analysis` as four **gauge meters** (semicircular progress arcs):

| Gauge | Field |
|-------|-------|
| Fear Propaganda | `fear_propaganda_score` |
| Scapegoating | `scapegoating_score` |
| Polarization | `polarization_score` |
| Overall Intensity | `propaganda_intensity` |

Color scale: green (0–0.3) → yellow (0.3–0.6) → red (0.6–1.0)

---

### 4.8 Narrative Role Donut Chart
Display `narrative.roles` hero/villain/victim ratios as a **donut chart**.

```
hero_ratio + villain_ratio + victim_ratio → 3-segment donut
Colors: green (hero) · red (villain) · orange (victim)
```

---

### 4.9 Explainability Quality Score Card
Display `explainability.explanation_quality_score` (0–1) as a **circular progress ring** with letter grade:

| Score | Grade |
|-------|-------|
| 0.8–1.0 | A |
| 0.6–0.8 | B |
| 0.4–0.6 | C |
| < 0.4 | D |

Also show the normalized metrics table:
```
Faithfulness    0.50   ████████░░
Comprehensiveness 0.50 ████████░░
Insertion score 0.85   █████████████░
```

---

### 4.10 Aggregation Weights Donut
Display `aggregation.analysis_modules.weights` as a donut/pie showing how much each module contributed to the final score:
`bias (40%) · emotion (30%) · narrative (20%) · discourse (10%)`

---

### 4.11 Method Weights (Explainability)
Display `aggregated.method_weights` as a horizontal stacked bar:
`SHAP 35% | IG 25% | Attention 20% | LIME 10% | Graph 10%`

---

## 5. Explainability UI

### Token-level highlights (interactive)
Render the article text with each token wrapped in a `<span>`:

```jsx
<span
  style={{ backgroundColor: `rgba(59,130,246,${importance * 10})` }}
  title={`Importance: ${importance.toFixed(4)}`}
>
  {token}
</span>
```

### Toggle panel
```
[LIME]  [Aggregated]  [Emotion]
```
Each tab swaps which importance scores drive the highlighting.

### Faithfulness badge
Show `lime.faithful` as a ✅ / ❌ badge next to the explanation:
- ✅ Faithful — explanation matches model's internal reasoning
- ❌ Not faithful — explanation is approximate

### Latency breakdown (metadata)
Show module timings from `metadata.latency_ms` as a small table at the bottom of the explainability panel:
```
LIME          21.4 ms
Graph         0.08 ms
Aggregation   0.16 ms
```

---

## 6. User Flow

```
1. USER OPENS APP
   → Homepage loads with InputForm + 3 example prefill buttons
   → Health check runs silently; if degraded, show "Heuristic mode" banner

2. USER ENTERS TEXT
   → Real-time char counter updates
   → Validation fires on blur (min length, max length)

3. USER CLICKS ANALYZE
   → Button shows spinner + "Analyzing..."
   → API call fires to POST /analyze (and optionally POST /explain in parallel)
   → Skeleton loaders appear for each result card

4. RESULTS LOAD (< 500 ms typical)
   → Verdict card animates in first (most important)
   → Charts render with animated fills (300ms ease-in)
   → Token heatmap highlights

5. USER EXPLORES
   → Tabs switch between analysis sections
   → Hover tooltips on all chart elements
   → "Deep Explain" button triggers POST /explain if not already fetched

6. ERROR STATES
   → Network error → red toast + retry button
   → 422 validation → inline form error (no API call made)
   → 503 model missing → yellow banner, results still show with heuristic label
   → 500 server error → error card with "Try again" button

7. USER RESETS
   → "Analyze another" button clears results and scrolls to input
```

---

## 7. Pages & Components

### Pages
```
/           → Home page (input + hero)
/results    → Full results page (all panels)
/dashboard  → (optional) History of past analyses (localStorage)
```

### Component tree
```
App
├── Header
│   ├── Logo ("TruthLens AI")
│   ├── NavLinks (Home · About · GitHub)
│   └── StatusBadge (server health)
│
├── HomePage
│   ├── HeroSection (tagline + animated graphic)
│   ├── InputForm
│   │   ├── TextArea (article text)
│   │   ├── CharCounter
│   │   ├── ModeToggle (Quick / Deep / Report)
│   │   ├── ExampleButtons (3 prefill examples)
│   │   └── SubmitButton (with loading spinner)
│   └── HeuristicBanner (shown when model not trained)
│
├── ResultsPage
│   ├── VerdictCard
│   │   ├── PredictionBadge (REAL/FAKE)
│   │   ├── ProgressBar (fake_probability)
│   │   ├── ProgressBar (confidence)
│   │   └── ScoreSummary (credibility + manipulation)
│   │
│   ├── TabPanel [Bias · Emotion · Narrative · Rhetoric · Propaganda · Explainability · Graph]
│   │
│   ├── BiasPanel
│   │   ├── SentenceHeatmap
│   │   ├── BiasScoreGauge
│   │   └── MediaBiasBadge
│   │
│   ├── EmotionPanel
│   │   ├── RadarChart (emotion_scores)
│   │   ├── DominantEmotionBadge
│   │   └── IntensityGauge
│   │
│   ├── NarrativePanel
│   │   ├── RoleDonutChart (hero/villain/victim)
│   │   ├── ConflictMetrics (key numbers)
│   │   └── TemporalMetrics
│   │
│   ├── RhetoricPanel
│   │   ├── FramingRadar
│   │   └── RhetoricScoreTable
│   │
│   ├── PropagandaPanel
│   │   └── GaugeGrid (4 gauges)
│   │
│   ├── ExplainabilityPanel
│   │   ├── TokenHighlighter (article with coloured spans)
│   │   ├── MethodToggle (LIME / Aggregated / Emotion)
│   │   ├── ImportanceBarChart
│   │   ├── QualityScoreRing
│   │   ├── MethodWeightsBar
│   │   └── LatencyTable
│   │
│   └── AggregationPanel
│       ├── ModuleWeightsPie
│       ├── SectionScoreTable
│       └── RiskBadges
│
└── Footer
```

---

## 8. Tech Stack Recommendation

| Category | Choice | Reason |
|----------|--------|--------|
| Framework | **React 18 + Vite** | Fast HMR, zero config, perfect for demo |
| Styling | **Tailwind CSS** | Utility-first, no custom CSS needed |
| UI components | **shadcn/ui** | Polished, accessible, Tailwind-based |
| Charts | **Recharts** | Simple API, React-native, radar + bar + pie all built-in |
| Icons | **Lucide React** | Clean, consistent |
| HTTP | **fetch / axios** | Lightweight; axios for automatic JSON + interceptors |
| State | **React useState + useReducer** | Simple enough; no Redux needed for a demo |
| Routing | **React Router v6** | Home → Results page navigation |
| Animations | **Framer Motion** | Smooth card reveals and progress bar fills |

### Install command
```bash
npm create vite@latest truthlens-frontend -- --template react
cd truthlens-frontend
npm install recharts framer-motion axios react-router-dom lucide-react
npx shadcn-ui@latest init
```

---

## 9. Performance Considerations

### API latency
- `/analyze` typically responds in **< 100 ms** (heuristic mode)
- `/explain` typically responds in **< 200 ms**
- Show skeleton loaders immediately on submit — never show a blank screen

### Parallel requests
```js
// Fire analyze and explain simultaneously
const [analyzeResult, explainResult] = await Promise.allSettled([
  fetch('/analyze', { method: 'POST', body }),
  fetch('/explain', { method: 'POST', body }),
]);
```

### Caching
- Cache the last result in `sessionStorage` keyed by text hash
- If user submits the same text twice, return cached result instantly

### Large responses
- `/analyze` returns ~150 keys; parse once and pass slices to each panel via props
- Never re-parse JSON in child components

### Chart rendering
- Lazy-load heavy chart panels (React.lazy + Suspense) — only render when tab is active
- Use `useMemo` for transformed chart data (sorted importance arrays, donut segments)

---

## 10. Sample UI Layout

```
┌──────────────────────────────────────────────────────────┐
│  🔍 TruthLens AI                          [Healthy ✅]    │  ← Header
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│   Detect Misinformation. Understand Why.                 │  ← Hero
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Paste a news article or claim here...              │  │  ← TextArea
│  │                                                    │  │
│  │                                          234/5000  │  │  ← char counter
│  └────────────────────────────────────────────────────┘  │
│  [Example 1]  [Example 2]  [Example 3]                   │  ← prefill btns
│  Mode: ( Quick )  ( Deep )  ( Report )                   │
│  [       Analyze Now ▶       ]                           │  ← submit
└──────────────────────────────────────────────────────────┘

── RESULTS ────────────────────────────────────────────────

┌─── Verdict ──────────────────────────────────────────────┐
│  ✅ LIKELY REAL                Confidence  90%           │
│  Fake probability  ███░░░░░░  0.10                       │
│  Manipulation risk ░░░░░░░░░  0.00                       │
│  Credibility score ░░░░░░░░░  0.00  (model not trained)  │
└──────────────────────────────────────────────────────────┘

┌─── Tabs ─────────────────────────────────────────────────┐
│ [Bias] [Emotion] [Narrative] [Rhetoric] [Propaganda]     │
│ [Explainability] [Graph] [Aggregation]                   │
└──────────────────────────────────────────────────────────┘

┌─── Active Tab Content ───────────────────────────────────┐
│  Explainability                                          │
│                                                          │
│  Politicians are spreading [dangerous] lies about        │  ← highlighted
│  [vaccines] to manipulate [vulnerable] populations       │
│  into [fear] and [panic].                                │
│                                                          │
│  [LIME ●]  [Aggregated]  [Emotion]                       │  ← toggle
│                                                          │
│  Top contributing words:                                 │
│  dangerous    ████████████████  0.036                    │
│  Politicians  █████████         0.020                    │
│  fear         █████████         0.020                    │
│                                                          │
│  Quality score: ◉ 0.57  Grade: C                        │
│  Faithfulness: ✅  Modules: LIME ✅  Graph ✅            │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  © 2026 TruthLens AI · Powered by RoBERTa + LIME         │  ← Footer
└──────────────────────────────────────────────────────────┘
```

---

## 11. Example Code

### 11.1 API service layer (`src/api/truthlens.js`)
```js
const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:5000';

const post = async (path, text) => {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw Object.assign(new Error(err.detail ?? 'Request failed'), { status: res.status });
  }
  return res.json();
};

export const analyze  = (text) => post('/analyze', text);
export const explain  = (text) => post('/explain', text);
export const report   = (text) => post('/report', text);
export const health   = ()     => fetch(`${BASE}/health`).then(r => r.json());
```

---

### 11.2 Main state hook (`src/hooks/useAnalysis.js`)
```js
import { useState, useCallback } from 'react';
import { analyze, explain } from '../api/truthlens';

export function useAnalysis() {
  const [state, setState] = useState({
    loading: false,
    error: null,
    analyzeResult: null,
    explainResult: null,
  });

  const run = useCallback(async (text, mode = 'quick') => {
    setState(s => ({ ...s, loading: true, error: null }));
    try {
      const calls = [analyze(text)];
      if (mode === 'deep') calls.push(explain(text));
      const [analyzeResult, explainResult] = await Promise.all(calls);
      setState({ loading: false, error: null, analyzeResult, explainResult: explainResult ?? null });
    } catch (err) {
      setState(s => ({ ...s, loading: false, error: err.message }));
    }
  }, []);

  return { ...state, run };
}
```

---

### 11.3 InputForm component (`src/components/InputForm.jsx`)
```jsx
import { useState } from 'react';

const EXAMPLES = [
  'Scientists discover new evidence supporting climate change based on peer-reviewed research from 50,000 participants.',
  'BREAKING: Government secretly adds mind-control chemicals to tap water! Leaked documents reveal shocking truth!',
  'Politicians are spreading dangerous lies about vaccines to manipulate vulnerable populations into fear and panic.',
];

export default function InputForm({ onSubmit, loading }) {
  const [text, setText] = useState('');
  const [error, setError] = useState('');

  const validate = (t) => {
    if (!t.trim()) return 'Please enter some text.';
    if (t.trim().length < 20) return 'Text must be at least 20 characters.';
    if (t.length > 5000) return 'Text must be under 5,000 characters.';
    return '';
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const err = validate(text);
    if (err) { setError(err); return; }
    setError('');
    onSubmit(text);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <textarea
        value={text}
        onChange={e => { setText(e.target.value); setError(''); }}
        rows={6}
        placeholder="Paste a news article or claim here..."
        className="w-full p-3 border rounded-lg resize-y font-medium text-sm focus:ring-2 focus:ring-blue-500"
        maxLength={5000}
      />
      <div className="flex justify-between text-xs text-gray-400">
        {error && <span className="text-red-500">{error}</span>}
        <span className="ml-auto">{text.length}/5000</span>
      </div>

      <div className="flex gap-2 flex-wrap">
        {EXAMPLES.map((ex, i) => (
          <button key={i} type="button"
            onClick={() => { setText(ex); setError(''); }}
            className="text-xs px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-full"
          >
            Example {i + 1}
          </button>
        ))}
      </div>

      <button type="submit" disabled={loading}
        className="w-full py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold disabled:opacity-50"
      >
        {loading ? 'Analyzing...' : 'Analyze Now'}
      </button>
    </form>
  );
}
```

---

### 11.4 VerdictCard component (`src/components/VerdictCard.jsx`)
```jsx
import { motion } from 'framer-motion';

const ProgressBar = ({ label, value, color = 'blue' }) => (
  <div className="mb-2">
    <div className="flex justify-between text-xs mb-1">
      <span className="text-gray-600">{label}</span>
      <span className="font-mono font-bold">{(value ?? 0).toFixed(2)}</span>
    </div>
    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${(value ?? 0) * 100}%` }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
        className={`h-full bg-${color}-500 rounded-full`}
      />
    </div>
  </div>
);

export default function VerdictCard({ result }) {
  const { prediction, fake_probability, confidence } = result;
  const isFake = prediction === 'FAKE';
  const borderColor = isFake ? 'border-red-500' : 'border-green-500';
  const bg = isFake ? 'bg-red-50' : 'bg-green-50';
  const icon = isFake ? '⚠️' : '✅';

  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`border-2 ${borderColor} ${bg} rounded-xl p-5 shadow-md`}
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold">
          {icon} {isFake ? 'LIKELY FAKE' : 'LIKELY REAL'}
        </h2>
        <span className="text-sm text-gray-500 bg-white px-2 py-1 rounded-full border">
          Confidence {((confidence ?? 0) * 100).toFixed(0)}%
        </span>
      </div>
      <ProgressBar label="Fake probability" value={fake_probability} color={isFake ? 'red' : 'green'} />
      <ProgressBar label="Confidence"       value={confidence}       color="blue" />
    </motion.div>
  );
}
```

---

### 11.5 Token Heatmap component (`src/components/TokenHeatmap.jsx`)
```jsx
const getColor = (importance, max) => {
  const ratio = max > 0 ? Math.min(importance / max, 1) : 0;
  return `rgba(59, 130, 246, ${ratio * 0.7})`;   // blue scale
};

export default function TokenHeatmap({ structured }) {
  const max = Math.max(...structured.map(t => Math.abs(t.importance)), 1e-10);

  return (
    <div className="leading-8 text-base font-medium">
      {structured.map(({ token, importance }, i) => (
        <span
          key={i}
          style={{ backgroundColor: getColor(Math.abs(importance), max), borderRadius: 3, padding: '1px 2px' }}
          title={`Importance: ${importance.toExponential(2)}`}
          className="mr-1 cursor-help transition-all hover:opacity-80"
        >
          {token}
        </span>
      ))}
    </div>
  );
}
```

---

### 11.6 LIME Bar Chart (`src/components/LimeBarChart.jsx`)
```jsx
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export default function LimeBarChart({ structured }) {
  const max = Math.max(...structured.map(t => Math.abs(t.importance)), 1e-10);
  const data = [...structured]
    .sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance))
    .slice(0, 10)
    .map(t => ({ token: t.token, importance: Math.abs(t.importance) / max }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} layout="vertical" margin={{ left: 80 }}>
        <XAxis type="number" domain={[0, 1]} tickFormatter={v => v.toFixed(1)} />
        <YAxis type="category" dataKey="token" width={75} tick={{ fontSize: 13 }} />
        <Tooltip formatter={(v) => v.toFixed(4)} />
        <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
```

---

### 11.7 Emotion Radar Chart (`src/components/EmotionRadar.jsx`)
```jsx
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer } from 'recharts';

export default function EmotionRadar({ emotionScores }) {
  const data = Object.entries(emotionScores).map(([emotion, value]) => ({
    emotion,
    score: value,
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <RadarChart data={data}>
        <PolarGrid />
        <PolarAngleAxis dataKey="emotion" tick={{ fontSize: 11 }} />
        <Radar name="Emotion" dataKey="score" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.4} />
      </RadarChart>
    </ResponsiveContainer>
  );
}
```

---

## 12. Demo-Friendly Design Tips

### Make it fast
- Show results immediately — even if deep explain hasn't loaded yet
- Use the `quick` mode (only `/analyze`) by default; offer `deep` as an option
- Total demo time: type → click → see verdict in < 1 second

### Make it impressive
- Animate everything: bars fill left-to-right on load (Framer Motion)
- The token heatmap is the most "wow" visual — put it front and center
- Show the LIME bar chart with a title like "Why did the AI decide this?"
- Use the propaganda gauges for dramatic effect on a fake-news example

### Make it clear
- Show the predict source badge ("Heuristic mode") — judges appreciate honesty
- Add a one-line tooltip on every chart axis explaining what the score means
- Include a "What does this mean?" expandable panel below the verdict

### Demo sequence (recommended)
```
1. Open app → paste EXAMPLE 2 (fake/sensational headline)
2. Click "Analyze Now" → show VerdictCard (FAKE, high manipulation)
3. Switch to Explainability tab → show heatmap + LIME bars
4. Switch to Emotion tab → show radar chart
5. Click "Deep Explain" → show aggregated heatmap + quality score
6. Paste EXAMPLE 1 (credible science news) → contrast the two results
```

### Color scheme
| Meaning | Color |
|---------|-------|
| REAL / safe | `green-500` (#22c55e) |
| FAKE / danger | `red-500` (#ef4444) |
| Borderline / warning | `yellow-500` (#eab308) |
| LIME importance | `blue-500` (#3b82f6) |
| Emotion / explainability | `violet-500` (#8b5cf6) |
| Propaganda | `orange-500` (#f97316) |
| Neutral / bg | `gray-50 / gray-100` |

### Responsive layout
- Desktop: two-column grid (verdict left, charts right)
- Tablet: single column, tabs stacked
- Mobile: hide charts, show only verdict + key numbers

---

*Generated from live API response inspection against TruthLens AI v2.0 running on port 5000.*
