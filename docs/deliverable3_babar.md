# Deliverable 3 — Treatment Advisory and Local Deployment

## Scope

M. Babar Shahzad (24L-3047): treatment retrieval, confidence-aware advisory panels,
local startup automation, integration testing, Sections 11.4–11.5, citation audit
and prediction/report figures. The original planning document specified Streamlit;
the implementation integrates with the team's current Flask dashboard instead.

## Module map

| File | Responsibility |
|---|---|
| `knowledge_base/kb_service.py` | Load and validate the 38-class KB; exact-class, independent-copy retrieval |
| `app/treatment_advisor.py` | Server-side 60% policy; healthy/uncertain/disease advisory view models |
| `app/static/advisory.js` | Render only gated advice, safety messages, sources and review dates |
| `app/model_engine.py` | Connect the shared inference/Grad-CAM modules to Flask; serialize model/hook access |
| `app/app.py` | Input validation, HTTP errors, advisory integration and truthful readiness |
| `scripts/run_app.py`, `.bat`, `.sh` | Preflight, local serving and browser startup |
| `tests/test_treatment_kb.py` | All-class retrieval, corruption, immutability and threshold tests |
| `tests/test_app_integration.py` | HTTP safety boundaries and a real-checkpoint/image smoke test |
| `tests/test_delivery_tools.py` | Launcher diagnostics, report audit and safe compilation behavior |
| `scripts/generate_advisory_figures.py` | Real prediction/Grad-CAM examples and workflow diagram |
| `scripts/check_app_browser.py` | Optional real Chromium upload/render/mobile smoke test |
| `scripts/audit_report.py` | Structural citations, references, figures and KB source-registry checks |

The existing shared `src/inference.py` now rejects missing explicit checkpoints and
class-order mismatches and retains unrounded probabilities. Its structural-test fixture
uses random weights offline, avoiding an unnecessary ImageNet download. Production
Flask startup always requires the trained checkpoint.

## Launch

1. Install `requirements.txt` into `venv` or `.venv`.
2. Download `runs/baseline/best.pt` with `python scripts/download_weights.py` if absent.
3. Run `scripts\run_app.bat` on Windows or `bash scripts/run_app.sh` on POSIX/Git Bash.
4. Open `http://127.0.0.1:8501` if the browser does not open automatically.

Launchers resolve their own repository root and do not need a particular caller working
directory. Weights and packages are never silently installed. `--check` validates startup;
`--no-browser` suppresses opening a browser; `--port 8502` selects an alternate port.
The direct command `python app/app.py` still uses port 5000. Both serve only loopback,
with debug mode disabled. The Flask/Werkzeug local server is for the course demonstration,
not a public production hosting configuration.

## Advisory policy

Confidence is a **percentage from 0 to 100**, using the original 38-class softmax.
Crop selection changes ranking only; it does not renormalize scores. Numeric threshold
comparisons use unrounded values, not the displayed one-decimal percentage.

| Condition | Status | Product advice |
|---|---|---|
| Confidence <60%, including a tentative healthy class | `uncertain` | Chemical/organic instructions withheld; conservative scouting and photo-retaking guidance |
| Confidence >=60%, healthy class | `healthy` | No disease-control products indicated; continue monitoring |
| Confidence >=60%, disease class | `disease_detected` | Reviewed source-backed text, subject to label/registration/extension-officer verification |

The low-confidence warning is: **“Ambiguous symptoms detected. Verify in person before
applying chemical treatments.”** Both the top-level `treatment` and `advisory.treatment`
contain the gated version. The response never carries a second ungated product recipe.
Immediate-action and prevention fields are also replaced below the threshold because
some curated entries include spraying instructions in those fields.

No universal numeric dose is invented. The existing reviewed KB deliberately defers
rates to registered product labels. PPE, re-entry and pre-harvest intervals and local
crop/product registration remain the user's responsibility with a local agronomist.
English/Urdu controls and warnings are available; reviewed agronomic text remains in
English, explicitly labelled, rather than using unreviewed translated prescriptions.

## HTTP contract

`POST /api/predict` accepts `{ "image": "data:image/jpeg;base64,...", "crop": "all", "alpha": 0.55 }`
or a multipart `image` file. `crop` is optional and `alpha` must be finite in `[0,1]`.
The response includes the predicted class, top candidates, original softmax confidence,
Grad-CAM data URL and the advisory view model (status, threshold, warnings, label notice,
source-backed treatment and disclaimer). Confidence is not disease severity.

| Result | HTTP status / behavior |
|---|---|
| Invalid JSON/base64/image/crop/opacity | 400 with a readable error |
| Request >10 MB | 413 |
| Missing/incompatible model or unavailable KB/mapping | 503; no diagnosis or treatment |
| Valid diagnosis, failed Grad-CAM | 200; null overlay and explicit `gradcam_error` |
| Standalone failed `/api/gradcam` | 503; no substitute overlay |
| Missing real `/api/viva-sample/<class>` | 404; no procedural specimen |
| Ready `/api/health` | 200; `model_ready` and `kb_ready` true |
| Unready `/api/health` | 503 with truthful readiness flags |

JPEG/PNG/WebP images require at least 32 pixels per side and at most 20 megapixels.
Browser source files are capped at 7 MB; the browser preserves aspect ratio and limits
the longest side to 1024 pixels. The backend uses the shared resize/center-crop/ImageNet
normalization pipeline. Heatmaps are overlaid on the exact 224×224 center-cropped model
view so that a non-square photograph does not misregister the attention map. Model/hook
access is serialized within each process.

## Verification and artifacts

```bash
python -m pytest -q
python data_scripts/validate_treatments.py --require-complete
python scripts/run_app.py --check
python scripts/generate_advisory_figures.py
python scripts/audit_report.py --out results/advisory_demo/report_audit.json
```

The example generator uses the first manifest row for fixed lab disease, lab healthy and
field disease classes. It records reference labels, actual predictions, advisory outputs
and checkpoint SHA-256 in `results/advisory_demo/predictions.json`. These illustrative
forward passes do not rerun or overwrite the benchmark evaluation.

Optional browser verification (installed Chromium browser plus development-only
`websocket-client`):

```bash
python scripts/check_app_browser.py --screenshot reports/figures/advisory_dashboard.png
```

It uses a temporary browser profile and a real validation image, checks upload/diagnosis,
Grad-CAM, source rendering, language switching and mobile-width overflow, and closes its
own browser/server. A clearly labelled boundary fixture tests warning rendering before
restoring the real diagnosis; screenshots are captured from the real result only.

Compile the report locally with an installed `pdflatex` or Tectonic:

```bash
python scripts/compile_report.py
```

If no local engine is available, `python scripts/compile_report.py --online` explicitly
permits the repository's existing LaTeXOnline service. Only report TeX and referenced
figures are uploaded, not source code, datasets, weights, credentials or `HANDOFF.md`.
A failed build does not replace a previously valid PDF. Original high-resolution
figures remain committed even when the online compiler requires compressed images.

The structural citation audit checks undefined/duplicate citation keys, cross-references,
figure paths and whether treatment-source URLs occur in `knowledge_base/SOURCES.md`.
It is **not** a fresh live-link or agronomic-content verification. The current registry
includes one uncited historical source; it is recorded as informational, not treated as
a missing citation. Original source access/review dates are not fabricated or refreshed.

## Viva walkthrough

1. Start the app using the launcher and explain dependency/checkpoint/KB preflight.
2. Upload a real diseased leaf; show the exact class lookup and sourced control panels.
3. Show a healthy result: no disease-control products are recommended.
4. Explain `<60` versus `>=60` in `build_advisory`, and demonstrate the tested warning policy.
5. Explain why organic sprays and product-bearing immediate actions are gated too.
6. Show sources, review date, disclaimer and product-label dosage precautions.
7. Explain why the app refuses missing model weights instead of returning a demo diagnosis.
8. Distinguish softmax confidence, visual attention and disease severity.

**Limitations to explain:** lab test accuracy is 99.71% (4,768 images), while field test
accuracy is 24.73% (93 images). A confidently wrong field prediction can pass the gate;
the threshold is neither calibrated certainty nor an unknown-image detector. Grad-CAM
is not lesion segmentation or proof of disease. Local agronomic verification remains
necessary even for confident output.

## Local verification

Verified on 12 November 2026:

- Full pytest suite: **216 passed** (including the actual checkpoint/image smoke test).
- KB validation: all **38 classes**, no missing entries or schema errors.
- Windows `.bat` and Git Bash `.sh` launch preflight: passed.
- Real Chrome smoke test: upload, prediction, heatmap, sources, language switch,
  confidence-warning fixture and 390-pixel mobile layout passed.
- Structural report audit: no undefined citations/references, duplicate labels or missing
  figures; all 53 KB source URLs are registered. Live source pages were not reverified.
- Consolidated report: successfully compiled with the existing online service after
  adaptive compression; original high-resolution figures are retained in the repository.

Before submission, the team should confirm actual GitHub commit/review requirements
and perform its viva rehearsal. Windows and Git Bash launchers can be checked on this
machine; native macOS/Linux behavior still requires a machine running those platforms.
