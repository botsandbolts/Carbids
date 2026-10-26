# CarBids Studio 🏁

An auction-sale counterpart to [Pricing](https://github.com/botsandbolts/Pricing): real reported auction sale targets, MiniLM description regression, LightGBM, SHAP, a humorous business-friendly GUI and an explicit statistical bridge between markets.

**Reported sold price is not verified payment settlement.** This project accepts provider/platform-reported completed-sale amounts. It does not certify funds cleared, infer unseen buyer fees or equate wholesale auction outcomes with retail transactions.

## What works now

- A real trained tabular baseline on **550,291 distinct vehicles**, using the complete eligible public wholesale dataset, not a percentage pilot.
- Nine GUI workspaces, including the default structural Pricing + CarBids comparison: valuation, explainable ML, auction explorer, model report card, import/training, bridge to Pricing, Census context, and plain-English/technical learning.
- Native LightGBM TreeSHAP, additive reconstruction checks, source/year warnings and a mileage scenario; explanations concern model behaviour, not causation.
- Batch CSV valuation, valuation/evaluation downloads, isolated export imports, an optional REST API, Codespaces startup and CI checks.
- Full MiniLM-L6 regression → three-fold OOF `predicted_text_price` → training-only Isolation Forest auto → Huber LightGBM workflow, with atomic optimizer and fold checkpoints.
- Self-contained Kaggle notebook, eight-hour session budget and saved-output recovery.
- Auction auxiliary-feature export with a conservative training-VIN overlap index; weighted conformal quantile and effective-sample-size primitives.

**Not trained:** the text-enhanced model and calibrated retail-sale conversion. The wholesale dataset contains no long descriptions. The app refuses to turn tabular names into pretend seller text and refuses to load an asking-price bundle.

## Actual held-out assessment

| Measure | Result |
|---|---:|
| Train / validation / test | 384,271 / 82,393 / 83,627 |
| Test MAE | **$1,767** |
| Test RMSE | **$3,455** |
| Median absolute percentage error | **9.34%** |
| Test R² | **0.8905** |
| Indicative 80% band's test coverage | **82.54%** |
| Selected target / Huber alpha / rounds | log1p / 0.9 / 700 |
| Deployment refit | all 550,291 eligible vehicles |

Assessment is chronological and vehicle-disjoint. Entire equal-date groups stay in one partition. Validation MAE selects the setting; the untouched test is evaluated before refitting all eligible rows. These scores describe the assessment model, not an independent evaluation of the final refit. This baseline has no text feature and no Isolation Forest. Full grid and metrics: `models/tabular/evaluation.json`.

## Run the studio

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The shipped compressed LightGBM model loads without PyTorch. For the enhanced branch, install `requirements-training.txt` too. Codespaces installs baseline dependencies and starts the GUI on forwarded port **8501** automatically. If needed: `bash scripts/start-studio.sh`.

### If the GUI does not open

The GitHub repository page displays code; it does not host the running dashboard. In a Codespace, run `git pull` followed by `bash scripts/start-studio.sh`. The launcher waits for a healthy server and prints your forwarded HTTPS address. Open that address, or use **Ports → 8501 → Open in Browser**. Keep the Codespace running. If port 8501 is absent, add it in the Ports panel. A local `localhost:8501` link refers to the computer running Python, so it will not reach a remote Codespace from your phone. For an always-available separate app, deploy `app.py` with `requirements.txt` on a Python-capable Streamlit host.

## Reproduce the real baseline

```bash
python -m carbids.data --download
python -m carbids.baseline --threads 4 --trees 700
python -m carbids.bridge index
```

Source: [Used Car Auction Prices](https://www.kaggle.com/datasets/tunguz/used-car-auction-prices), historical **2014–2015** data. `sellingprice` is the provider's sale target; `mmr` is an estimate and is never a target or feature. Raw CSV archive SHA-256 and detailed exclusions are recorded in `data/cohort.meta.json`. The file has 558,837 logical rows, including 26 malformed-width records and five additional unparsed rows in the pinned preparation path. 8,515 duplicate vehicle records are removed from parsed candidates; 550,291 distinct vehicles remain. Model-year coverage and sale dates are in the model manifest. Original condition grades stored above 5 are rescaled by 10; do not assume the same grading scheme on another auction source.

Historical wholesale performance is not evidence of present-day retail accuracy. The original archive and row-level cohort are not committed; the learned model, schema, evaluation and audit metadata are.

## Rich auction descriptions: strict import contract

Export CSV or JSONL from a permitted, source-audited collection of Bring a Trailer / Cars & Bids listings. A public page is not necessarily a bulk-download dataset or blanket reuse license; this project supplies an importer, not a paywall/bot-protection bypass.

Required columns:

| Column | Meaning |
|---|---|
| `source` | `bringatrailer` or `carsandbids` |
| `source_url`, `auction_id`, `vin` | Original listing URL, lot ID and valid VIN |
| `status` | `sold`, `sold_after_auction`, or `completed_sale` |
| `target_kind` | exactly `platform_reported_sale_price` |
| `currency`, `sale_price`, `sale_date` | USD, positive published sale amount, UTC closing/sale time |
| `price_evidence` | Explicit result text such as `Sold for USD $15,000`; amount must match |
| `year`, `make`, `model`, `mileage_miles` | Pre-sale vehicle facts |
| `description_pre_sale` | Equipment/history/known flaws only; no result line or post-sale comments |
| `text_scope`, `text_as_of` | `pre_sale` and a timestamp no later than the sale |

Optional structured fields: condition, body_type, transmission, paint_color, state, fuel, cylinders, title_status, drive. Rich-text regression requires at least **80 cleaned characters on every training row**, up to **8,000 characters**, in **256-token windows**. Explicit sale evidence is kept for audit but excluded from both model branches. Money amounts are redacted from descriptions as an additional guard; attestation does not independently prove that source text was unedited before sale.

```bash
python -m carbids.data --input auction_export.csv --source auction_export --output data/rich-cohort.jsonl
python -m pip install -r requirements-training.txt
python -m carbids.train --data data/rich-cohort.jsonl --output artifacts/enhanced \
  --device cuda --folds 3 --epochs 2 --batch-size 4 --accumulation 2 --threads 4
```

`data/example_auction_export.csv` contains **synthetic schema fixtures, not real transactions**. Production cohort loading rejects its EXAMPLE IDs. It cannot supply pricing-quality evidence.

## Model design

MiniLM: `sentence-transformers/all-MiniLM-L6-v2`, pinned to `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. All transformer weights and a regression head learn standardized log1p sale prices from descriptions only. Consecutive chunks are pooled into a single description vector and scalar dollar-price prediction. Make/model are categorical; no separate name transformer, TF-IDF, ridge or Extra Trees.

OOF fitting predicts each row from a model excluding that vehicle and its target. Target normalization is fold-local. Isolation Forest uses auto contamination on the training OOF text-price scalar only; it does not remove a fixed percentage. Validation/test are unfiltered. LightGBM compares raw/log targets and Huber alpha 0.5/0.9/1.5, then refits the selected setup after evaluation. Native TreeSHAP explains the tree ensemble, including the text scalar when present, not individual MiniLM words. Contributions add on the fitted target scale, not necessarily in dollars.

## Connecting to Pricing statistically

See [STATISTICAL_BRIDGE.md](STATISTICAL_BRIDGE.md). The recommended path is an independent auction auxiliary signal, a hierarchical robust residual model fitted on matched retail outcomes, source-overlap diagnostics/transport weighting, and target-domain conformal calibration. No fixed 10% haircut and no fabricated retail-sale target.

```bash
python -m carbids.bridge index --model models/tabular
python -m carbids.bridge export --input pricing_vehicles.csv --output auction_features.csv
```

A valid VIN is required; possible upstream training overlaps are refused. An asking-to-auction gap is **not exported by default** because observed asking price leaks the label in an asking-price model. Its opt-in is only for a retail-sale task where a genuine pre-sale ask is available at inference. Unseen market/time regimes need abstention or separately validated transport. The current module does not silently retrain the original Pricing app.

## Optional API

```bash
pip install -r requirements-api.txt
uvicorn carbids.api:app --host 0.0.0.0 --port 8000
```

`GET /health`, `POST /predict`, interactive API docs `/docs`. Set `CARBIDS_MODEL_DIR` to an enhanced bundle to switch models. Inference rejects unsupported auction sources. Authentication/rate limiting are deployment responsibilities; this development server is not an internet-facing production gateway.

## Kaggle

Import `train_kaggle.ipynb`. Default tabular mode reproduces the wholesale benchmark and needs Internet, not a GPU. Enhanced mode needs GPU and exactly one attached audited `auction_export.csv`/`.jsonl`. The notebook embeds source and dependencies, preserves the platform's CUDA PyTorch build, writes explicit ready/pending/failed status and exports `CarBids_Trained.zip` only when a manifest is ready. It uses one GPU; it does not claim distributed T4×2 training or automatically schedule more sessions.

## Verification

```bash
pip install pytest
pytest -q tests/test_sales.py
python -m compileall -q carbids app.py
```

Checks cover sale evidence/amount mismatches, currency/source rejection, pre/post-sale text boundaries, VIN checks, excluded target/bid/comment features, price redaction, whole-date partitions, model target semantics, conservative overlap indexing and conformal test-point mass. Streamlit estimate/explanation/scenario controls and API validation are checked during project delivery.

Code is MIT-licensed; upstream datasets/content retain their own terms and licenses. No secrets, login cookies or private account information are stored in this repository.

## Census Bureau enrichment (ACS5 vintages 2016–2022)

`carbids.census` fetches official ACS five-year estimates at state or county level and performs conservative release-date-aware joins. It includes median household income and its margin of error, population, unemployment, households without vehicles, homeownership and working-from-home shares. It preserves raw estimate/MOE variables, converts Census negative sentinel values to missing, and records source/vintage/geography/release dates.

The live API returned a **Missing Key** page during preparation, so regional values have not been downloaded or substituted. Set `CENSUS_API_KEY` privately (environment or Kaggle Secrets); do not store a key in code or commit caches containing credentials. Official data-only JSON responses may also be placed in the documented cache filenames.

```bash
python -m carbids.census fetch --geography state
python -m carbids.census enrich --input data/rich-cohort.jsonl \
  --panel data/census/acs5-2016-2022-state.csv --output data/rich-census.jsonl
```

The 2015 wholesale benchmark intentionally has **no Census features**: all requested vintages were published after its sales. For a 2021 sale, the 2020 vintage is unavailable until its delayed March 2022 release; for a 2022 sale, the 2022 vintage is unavailable until December 2023. No forward/backward fill from a future vintage is allowed. The GUI shows the full verified release calendar.

When enriched columns are present in a training cohort, the tabular feature builder includes the selected ACS numeric fields; no existing shipped model is silently altered. A Census-enabled inference model requires either its complete enriched input vector or an explicit `as_of` date and a panel at `CARBIDS_CENSUS_PANEL` (default `data/census/acs5-2016-2022-state.csv`). The API accepts an optional `as_of`.

See [CENSUS_CONTEXT.md](CENSUS_CONTEXT.md) for the statistical interpretation, release references, margins of error, geographic caveats and revision limits. No model-performance improvement is claimed without matched future evaluation.

## Integrated structural Pricing + CarBids GUI

The default **Pricing + CarBids** tab transfers auction depreciation/relative-value structure using the vehicle's **age at valuation**, actual mileage and a lagged official BLS used-car price index. It shows the 2021 asking estimate separately, an age/market-adjusted auction equivalent, an experimental asking LightGBM with auction features, decompositions, support warnings and per-car batch export. Both engines and trained weights are bundled; no second service or GPU is required.

The transfer experiment slightly worsened test MAE ($3,329 → $3,338) while improving RMSE ($6,839 → $6,807); the original asking model remains the default. The full 211,844-row Pricing cohort was processed, with 717 unavailable auction equivalents explicitly flagged. Retail-channel premiums remain assumptions, not fitted sale calibration. See [INTEGRATION.md](INTEGRATION.md) for equations/specification, actual assessments, index provenance, leakage controls and reproduction commands.
