# Anti-Fraud: detecting fraudulent transactions with LightGBM II

*Русская версия: [README.md](README.md)*

**Live demo:** [anti-fraud-lightgbm.streamlit.app](https://anti-fraud-lightgbm.streamlit.app)
— the model scores synthetic transactions, nothing to install. If the app is asleep, click the wake-up button and wait
about a minute.

## Project goal

Build a model that estimates the **fraud risk** of a bank transaction and catches as many fraudulent transactions as
possible with few false alarms, and demonstrate it in an application.

**Data** — the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) competition
(Kaggle): 590,540 labelled transactions, about 3.5% of them fraudulent. Because the classes are heavily imbalanced,
plain accuracy is useless: a model that calls every transaction legitimate already scores 96.6%.

## Final model — LightGBM II

Gradient boosting with LightGBM: 4,628 trees, 458 features. The model outputs a **risk score** from 0 to 1; a
transaction scoring at or above the threshold **0.0486** raises an **alert**.

| Metric | Value | Meaning |
|---|---|---|
| **AP** (average precision) — main | **0.621** | average share of correct alerts across thresholds; focuses on the rare class. A random model would get ≈0.034 |
| ROC-AUC | 0.919 | probability that a fraudulent transaction scores higher than a legitimate one (0.5 = coin flip, 1 = perfect) |
| Recall | 0.552 | share of fraudulent transactions caught |
| Precision | 0.649 | share of alerts that are correct |
| F1 | 0.597 | balance of precision and recall |
| Caught at 1% false alarms | 55% | when 1,000 false alarms per 100,000 legitimate transactions are allowed |

**Confusion matrix** (threshold 0.0486, 118,108 validation transactions):

| | Model: alert | Model: legitimate |
|---|---|---|
| **Fraud** | caught 2,243 | missed 1,821 |
| **Legitimate** | false alarms 1,213 | correct 112,831 |

## How the model was evaluated

Transactions are ordered **by time**: the model learns from the past and is checked on the future, as in real use.

```
|------------- first 80% of transactions (training) -------------|---- last 20% ----|
|        Training — 401,567         |    Holdout — 70,865        | Validation —     |
|        the model learns           |  settings and threshold    | 118,108, 42 days |
```

- **Training** — the model sees the correct answers here and learns.
- **Holdout** (internal check) — not used for learning; settings and the threshold were chosen here.
- **Validation** — the last 42 days; nothing was tuned here, the metrics above are measured on it.
- The competition Test set was not used: it has no labels.
- History features use **only earlier transactions**; the label (`isFraud`) is never a feature.

## How the settings were chosen

Hyperparameters are set before training: how complex the trees are, how strongly complexity is penalised and when to
stop training. All variants were compared on the internal check (Holdout).

1. **Early stopping on AP** — trees are added while AP on the internal check keeps improving. This gave the main gain:
   AP from 0.577 to 0.655.
2. **Final choice among 4 variants** (127 / 255 leaves × regularisation 0.5 / 5). Best: 255 leaves, regularisation 0.5
   (AP 0.672 on the internal check). This is LightGBM II.
3. **Threshold 0.0486** chosen on the same check by maximum F1.

## What matters to the model

Top features by gain: recipient e-mail domain, Vesta features (V258, C1, C13, C14, V294), card codes (card1, card2),
transaction amount and a custom history feature — the maximum amount of earlier transactions of the same card+address
group. The model is strongest on products C and R; the hardest cases are the mass product W and transactions without
device data.

## Repository contents

| Part | What it does |
|---|---|
| `app.py` | main app: upload a CSV with 458 prepared features → probability and decision for each transaction |
| `demo/` | demo: generates synthetic transactions and scores them with the real LightGBM II model. `demo/cloud_app.py` is the Streamlit Community Cloud entry point (downloads the model using the `HF_REPO_ID`, `HF_TOKEN` secrets) |
| `fraud_app/` | model loading and verification, input validation, prediction |
| `config/` | model parameters and the list of 458 features |
| `tests/`, `demo/tests/` | automated tests |
| `tools/` | pre-publication check and local data preparation for the acceptance test |

## The model is stored privately

- The model file is **not** in this repository. The model (`lgbm_B_full.txt`, ≈130 MB) is kept in a **private**
  Hugging Face Hub repository: public distribution is not allowed until it is confirmed that the competition rules permit it.
- At start-up the app downloads the model from `HF_REPO_ID` using `HF_TOKEN` and verifies its SHA-256
  (`config/app_config.json`), tree count (4,628) and feature names. On any mismatch no prediction is made.
- Alternatively, point `MODEL_PATH` to a local model file (takes priority).
- The repository name and token are not stored in code; secrets go to `.streamlit/secrets.toml` (git-ignored, template
  `secrets.toml.example`) or environment variables.

## 458 model features

Full list and types — `config/feature_schema.json`; header template — `templates/features_template.csv`.

| Group | Count | Contents |
|---|---|---|
| Raw transaction features | 391 | TransactionAmt, ProductCD, card1–card6, addr1–addr2, dist1–dist2, P_/R_emaildomain, C1–C14, D1–D15, M1–M9, V1–V339 |
| Raw device features (identity) | 40 | id_01–id_38, DeviceType, DeviceInfo |
| Simple derived | 7 | has_identity, relative hour and weekday, missing counts in V / D / M / id |
| Interpretable features | 12 | amount group, card type and category, e-mail provider and domain frequency, same-domain flag, OS, device brand and model |
| card1+addr1 group history | 8 | count, mean / min / max amount of earlier transactions, time since previous one and their bins |
| **Total** | **458** | 41 categorical, 417 numeric |

The meaning of many raw features (C, D, V, M, id_*) was not disclosed by the organisers.

## Input and output CSV (main app)

**The app does not build features from raw transactions** — the input is a table with the 458 features already built.

| Rule | Behaviour |
|---|---|
| Encoding | UTF-8 (BOM allowed); separator «,», decimal point «.» |
| Columns | exactly the 458 template features and, optionally, `TransactionID` |
| Missing or extra column (including `isFraud`) | error |
| Different column order | columns are reordered by name, a warning is shown |
| Missing value | empty cell only |
| Numeric features | a number or an empty cell; any other text is an error with row numbers |
| Unseen category value | treated as missing, as during model evaluation; the app reports how many |
| Size | up to 50,000 rows, file up to 100 MB |

Output: `lgbm2_results_<model SHA-256, 8 chars>_thr<threshold>.csv` with columns `row`, `TransactionID` (if present
in the input), `fraud_probability`, `predicted_class` (1 = alert) and `threshold`.

The threshold can be changed in the UI; quality at other thresholds was not evaluated. Scores are not calibrated: they
are a risk scale for comparison with the threshold, not exact probabilities.

## Running

Python 3.12 is required.

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
$env:MODEL_PATH = "<path_to_lgbm_B_full.txt>"      # or HF_REPO_ID and HF_TOKEN
.venv\Scripts\streamlit run app.py --server.address 127.0.0.1
```

Demo locally: also set `FRAUD_PARAMS_DIR` (folder with the project's feature parameters) and run
`streamlit run demo/app_demo.py`.

On Linux and macOS use the same steps with `source .venv/bin/activate` and `export`.

## Tests and pre-publication check

```bash
pip install -r requirements-dev.txt
python -m pytest tests demo/tests -q
python tools/check_publish_safety.py
```

Tests that need the model or local data are skipped without them. `check_publish_safety.py` verifies that no models,
data, CSV files (except the template), tokens, absolute paths or files over 1 MB will be committed.

## Limitations

- The metrics come from a single time split (validation period of 42 days). Other model variants were also checked on
  this period during development, so the result may be slightly optimistic; there was no check on completely new data (Test).
- One seed; the threshold was chosen by F1 — the real cost of a missed fraud and of a false alarm is unknown.
- The model was trained on one period of IEEE-CIS-format data; transfer to other data was not tested.

## Data and licence

- Competition data is **not distributed**: the repository contains no raw rows, real features, category dictionaries
  or predictions on real data. Use of the data is governed by the competition rules on Kaggle.
- **The MIT licence (`LICENSE`) covers the source code only.** It does not cover Kaggle data, tables derived from it or
  the trained model.
