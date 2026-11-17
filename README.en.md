# Anti-Fraud: fraud scoring with the LightGBM II model (Streamlit)

*Русская версия: [README.md](README.md)*

A Streamlit web app that estimates the probability of fraud for transactions whose **458 features have already been
prepared**. It uses the final model of a research project built on the
[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) competition data (Kaggle).

**Model — LightGBM II:**
- 4,628 trees, 458 features;
- trained on the first 80% of transactions by time;
- default threshold 0.0486;
- on the time-based Validation set (42 days): AP 0.621, ROC-AUC 0.919, F1 0.597.

> **About the metrics.** All quality metrics in this README and in the interface were obtained on the
> **Validation** set. This set was already used during model development and comparison, so the metrics are
> **not an independent evaluation on Test**. The competition test set (Test) was **not used** to evaluate the model.

**What the app does:**
1. accepts a CSV file with prepared features;
2. checks the names, count, order and types of all 458 features;
3. computes the fraud probability and the class at the threshold;
4. shows a summary and lets you download the results.

> **The app does not preprocess raw transactions.** Frequency features, the card1+addr1 group history and other
> derived features must be built in advance with the same code that was used to train the model. Raw competition
> tables (transaction / identity) are not accepted as input. The research notebooks that build the features are
> **not part of this repository**.

## The model is stored privately — the app will not start without access to it

- The repository does **not** contain the model file. The model (`lgbm_B_full.txt`, ≈130 MB) is kept in a **private**
  Hugging Face Hub repository. It is not distributed publicly until it is confirmed that the competition rules allow this.
- At startup the app downloads the model from the `HF_REPO_ID` repository using the `HF_TOKEN` token. It then checks the
  SHA-256 hash (`config/app_config.json`), the number of trees (4,628) and the feature names. On any mismatch, no
  predictions are made.
- **Without access to the model the app will not start.** One of the following is required:
  - `HF_REPO_ID` (e.g. `owner/repo-name`) and `HF_TOKEN` (a read-only token for that repository), set as environment
    variables or Streamlit secrets;
  - a local model file whose path is given in `MODEL_PATH` (takes priority).
- The repository name and the token are not stored in the code. The `hf_repo_id` field in `config/app_config.json` is
  a placeholder. It is used only when `HF_REPO_ID` is not set, and the placeholder is never treated as a value.
- Secrets go into `.streamlit/secrets.toml` (git-ignored; see `secrets.toml.example`) or environment variables.
- The model is loaded once per process (`st.cache_resource`).

## The 458 model features

The full list with types is in `config/feature_schema.json`. The header template is in `templates/features_template.csv`.

| Group | Count | Contents | Source |
|---|---|---|---|
| Raw transaction features | 391 | TransactionAmt, ProductCD, card1–card6, addr1–addr2, dist1–dist2, P_/R_emaildomain, C1–C14, D1–D15, M1–M9, V1–V339 | competition transaction table (without TransactionID, TransactionDT, isFraud) |
| Raw identity features | 40 | id_01–id_38, DeviceType, DeviceInfo | identity table (joined on TransactionID; empty for transactions without identity) |
| Simple derived features | 7 | has_identity, hour, weekday, n_missing_V / D / M / id | computed per row |
| Interpretable features | 12 | amount_group, is_large_amount, card_type_group, card_category_group, P_/R_email_provider, P_/R_email_domain_freq, email_same_domain, device_os, device_brand, device_model_family | rules and statistics fitted on the training part |
| card1+addr1 group history | 5 | group_prev_transactions, group_prev_amount_mean / min / max, group_time_since_prev | from earlier transactions of the same group, **strictly earlier in time** (the group is an approximate grouping of transactions, not a customer ID) |
| History-derived features | 3 | group_first_operation, group_time_since_prev_bin, group_prev_amount_mean_bin | from the history features |
| **Total** | **458** | 41 categorical, 417 numeric | |

The meaning of many raw features (C, D, V, M, id_*) was not disclosed by the competition organizers.

## Input CSV

| Rule | App behaviour |
|---|---|
| Encoding | UTF-8 (BOM allowed); separator ",", decimal point "." |
| Columns | exactly the 458 features named in the template, plus an optional `TransactionID` |
| Missing feature | **error**; it is not filled with zeros or anything else |
| Extra columns | error (including `isFraud`: the true label is not an input) |
| Duplicate column names | error |
| Different column order | columns are reordered by name, then the order is checked against the model; a warning is shown |
| Missing value | **an empty cell only** |
| Numeric features (417) | a finite number or an empty cell; any other text ("NA", "nan", "inf", "1,5") is an error, reported with row numbers |
| Categorical features (41) | text is kept as is: "NA", "None", "–", "≥1000" are values, not missing values |
| Category value unseen in training | the model treats it as missing, exactly as during model evaluation; the app reports how many such values there are and does not replace them |
| Duplicate `TransactionID` | warning |
| Size | at most 50,000 rows, file up to 100 MB |

Write numeric values with full float64 precision. Rounding may change the predictions compared with the research project.

## Output CSV

File name: `lgbm2_results_<first 8 characters of the model SHA-256>_thr<threshold>.csv`, e.g.
`lgbm2_results_f413c46f_thr0.0486.csv`.

| Column | Meaning |
|---|---|
| `row` | row number in the input file (from 1, excluding the header) |
| `TransactionID` | identifier from the input file (only if the column was present) |
| `fraud_probability` | model probability of fraud, from 0 to 1 |
| `predicted_class` | 1 = alert (probability ≥ threshold), 0 = no alert |
| `threshold` | the threshold used to assign the class |

## Classification threshold

- **Default: 0.0486** (exactly 0.04861665045870177). It maximizes F1 on an internal time-based check within the
  training part and was transferred to the final model. At this threshold on Validation: precision 0.649,
  recall 0.552, F1 0.597. These are results on the Validation set used during model development and comparison,
  not an independent evaluation on Test.
- The model was trained with class weights (fraud is ≈3.5% of transactions), and the probabilities were not
  calibrated. Treat `fraud_probability` as a score for ranking and comparison with the threshold, not as a
  calibrated fraud rate.
- The threshold can be changed in the interface. The actual threshold is shown next to the number of alerts and
  written to the results file. **Model quality at other thresholds has not been evaluated.** In practice the threshold
  should be chosen from the cost of a missed fraud versus a false alert, or from the acceptable review workload.
  These values are unknown here.

## Running the app

Requires Python 3.12.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
$env:MODEL_PATH = "<path_to_lgbm_B_full.txt>"      # or: $env:HF_REPO_ID = "<owner>/<repo>"; $env:HF_TOKEN = "<token>"
.venv\Scripts\streamlit run app.py --server.address 127.0.0.1
```

**Linux and macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export MODEL_PATH="<path_to_lgbm_B_full.txt>"      # or: export HF_REPO_ID="<owner>/<repo>" HF_TOKEN="<token>"
streamlit run app.py --server.address 127.0.0.1
```

With the private Hugging Face Hub: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in
`HF_REPO_ID` and `HF_TOKEN`, or set them as environment variables.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

| File | What it checks | Requirements |
|---|---|---|
| `test_validation.py` | reading and validation rules on synthetic tables | none |
| `test_model_loader.py` | model source selection: `MODEL_PATH`, `HF_REPO_ID`, config placeholder (no network) | none |
| `test_publish_safety.py` | the publication check script detects forbidden content | none |
| `test_categories.py` | handling of the 41 categorical features matches the model's behaviour | model (`MODEL_PATH`) |
| `test_app_smoke.py` | interface: startup, input errors, threshold, results file name | model; local data for the full scenario |
| `test_local_acceptance.py` | predictions from the CSV match the research project (tolerance 1e-12), including after column reordering | model and local data (`LOCAL_DATA_DIR`) |

Tests that need the model or local data are skipped when these are not available. Expected values for checks on real
data are not stored in the test code. They are computed at run time from a local reference file (`LOCAL_DATA_DIR`).

`tools/export_features_local.py` prepares a local feature CSV for the acceptance test **from the research project
data**, which is not included here. The project root is set with `--project-root` or the `PROJECT_ROOT` environment
variable. The output is written only to a folder outside the repository.

## Pre-commit check

`python tools/check_publish_safety.py` lists the files git would include in the repository and searches them for
anything that must not be published:
- models, data, CSV files other than the template, archives;
- absolute paths, tokens, keys, passwords, e-mail addresses;
- files larger than 1 MB.

It also checks that the `.gitignore` rules work. Without git, a simplified `--no-git` mode is available.

## Data and license

- The IEEE-CIS Fraud Detection competition data is **not distributed** with this code. The repository contains no raw
  rows, real features, category dictionaries or predictions on real data. Use of the data is governed by the
  competition rules on Kaggle.
- **The MIT License (`LICENSE`) covers only the source code of this app.** It does not cover the Kaggle data, any tables
  derived from it, or the trained model.

## Model limitations

- The evaluation comes from a single time-based split and a single Validation set (42 days). This set was used
  during model development and comparison, so it is not an independent evaluation on Test. Test was not used to
  evaluate the model.
- One random seed. The threshold was chosen by F1; the cost of errors is unknown.
- The model was trained on data from one period. Its transferability to later periods has not been checked.
