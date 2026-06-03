The Architecture of Agreement: Value-Based Persuasion in r/changemyview

The replication code and data for the thesis “The
Architecture of Agreement: Value-Based Persuasion in r/changemyview”.

This repo explores how alignment on human values (based on Schwartz's Basic
Human Values) between an original poster (OP) and a responder correlates with
persuasive success on r/changemyview (CMV) subreddit. To bypass
classifier noise at the individual level, this study implements a Group Level
Quantification (Prevalence Estimation) pipeline using the Adjusted Classify and
Count (ACC) algorithm.

Repository Structure
```
├── data/
│   ├── complete_dataset.json            # 500-comment Gold Standard (Human labeled)
│   ├── group_level_prevalences.csv      # 20-group continuous results (all features)
│   └── group_level_prevalences12values.csv # 50-group continuous results (12 active features)
│   └── pairs.jsonl                      # each record contains submission, delta-comment and nondelta-comment and the comments' similarity score (Webis-CMV-20, Al-Khatib et al., 2020)
|   └── triage_pairs.json                # each record contains an Original Post, a Delta-winning rebuttal, and a non-winning rebuttal from the same discussion tree
|
├── src/
│   ├── main.py                          # Model Fine-tuning (SetFit + ModernBERT)
│   ├── get_error_rates.py               # Extract Validation TPR/FPR for ACC
│   ├── label_dataset.py                 # GPU-optimized batch inference script
│   ├── quantify_groups.py               # ACC Calibration (original 20 features)
│   ├── quantify_groups12_values.py      # ACC Calibration (12 approved features)
│   ├── run_statistical_tests.py         # Hypothesis testing (20 features)
│   └── run_statistical_tests12_values.py # Hypothesis testing (12 approved features)
│   └── extract_op_delta_rebuttal.py     # Extracts the 10,303 triade pairs from pairs.jsonl (Al-Khatib et al., 2020)
|
├── diagnostics/
│   ├── analyze_bias_and_distributions.py # Model systematic bias audit script
│   ├── pilot_comparison.py              # Sequence length comparison script (256 vs 512)
│   └── seq_length_dis.py                # Token length distribution analyzer
│   └── baseline_singlelabel.py          # TF-IDF + Logistic Regression/SVM predicting only the first human value per text (single-label) 
|   └── baseline_multilabel.py           # TF-IDF + Logistic Regression/SVM predicting all human values per text simultaneously (multi-label).
|   └── iaa.py                           # computes IAA (pairwise F1, per-label consensus, and label frequencies) across three annotators' JSON files


|
└── requirements.txt                     # Python packages list
```
Installation & Setup

1.  Clone the repository:

    git clone https://github.com/yourusername/architecture-of-agreement.git
    cd architecture-of-agreement

2.  Create and activate a virtual environment:

    This project requires **Python 3.10** due to dependency constraints between `setfit`, `sentence-transformers`, and `transformers`. Python 3.12+ will cause import errors.
    
    Install Python 3.10 via [pyenv](https://github.com/pyenv/pyenv) if needed:
    brew install pyenv
    pyenv install 3.10.13

3.  Install dependencies:

    ~/.pyenv/versions/3.10.13/bin/python -m venv thesis_env
    source thesis_env/bin/activate
    pip install -r requirements.txt

Replication Pipeline

To reproduce the findings of this thesis, execute the scripts in the following
order:

Step 1: Model Training & Validation

Fine-tune the ModernBERT model using the SetFit framework on the human-labeled
gold standard:

python3 src/main.py

Calculate the model's True Positive Rate (TPR) and False Positive Rate (FPR)
across the validation set:

python3 src/get_error_rates.py

Step 2: Large-Scale Inference

Run batched GPU-accelerated inference across the full 10,303 structured argument
pairs:

python3 src/label_dataset.py

(Note: To run this step, you must obtain the raw pairs.jsonl file from the
Webis-CMV-20 corpus and place it in the root directory).

Step 3: Group-Level Quantification (ACC Calibration)

Aggregate the inference predictions into 50 sub-populations and apply the ACC
mathematical calibration formula over the 12 approved features:

python3 src/quantify_groups11.py

Step 4: Hypothesis Testing

Run the final statistical tests (Paired t-tests) to evaluate value-matching
distance and individual value predictors:

python3 src/run_statistical_tests_filtered.py
