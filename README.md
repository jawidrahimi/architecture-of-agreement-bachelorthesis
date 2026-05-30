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

├── data/
│   ├── complete_dataset_test.json       # 500-comment Gold Standard (Human labeled)
│   └── group_level_prevalences.csv      # 50-group continuous dataset (Final output)
│
├── src/
│   ├── main.py                          # Model Fine-tuning (SetFit + ModernBERT)
│   ├── get_error_rates.py               # Extract Validation TPR/FPR for ACC
│   ├── label_dataset.py                 # GPU-optimized batch inference script
│   ├── quantify_groups11.py             # ACC Calibration (12 approved features)
│   └── run_statistical_tests_filtered.py # Hypothesis testing (Paired t-tests)
│
├── diagnostics/
│   ├── analyze_lengths.py               # Token length distribution analyzer
│   ├── analyze_bias_and_distributions.py # Model systematic bias audit script
│   └── pilot_comparison.py              # Sequence length comparison script
│
├── .gitignore                           # Excludes large files and model weights
└── requirements.txt                     # Python packages list

Installation & Setup

1.  Clone the repository:

    git clone https://github.com/yourusername/architecture-of-agreement.git
    cd architecture-of-agreement

2.  Create and activate a virtual environment:

    python3 -m venv thesis_env
    source thesis_env/bin/activate

3.  Install dependencies:

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
