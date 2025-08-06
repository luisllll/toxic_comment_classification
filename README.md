# toxic_comment_classification



## 📁 Repository Structure

```
toxic-comment-multilabel/
│
├── README.md                    # Project description
├── requirements.txt             # Python dependencies
├── .gitignore                  # Ignore large files
├── mlflow.db                   # MLflow database
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_eda_multilabel.ipynb        # Multi-label analysis
│   ├── 02_baseline_hf_models.ipynb    # HuggingFace baseline models
│   ├── 03_multilabel_finetuning.ipynb # Fine-tuning strategies
│   └── 04_ensemble_experiments.ipynb   # Model ensembling
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── raw/         # csv data
│   │   ├── dataset.py          # Custom Dataset for multi-label
│   │   ├── preprocessing.py    # Text preprocessing
│   │   └── augmentation.py     # Data augmentation for imbalanced labels
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hf_baselines.py     # HuggingFace model wrappers
│   │   ├── multilabel_heads.py # Custom multi-label heads
│   │   ├── model_factory.py    # Model creation utilities
│   │   └── ensemble.py         # Ensemble strategies
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Custom trainer for multi-label
│   │   ├── loss_functions.py   # Multi-label loss functions
│   │   └── callbacks.py        # Training callbacks
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py          # Multi-label metrics
│   │   ├── threshold_tuning.py # Per-label threshold optimization
│   │   └── bias_analysis.py    # Bias and fairness checks
│   │
│   └── experiments/
│       ├── __init__.py
│       ├── mlflow_manager.py   # MLflow experiment tracking
│       └── hyperparameter_search.py # Optuna integration
│
├── experiments/                # Experiment configurations
│   ├── configs/
│   │   ├── bert_base.yaml      # BERT baseline config
│   │   ├── roberta_toxic.yaml  # RoBERTa for toxicity
│   │   ├── deberta_v3.yaml     # DeBERTa v3 config
│   │   ├── toxic_bert.yaml     # Toxic-BERT specialized
│   │   └── ensemble_best.yaml  # Best ensemble config
│   │
│   └── run_experiment.py       # Main experiment runner
│
├── scripts/                    # Executable scripts
│   ├── download_data.sh        # Download Kaggle data
│   ├── setup_mlflow.sh         # Setup MLflow
│   ├── train_baselines.py      # Train all baseline models
│   ├── finetune_best.py        # Fine-tune best model
│   └── make_submission.py      # Generate submission
│
├── models/                     # Model artifacts
│   ├── pretrained/             # Downloaded HF models
│   ├── finetuned/              # Fine-tuned models
│   └── mlflow_registry/        # MLflow model registry
│
└── submissions/                # Kaggle submissions
    └── .gitkeep
```