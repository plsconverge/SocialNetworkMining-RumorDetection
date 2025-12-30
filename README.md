# SocialNetworkMining-RumorDetection

Project for lecture Social Network Mining.

## Project Structure

```
root/
├── data/                   # Data storage
│   ├── CED_Dataset/        # Original dataset (Chinese_Rumor_Dataset)
│   ├── CED_Processed/      # Processed data artifacts
│   └── models/             # Saved models and pre-trained weights (e.g., bert-base-chinese)
├── docs/                   # Documentation
├── results/                # Experiment results and outputs
├── src/                    # Source code
│   ├── data/               # Data loading, processing and dataset definitions
│   │   ├── bert_dataset.py     # Dataset wrapper for BERT
│   │   ├── data_loader.py      # Base raw data loading
│   │   ├── feature_engineer.py # Feature extraction for ML models
│   │   ├── gcn_dataset.py      # Graph construction for GNNs
│   │   └── llm_dataset.py      # Data formatting for LLMs
│   ├── experiments/        # Ad-hoc analysis and experimental scripts
│   ├── models/             # Model architectures
│   │   ├── bert_classifier.py
│   │   ├── gnn_classifier.py
│   │   └── llm_classifier.py
│   ├── prompts/            # System prompts for LLM tasks
│   ├── trainers/           # Training scripts and entry points
│   │   ├── train_bert_baseline.py
│   │   ├── train_gcn.py
│   │   ├── train_llm_gemini.py
│   │   ├── train_logistics.py
│   │   ├── train_random_forest.py
│   │   └── train_xgboost.py
│   └── utils/              # Utilities and helper functions
│       ├── cross_validation.py # Cross-validation implementation
│       ├── evaluation.py       # Metric calculation (F1, Accuracy, etc.)
│       └── llm_cross_validation.py
└── visualization/          # Visualization scripts
    ├── visual_daily_distribution.py
    ├── visual_interval.py
    ├── visual_propagation.py
    └── visual_timeline.py
```

## Dataset

The CED dataset is contributed by thunlp group. 
It can be downloaded from [https://github.com/thunlp/Chinese_Rumor_Dataset](https://github.com/thunlp/Chinese_Rumor_Dataset).

## Implemented Models

The project currently implements the following approaches for rumor detection:

1.  **Baseline Machine Learning**: 
    *   Logistic Regression
    *   Random Forest
    *   XGBoost
    *   *Note: These use handcrafted features defined in `src/data/feature_engineer.py`.*

2.  **Deep Learning (NLP)**:
    *   BERT (Fine-tuning `bert-base-chinese`)

3.  **Graph Neural Networks**:
    *   GCN (Graph Convolutional Networks) modeling the propagation structure.

4.  **Large Language Models**:
    *   Gemini (via API) using specific prompts defined in `src/prompts/`.

## Usage

To train a model, execute the corresponding script in `src/trainers/`. For example:

```bash
# Train BERT model
python src/trainers/train_bert_baseline.py

# Train Random Forest
python src/trainers/train_random_forest.py
```