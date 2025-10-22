# MASTRO: Multi-Agent System for Time-series Dropout Prediction that integrates specialized learning agents and reasoning mechanisms within a unified coordination framework

This repository contains a sophisticated multi-agent system for predicting student dropout risk using time series data and large language models (LLMs). The system combines traditional ML models with specialized agents and LLM reasoning to provide accurate predictions with explainable rationales.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Agents Architecture](#agents-architecture)
- [Configuration](#configuration)
- [Output](#output)
- [Citation](#citation)

## Overview

The system implements a multi-agent architecture that includes:
- **Six specialized agents** focusing on different aspects of student behavior
- **Time series analysis** with temporal feature engineering
- **LLM integration** for generating human-interpretable explanations
- **Ensemble stacking** for final predictions

## Installation

### Prerequisites
- Python >= 3.10
- Required Python packages (see requirements.txt)

1. Clone the repository:
```bash
git clone https://github.com/sornharith/MASTRO.git
cd MASTRO
```

2. Install dependencies:
```bash
cd MASTRO
pip install -r requirements.txt
```

Or install the package in development mode using setup.py:
```bash
pip install -e .
```

## Directory Structure

The codebase is organized into logical modules for better maintainability:

```
MASTRO/
├── agents/                 # Agent implementations
│   ├── __init__.py
│   ├── cat_agent.py        # CatBoost-based agents
│   ├── prompt_agent.py     # Prompt-based agents using LLMs
│   └── sme_moe_agent.py    # SMETimes Mixture of Experts agent
├── data/                   # Data loading and preprocessing
│   ├── __init__.py
│   ├── dataloader.py       # Dataset loading functions
│   └── timeseries_features.py # Time series feature engineering
├── llm/                    # LLM integration
│   ├── __init__.py
│   ├── prompts.py          # LLM prompt templates
│   ├── model_setup.py      # LLM model initialization
│   └── utils.py            # LLM utilities
├── models/                 # ML model training and tuning
│   ├── __init__.py
│   └── tuning.py           # Hyperparameter tuning
├── utils/                  # Utility functions
│   ├── __init__.py
│   └── logger.py           # Logging utilities
├── logs/                   # Execution logs (created automatically)
├── results/                # Output files (created automatically)
├── model_save/             # Trained models (created automatically)
├── main.py                 # Main execution module
├── __init__.py
├── setup.py                # Package setup
├── requirements.txt        # Dependencies
├── runner.py               # Pipeline of the process
└── README.md               # Documentation
```

## Output Directory Structure

When running the system, the following directories are automatically created:

```
project_root/
├── logs/                   # Execution logs
│   └── agent_timeseries_[dataset].log
├── results/                # Output files
│   ├── timeseries_multi_agent_dropout_predictions_[dataset]_[n_students].csv
│   └── finetune.jsonl      # (if --train_log is used)
└── model_save/             # Trained models
    └── [dataset]
        └── [n_students]_students
             ├── [agent_name]_agent_model_[dataset].joblib
             └── stacker_model_[dataset].joblib
```

## Usage

### Basic Usage
```bash
# Run with default settings using OULAD dataset
python -m main.py --llama-size 7b --ollama True --N_students 50
```

### Using Different Datasets

**For UCI dataset:**
```bash
python -m main.py --llama-size 7b --ollama True --N_students 50 --uci True --uci_path "path/to/your/uci/data.csv"
```

**For XuetangX dataset:**
```bash
python -m main.py --llama-size 7b --ollama True --N_students 50 --xuetangx True --train_path "Train.csv" --test_path "Test.csv" --user_info_path "user_info.csv"
```

**For OULAD dataset** (specify the folder path):
```bash
python -m main.py --llama-size 7b --ollama True --N_students 50 --folder_path "raw_datasets/OULAD/data"
```
*Note: The OULAD dataset requires multiple CSV files in the specified folder: `assessments.csv`, `courses.csv`, `studentInfo.csv`, `studentRegistration.csv`, `studentAssessment.csv`, `studentVle.csv`, and `vle.csv`. The processed dataset will be saved as `datasets/ts_datasets.csv`.*

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--llama-size` | LLM size (7b or 13b) | 7b |
| `--time-window` | Days for time series window | 30 |
| `--prediction-horizon` | Days ahead to predict | 7 |
| `--ollama` | Use Ollama for LLM inference | False |
| `--N_students` | Number of students to sample | 50 |
| `--retune` | Force model retuning, ignoring saved models | False |
| `--uci` | Use UCI Student Dropout dataset | False |
| `--xuetangx` | Use XuetangX MOOC dataset | False |
| `--train_log` | Save for fine-tune model | False |
| `--folder_path` | Path to OULAD dataset folder | `raw_datasets/OULAD/data` |
| `--uci_path` | Path to UCI dataset CSV | `raw_datasets/UCI/data.csv` |
| `--train_path` | Path to XuetangX Train.csv | `raw_datasets/xuetangx/Train.csv` |
| `--test_path` | Path to XuetangX Test.csv | `raw_datasets/xuetangx/Test.csv` |
| `--user_info_path` | Path to XuetangX user_info.csv | `raw_datasets/xuetangx/user_info.csv` |

## Data Preparation

### Required Data Format

The system works with three types of educational datasets:

**OULAD (Open University Learning Analytics Dataset)**
- Individual CSV files: `assessments.csv`, `courses.csv`, `studentInfo.csv`, `studentRegistration.csv`, `studentAssessment.csv`, `studentVle.csv`, and `vle.csv`
- Download from: [Open University Learning Analytics Dataset](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad/data)

**UCI Student Performance Dataset**
- Single CSV file with columns for demographics, grades, and final target
- Download from: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

**XuetangX MOOC Dataset**
- `Train.csv`, `Test.csv`, `user_info.csv` files
- Available upon request from research team
- Download from: [XuetangX MOOC Dataset](https://www.kaggle.com/datasets/anasnofal/mooc-data-xuetangx)

### Dataset Directory Structure

The system expects the following directory structure for raw datasets:

```
MASTRO/
├── raw_datasets/
│   ├── OULAD/
│   │   └── data/
│   │       ├── assessments.csv
│   │       ├── courses.csv
│   │       ├── studentInfo.csv
│   │       ├── studentRegistration.csv
│   │       ├── studentAssessment.csv
│   │       ├── studentVle.csv
│   │       └── vle.csv
│   ├── UCI/
│   │   └── data.csv
│   └── xuetangx/
│       ├── Train.csv
│       ├── Test.csv
│       └── user_info.csv
```

To download the OULAD dataset:
1. Visit the [Open University Learning Analytics Dataset](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad/data) page
2. Register for access to the dataset
3. Download the individual CSV files (assessments, courses, studentInfo, studentRegistration, studentAssessment, studentVle, vle)
4. Place them in `raw_datasets/OULAD/data/` directory

### Combining OULAD Dataset

We provide a script to combine the individual OULAD files into a single file that can be used as a fallback. To run the script:

```bash
python scripts/combine_oulad.py
```

This script will create `raw_datasets/OULAD/oulad_all_combined.csv` which is used when individual files are not available.

### Data Columns

**Required for all datasets:**
- `student_id`: Unique identifier for each student
- `dropout`: Target variable (1 for dropout, 0 for persist)

**Time Series Features:**
- `snapshot_day`: Days into the course
- `clicks_total`: Total interactions with VLE
- `clicks_mean`: Average daily clicks
- `assessments_completed`: Number of completed assessments
- `assessment_score_mean`: Average assessment score
- `days_since_last_activity`: Days since last login

## Agents Architecture

### Specialized Agents

1. **Performance Agent**
   - Analyzes academic performance metrics
   - Focus: Grades, assessment completion, score trends

2. **Profile Agent**  
   - Analyzes demographic and static characteristics
   - Focus: Gender, age, education level, previous attempts

3. **Engagement Agent**
   - Analyzes behavioral engagement patterns
   - Focus: Click patterns, activity consistency, engagement volatility

4. **Temporal Agent**
   - Analyzes time-based patterns
   - Focus: Inactivity periods, activity timing, temporal trends

5. **Course Difficulty Agent**
   - Analyzes course-specific difficulty
   - Uses clustering to identify challenging course patterns

6. **SMETimes Agent**
   - Advanced time-series analysis using MoE (Mixture of Experts)
   - Focus: Trend, volatility, level, and temporal modeling

### Final Decision Agent
The system includes a "Final Decision Agent" that:
- Synthesizes outputs from all specialized agents
- Provides structured reasoning using internal monologue
- Generates human-interpretable explanations
- Uses historical context to track risk trajectory

## LLM Integration

The system uses two types of LLMs:

1. **Ollama** (recommended for deployment):
   - Requires Ollama to be installed and running
   - Supports Llama 2 models

2. **Fine-tuned GPT-2**:
   - Used specifically for the SMETimes agent
   - Located at `utils/gpt2_dropout_qlora`

## Output

### Generated Files

1. **Dataset Files**: `datasets/ts_datasets_[dataset].csv`
   - Preprocessed time series features

2. **Model Files**: `model_save/[dataset]/[n_students]_students/[agent_name]_agent_model_[dataset].joblib`
   - Trained agent models for reuse

3. **Results File**: `results/timeseries_multi_agent_dropout_predictions_[dataset]_[n_students].csv`
   - Comprehensive results with predictions, rationales, and metrics

4. **Log Files**: `logs/agent_timeseries_[dataset].log`
   - Execution logs with detailed information

5. **Fine-tuning Data**: `results/finetune.jsonl` (if `--train_log` is used)
   - Prompt-completion pairs for model fine-tuning

### Results Columns

The output CSV includes:
- `student_id`: Student identifier
- `snapshot_day`: Day of prediction
- `[agent]_risk`: Risk score from each agent
- `[agent]_rationale`: Explanation from each agent
- `final_prediction`: Final dropout prediction (0/1)
- `final_risk`: Final risk score (0-1)
- `final_confidence`: Confidence in prediction
- `final_rationale`: Human-readable explanation
- `internal_reasoning`: Technical reasoning (JSON)
- `stack_risk`: Risk score from the stacker model
- Course stage and temporal feature columns

## Configuration

### Pre-trained Models
The system will automatically save and reload trained models to avoid retraining. To force retraining:
```bash
python -m main.py --llama-size 7b --ollama True --N_students 50 --retune
```

### Threshold Optimization
The system automatically optimizes the decision threshold on a validation split to maximize F1-score, testing values between 0.3 and 0.7.

### Fine-tuning Data
To generate fine-tuning data for the final arbiter:
```bash
python -m main.py --llama-size 7b --ollama True --N_students 50 --train_log True
```
This creates `finetune.jsonl` with prompt-completion pairs.

## Performance Metrics

The system calculates multiple metrics:
- **Accuracy**: Overall prediction accuracy
- **F1-Score**: Balanced measure of precision and recall
- **AUC-ROC**: Area under the ROC curve
- **Recall**: True positive rate
- **Stage-specific accuracy**: Performance by early/mid/late course stages

## Troubleshooting

### Common Issues

1. **CUDA Memory Issues**:
   - Reduce batch sizes or use CPU-only mode
   - Clear GPU cache: `torch.cuda.empty_cache()`

2. **Missing Dependencies**:
   - Install all requirements: `pip install -r requirements.txt`
   - Ensure PyTorch is compatible with your CUDA version

3. **LLM Connection Issues**:
   - Verify Ollama is running: `ollama list`
   - Check the model is available: `ollama pull llama2:7b`

4. **Dataset Issues**:
   - Ensure required columns exist in the dataset
   - Verify the target column is named appropriately ('dropout' or 'Target')

### Debugging

For detailed logging:
```bash
# Check the generated log files
tail -f logs/agent_timeseries_[dataset].log
```

## Citation

If you use this code in your research, please cite:

```
[TODO: Add your specific citation here when the paper is published]
```

## License

[Specify the license here if applicable]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For questions or issues, please open an issue in the GitHub repository.

## Dataset Setup

For detailed instructions on downloading and setting up datasets, see [DATASET_SETUP.md](DATASET_SETUP.md).