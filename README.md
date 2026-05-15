# Analogy Evaluation Baselines

Baseline for evaluating analogy models.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the root directory and add your Gemini API key:

```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

## Data Download

Before running the baselines, download the dataset using the following script:

```python
from huggingface_hub import snapshot_download
from datasets import load_dataset

LOCAL_DIR = "challenge-dataset"

# 1. Download everything (parquet and video folders) into one local directory
snapshot_download(
    repo_id="analogy-evaluation/challenge-dataset",
    repo_type="dataset",
    local_dir=LOCAL_DIR,
)

# 2. Load the splits from the downloaded parquet and save in load_from_disk format
ds = load_dataset(LOCAL_DIR)
ds.save_to_disk(LOCAL_DIR)
```

## Running Baselines

You can run the baselines using the scripts in the `baselines/` directory:

```bash
python baselines/baseline_text.py
python baselines/baseline_video.py
```
