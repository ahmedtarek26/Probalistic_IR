# Probabilistic_IR

## Installation

Use pip to install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python main.py
```

## Offline Setup

If the environment does not have internet access, download the required data
beforehand and copy it to the target machine.

1. **NLTK resources**
   ```python
   import nltk
   nltk.download("punkt")
   nltk.download("stopwords")
   ```
   Copy the resulting `nltk_data` folder (usually located in your home
   directory) to the offline environment.

2. **20 Newsgroups dataset**
   ```python
   from sklearn.datasets import fetch_20newsgroups
   fetch_20newsgroups(subset="all")
   ```
   The dataset will be stored in `scikit_learn_data`. Set the environment
   variable `SKLEARN_DATA` to this directory so that `dataset.py` can find the
   files without attempting to download them.
