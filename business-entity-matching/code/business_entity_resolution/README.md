# Business Entity Resolution Pipeline

Modular entity resolution & matching pipeline for business entities.

## Directory Structure

- `src/profiling/`: Exploratory analysis and statistical profiling of datasets.
- `src/preprocessing/`: Text cleaning, normalization, tokenization, and canonicalization.
- `src/blocking/`: Candidate pair indexing and blocking mechanisms (LSH, TF-IDF, Rule-based).
- `src/features/`: Similarity metrics computation, string distances, and embedding extraction.
- `src/models/`: Matching classifiers, neural architectures, or clustering algorithms.
- `src/evaluation/`: Metric computations (Precision, Recall, F1), benchmark analysis, error auditing.

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python -m src.models.train
```
