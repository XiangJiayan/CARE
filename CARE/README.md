# CARE source code

This anonymous package contains the CARE implementation and the following baselines:

- Popularity
- TF-IDF
- Sentence-T5 Direct
- BPR-MF
- NeuMF
- LightGCN
- MPGCF
- TSSGCF
- ST5-TIGER

## Setup

Use 64-bit Python 3.12 and install `requirements.txt`. Place the six ProgrammableWeb CSV files in `data/programmableweb/`:

`api.csv`, `mashup.csv`, `category.csv`, `mashupapi.csv`, `apicate.csv`, and `mashupcate.csv`.

The Sentence-T5 model is downloaded from its public model identifier by default. An existing local model can instead be selected through the `CARE_SENTENCE_ENCODER` environment variable.

## Run

Run `run_experiment.py` from the project root to execute CARE. Each baseline has its own `run_experiment.py` under `baselines/<method>/`.

All output artifacts are written beneath the local `outputs/` directory, which is intentionally absent from this source-only package.
