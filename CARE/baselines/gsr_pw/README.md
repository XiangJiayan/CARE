# GSR_REC ProgrammableWeb baseline

This folder contains a clean-room task adaptation of *Service Recommendations
for Mashup Based on Generation Model*. It implements the paper's bi-GRU
requirement/category encoders, attention-based GRU decoder, training-only
semantic retrieval, current/historical probability fusion, teacher forcing,
and Self-Critical Sequence Training (SCST).

For a fair Top-K comparison, the reported method is `GSR_REC`: the API ranking
is obtained from the first decoder distribution conditioned on `<SOS>`, as
defined in the paper. Whole Mashups are split with `split_seed=2026`, and a
test or validation Mashup retrieves its nearest historical requirement only
from the training split.

The official repository does not distribute its original BERT/GloVe artifacts
or a complete environment specification. This portable reproduction therefore
uses Sentence-T5 plus training-fitted whitening for semantic retrieval and
retains the paper's 300-dimensional trainable word-embedding capacity for the
generator. This substitution is recorded in every generated report.

Run `run_experiment.py` in PyCharm. If the dataset or encoder is stored
elsewhere, set `CARE_DATA_DIR` and `CARE_SENTENCE_ENCODER`. Experimental
artifacts are written outside this blind-review source directory by default.

The official repository is not vendored because it does not state a software
license. This implementation was written from the published method description.
