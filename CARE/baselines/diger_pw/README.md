# DIGER-PW baseline

This folder contains the ProgrammableWeb adaptation of DIGER. It follows the
official corrected `gradient-fix` principle: deterministic hard semantic IDs
are used in the forward pass and current soft assignments carry gradients to
the RQ-VAE tokenizer. Frequency-based uncertainty decay (FrqUD) applies Gumbel
exploration to over-used codes during training and is disabled for evaluation.
The reconstruction and quantization terms retain the `768` and `32` scaling
used to pretrain the shared ProgrammableWeb RQ-VAE, preventing a normalization
change from overwhelming the pretrained semantic structure.

ProgrammableWeb contains unordered sets of APIs rather than chronological user
histories. Each `(Mashup, called API)` pair is therefore a positive generation
example conditioned on the frozen Sentence-T5 Mashup representation. This is a
method-level task adaptation, not an unchanged run of the sequential code.

Run `run_experiment.py` in PyCharm. Set `CARE_WORKSPACE_DIR` when the prepared
CARE workspace is not in the default location. Experimental outputs are written
outside the blind-review source tree by default.
