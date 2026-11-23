"""Flags that record which research stages exist in this repository.

Keep a flag false until the corresponding function returns real results and
tests cover those results. The API and UI read these flags.
"""

TRAINING_IMPLEMENTED = True
INFERENCE_IMPLEMENTED = True
FEATURES_IMPLEMENTED = True
EVALUATION_IMPLEMENTED = True
