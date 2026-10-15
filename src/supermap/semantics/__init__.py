"""Bayesian semantic fusion (Sec. IV-B4, Eq. 10). REQ-040..REQ-045."""
# REQ-040, REQ-041, REQ-042, REQ-043, REQ-044: public API of the semantic fusion module
from supermap.semantics.confusion import (
    confusion_from_config,
    confusion_matrix_from_config,
    validate_confusion,
)
from supermap.semantics.fusion import bayes_update, representative_label, uniform_prior
from supermap.semantics.update import (
    init_from_first_observation,
    is_low_confidence,
    posterior_belief,
    remove_low_confidence,
    semantic_update,
)

__all__ = [
    "bayes_update",
    "confusion_from_config",
    "confusion_matrix_from_config",
    "init_from_first_observation",
    "is_low_confidence",
    "posterior_belief",
    "remove_low_confidence",
    "representative_label",
    "semantic_update",
    "uniform_prior",
    "validate_confusion",
]
