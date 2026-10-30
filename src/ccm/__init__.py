"""
Conformal Concept Models (CCM)
Framework for rigorous uncertainty quantification, conformal prediction, and intervention auditing
in Concept Bottleneck Models (CBMs).
"""

__version__ = "0.1.0"

from ccm.conformal import ConformalCalibrator, MondrianConformalCalibrator
from ccm.scores import (
    APSScore,
    RAPSScore,
    HingeScore,
    MarginScore,
    NormalizedScore,
    ConceptResidualScore,
    get_score_function,
)
from ccm.sets import PredictionSet, ConceptPredictionSet
from ccm.metrics import (
    empirical_coverage,
    average_set_size,
    conditional_coverage_by_group,
    concept_accuracy,
    intervention_efficiency_auc,
    nn_leakage_score,
)
from ccm.interventions import (
    InterventionPolicy,
    RandomInterventionPolicy,
    UncertaintyInterventionPolicy,
    GreedyConformalInterventionPolicy,
    OracleOptimalInterventionPolicy,
    apply_interventions,
)
from ccm.utils import (
    set_seed,
    load_config,
    get_device,
    VaultLogger,
)

__all__ = [
    "ConformalCalibrator",
    "MondrianConformalCalibrator",
    "APSScore",
    "RAPSScore",
    "HingeScore",
    "MarginScore",
    "NormalizedScore",
    "ConceptResidualScore",
    "get_score_function",
    "PredictionSet",
    "ConceptPredictionSet",
    "empirical_coverage",
    "average_set_size",
    "conditional_coverage_by_group",
    "concept_accuracy",
    "intervention_efficiency_auc",
    "nn_leakage_score",
    "InterventionPolicy",
    "RandomInterventionPolicy",
    "UncertaintyInterventionPolicy",
    "GreedyConformalInterventionPolicy",
    "OracleOptimalInterventionPolicy",
    "apply_interventions",
    "set_seed",
    "load_config",
    "get_device",
    "VaultLogger",
]
