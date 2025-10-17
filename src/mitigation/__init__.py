"""Bias Mitigation Module"""

from .preprocessing import Reweighing, DisparateImpactRemover, apply_preprocessing_mitigation
from .postprocessing import CalibratedEqualizedOdds, RejectOptionClassification, ThresholdOptimizer, apply_postprocessing_mitigation
from .advisor import MitigationAdvisor, get_mitigation_recommendations

__all__ = [
    'Reweighing', 'DisparateImpactRemover', 'CalibratedEqualizedOdds',
    'RejectOptionClassification', 'ThresholdOptimizer', 'MitigationAdvisor',
    'apply_preprocessing_mitigation', 'apply_postprocessing_mitigation',
    'get_mitigation_recommendations'
]
