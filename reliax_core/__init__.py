"""Reliax reliability-envelope components used by the whitepaper evaluation.

Methods only: conformal prediction (split + Mondrian), Venn-Abers calibration,
conformal test martingale, kNN OOD, PSI drift, error auditor, composite score
and subjective-logic fusion. The Reliax product layer (engine, routing policies,
audit chain, API) is not part of this release.
"""
from .conformal import ConformalCalibrator
from .venn_abers import VennAbersCalibrator
from .martingale import ConformalMartingale, WATCH_THRESHOLD, ALARM_THRESHOLD
from .ood import KNNOODDetector
from .auditor import ErrorAuditor
from .drift import PSIMonitor
from .fairness import MondrianConformal, ImpactMonitor, coverage_audit
from .scoring import reliability_score
from .sl_fusion import fuse_signals, averaging_fusion
