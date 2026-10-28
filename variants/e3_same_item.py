"""Stateless same-item residual correction of the frozen E2 predictor.

Only supplied attributes/labels are used. Item text is compared, never emitted.
The matching label remains in the E1b estimate of its observed subject; this is
an in-sample, shrunken residual, not a leave-one-out residual.
"""

import math

from submission_e2.model import predict as predict_e2, e1b, clip, sigmoid

ITEM_FIELDS = ("item_content", "item_features", "interactors", "benchmark_id")
VARIANCE_FLOOR = 0.05


def same_item_stats(input, labeled=None):
    """Return (count, probability residual sum, approximate logit residual sum).

    Exact equality of all four fields is required, including complete content.
    Target-subject entries never contribute a residual. E1b q_t uses all valid
    same-benchmark evidence, including that subject's matching label and any
    target-subject labels, exactly as ordinary E1b does.
    """
    try:
        subject, item = input
        if (not isinstance(subject, dict) or not isinstance(item, dict)
                or not all(k in item for k in ITEM_FIELDS)
                or not isinstance(item["benchmark_id"], str) or not item["benchmark_id"]
                or not isinstance(labeled, (list, tuple))):
            return 0, 0.0, 0.0
        residuals, logit_residuals = [], []
        # Local only: repeated labels for the same subject need one E1b scan.
        estimates = []
        for entry in labeled:
            try:
                (other_subject, other_item), y = entry
                if (y not in (0, 1) or not isinstance(other_subject, dict)
                        or other_subject == subject or not isinstance(other_item, dict)
                        or not all(k in other_item and other_item[k] == item[k] for k in ITEM_FIELDS)):
                    continue
                q = next((q for s, q in estimates if s == other_subject), None)
                if q is None:
                    q = e1b([other_subject, item], labeled)
                    estimates.append((other_subject, q))
                residual = float(y) - q
                residuals.append(residual)
                logit_residuals.append(residual / max(q * (1 - q), VARIANCE_FLOOR))
            except Exception:
                continue
        return len(residuals), math.fsum(residuals), math.fsum(logit_residuals)
    except Exception:
        return 0, 0.0, 0.0


def apply_shift(baseline, stats, shrinkage, eta, mode):
    """A failure or absent matching evidence returns the E2 value exactly."""
    try:
        n, residual, logit_residual = stats
        if not n:
            return baseline
        if (not math.isfinite(shrinkage) or shrinkage <= 0
                or not math.isfinite(eta) or eta < 0 or mode not in ("probability", "logit")):
            return baseline
        if mode == "probability":
            return clip(baseline + eta * residual / (n + shrinkage))
        offset = eta * logit_residual / (n + shrinkage)
        return clip(sigmoid(math.log(baseline / (1 - baseline)) + offset))
    except Exception:
        return baseline


def predict(input, labeled=None, *, shrinkage=4.0, eta=0.5, mode="probability"):
    """Return a finite E3 probability; all caches are confined to this call."""
    baseline = predict_e2(input, labeled)
    try:
        return apply_shift(baseline, same_item_stats(input, labeled), shrinkage, eta, mode)
    except Exception:
        return baseline
