"""Classification, calibration and selective-prediction metrics."""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss, log_loss

COVERAGES = (1.0, 0.95, 0.9, 0.8, 0.5)


def entropy(p):
    p = np.clip(p, 1e-12, 1)
    return -(p * np.log(p)).sum(1)


def ece(p, y, n_bins=15):
    """Top-label expected calibration error, equal-width bins."""
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def reliability_bins(p, y, n_bins=15):
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append((lo, hi, int(m.sum()), correct[m].mean() if m.any() else np.nan,
                     conf[m].mean() if m.any() else np.nan))
    return rows


def risk_coverage(p, y, score=None):
    """Accuracy of the retained part when abstaining on the least confident inputs.

    score: higher = more confident (default max prob). Returns coverage grid, accuracy, AURC (risk = error).
    """
    if score is None:
        score = p.max(1)
    order = np.argsort(-score, kind='stable')
    correct = (p.argmax(1) == y)[order].astype(float)
    n = len(y)
    cum_acc = np.cumsum(correct) / np.arange(1, n + 1)
    cov = np.arange(1, n + 1) / n
    aurc = float(np.mean(1 - cum_acc))
    return cov, cum_acc, aurc


def acc_at_coverage(p, y, c, score=None):
    cov, acc, _ = risk_coverage(p, y, score)
    k = max(1, int(np.ceil(c * len(y)))) - 1
    return float(acc[k])


def all_metrics(p, y, score=None):
    """p: (n, 2) probabilities, y: int labels."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    p = p / p.sum(1, keepdims=True)
    yhat = p.argmax(1)
    f1s = f1_score(y, yhat, average=None, labels=[0, 1], zero_division=0)
    out = {
        'accuracy': accuracy_score(y, yhat),
        'macro_f1': f1_score(y, yhat, average='macro', zero_division=0),
        'f1_class0': f1s[0], 'f1_class1': f1s[1],
        'auc': roc_auc_score(y, p[:, 1]) if len(np.unique(y)) > 1 else np.nan,
        'nll': log_loss(y, p, labels=[0, 1]),
        'brier': brier_score_loss(y, p[:, 1]),
        'ece': ece(p, y),
        'aurc': risk_coverage(p, y, score)[2],
    }
    for c in COVERAGES[1:]:
        out[f'acc@cov{int(c * 100)}'] = acc_at_coverage(p, y, c, score)
    return out


def fit_temperature(logits, y):
    """Temperature scaling on validation logits (log-probs are fine as logits)."""
    def nll(t):
        z = logits / t
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y)), y].mean()
    r = minimize_scalar(nll, bounds=(0.05, 20), method='bounded')
    return float(r.x)


def apply_temperature(logits, t):
    z = logits / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def to_logits(p):
    return np.log(np.clip(p, 1e-12, 1))
