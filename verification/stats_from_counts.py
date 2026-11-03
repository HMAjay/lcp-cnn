"""Uncertainty estimates derived only from the counts in test_metrics.json and the split table."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

M = {
    "accuracy": 0.7586206896551724,
    "auc": 0.9138888888888889,
    "auprc": 0.9582989336085312,
    "best_threshold": 0.01433563232421875,
    "sensitivity": 0.7,
    "specificity": 0.8888888888888888,
    "precision": 0.9333333333333333,
    "brier": 0.22960195414208132,
    "tp": 14,
    "tn": 8,
    "fp": 1,
    "fn": 6,
}
Z = stats.norm.ppf(0.975)


def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def hanley_mcneil_se(a, n1, n0):
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))


def logit_ci(a, se):
    lg = math.log(a / (1 - a))
    s = se / (a * (1 - a))
    return 1 / (1 + math.exp(-(lg - Z * s))), 1 / (1 + math.exp(-(lg + Z * s)))


def exact_u_sf(n1, n0, u_obs):
    """P(U >= u_obs) under H0 from the exact null distribution of U (no ties)."""
    # Recursion for the Mann-Whitney null distribution: f(i, j, u) = f(i-1, j, u-j) + f(i, j-1, u).
    table = [[None] * (n0 + 1) for _ in range(n1 + 1)]
    for i in range(n1 + 1):
        for j in range(n0 + 1):
            if i == 0 or j == 0:
                table[i][j] = np.array([1.0])
                continue
            a = np.zeros(i * j + 1)
            prev_i = table[i - 1][j]
            prev_j = table[i][j - 1]
            a[j : j + len(prev_i)] += prev_i
            a[: len(prev_j)] += prev_j
            table[i][j] = a
    pmf = table[n1][n0] / table[n1][n0].sum()
    return float(pmf[int(math.ceil(u_obs)) :].sum())


n1, n0 = M["tp"] + M["fn"], M["tn"] + M["fp"]
n = n1 + n0
auc = M["auc"]
u = auc * n1 * n0
se = hanley_mcneil_se(auc, n1, n0)
lo, hi = logit_ci(auc, se)
wald = (auc - Z * se, min(1.0, auc + Z * se))

prev_tt = (45 + 12) / (132 + 35)
fisher = stats.fisher_exact([[20, 9], [57, 110]])

val_auc = 262 / 276
val_se = hanley_mcneil_se(val_auc, 12, 23)

prev = n1 / n
brier_ref = prev * (1 - prev)
out = {
    "n1": n1,
    "n0": n0,
    "U": u,
    "auc": auc,
    "auc_se_hm": se,
    "auc_ci_logit": [lo, hi],
    "auc_ci_wald": list(wald),
    "exact_p_one_sided_vs_chance_u_ceil": exact_u_sf(n1, n0, u),
    "exact_p_two_sided": min(1.0, 2 * exact_u_sf(n1, n0, u)),
    "accuracy_ci": wilson(22, 29),
    "sensitivity_ci": wilson(14, 20),
    "specificity_ci": wilson(8, 9),
    "precision_ci": wilson(14, 15),
    "npv": 8 / 14,
    "npv_ci": wilson(8, 14),
    "test_prevalence": prev,
    "train_val_prevalence": prev_tt,
    "fisher_test_vs_trainval": {"odds_ratio": float(fisher[0]), "p": float(fisher[1])},
    "brier_reference_prevalence": brier_ref,
    "brier_skill_vs_prevalence": 1 - M["brier"] / brier_ref,
    "pos_weight": 87 / 45,
    "val_auc": val_auc,
    "val_auc_ci_logit": list(logit_ci(val_auc, val_se)),
    "all_malignant_accuracy": 20 / 29,
}
print(json.dumps(out, indent=2))
Path(sys.argv[1]).write_text(json.dumps(out, indent=2))
