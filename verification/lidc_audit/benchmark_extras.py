"""Numbers for the comparison table that are derived from the regenerated test nodules."""
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score

s = json.loads(Path(sys.argv[1]).read_text())
t = s["test_nodules"]
y = np.array([x[2] for x in t])
d = np.array([x[3] for x in t])
c = collections.Counter(x[0] for x in t)
cm = collections.Counter(x[0] for x in t if x[2] == 1)
Z = stats.norm.ppf(0.975)


def wilson(k, n):
    p = k / n
    den = 1 + Z * Z / n
    mid = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return [mid - h, mid + h]


def hm(a, n1, n0):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))


def logit_ci(a, se):
    lg, sd = math.log(a / (1 - a)), se / (a * (1 - a))
    return [1 / (1 + math.exp(-(lg - Z * sd))), 1 / (1 + math.exp(-(lg + Z * sd)))]


m = s["size_model_test"]
tp, tn, fp, fn = m["tp"], m["tn"], m["fp"], m["fn"]
a = m["auc"]
out = {
    "top_patient": c.most_common(1)[0], "top_patient_malignant": cm[c.most_common(1)[0][0]], "second_patient": c.most_common(2)[1],
    "patients_test": len(c), "patients_with_one_nodule": sum(v == 1 for v in c.values()),
    "average_precision": float(average_precision_score(y, d)),
    "auc": a, "auc_pairs": a * 180, "auc_ci_logit": logit_ci(a, hm(a, 20, 9)),
    "accuracy": [(tp + tn) / 29, wilson(tp + tn, 29)], "sensitivity": [tp / 20, wilson(tp, 20)], "specificity": [tn / 9, wilson(tn, 9)],
    "ppv": [tp / (tp + fp), wilson(tp, tp + fp)], "npv": [tn / (tn + fn), wilson(tn, tn + fn)],
}
print(json.dumps(out, indent=1))
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
