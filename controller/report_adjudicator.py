"""
report_adjudicator.py - Adjudicate escalated flows from the switch report alone.

The model is a gradient-boosted tree ensemble trained on exactly the twelve
registers the 70-byte report carries, exported as LightGBM JSON and scored
here in plain Python so the controller needs no ML runtime.
"""
import json

REPORT_FEATURES = ("byte_count", "mean_len", "mean_iat_us", "min_len", "max_len",
                   "fwd_pkts", "fwd_bytes", "iat_min_us", "iat_max_us",
                   "duration_us", "len_dev", "first_len")


def _leaf_value(node, x):
    while "leaf_value" not in node:
        v = x[node["split_feature"]]
        missing = node.get("missing_type", "None")
        if (missing == "Zero" and v == 0.0) or (missing == "NaN" and v != v):
            node = node["left_child"] if node["default_left"] else node["right_child"]
        elif v <= node["threshold"]:
            node = node["left_child"]
        else:
            node = node["right_child"]
    return node["leaf_value"]


class ReportAdjudicator:
    """Raw-margin score of a report and the calibrated block decision."""

    def __init__(self, path):
        spec = json.loads(open(path).read())
        self.trees = [t["tree_structure"] for t in spec["model"]["tree_info"]]
        self.features = spec["features"]
        self.threshold = spec["threshold"]

    def score(self, report):
        x = [float(report[f]) for f in self.features]
        return sum(_leaf_value(t, x) for t in self.trees)

    def block(self, report):
        return self.threshold is not None and self.score(report) >= self.threshold
