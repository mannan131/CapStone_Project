"""Config-driven cause -> action rule engine (YAML, no code changes needed)."""

from __future__ import annotations

import operator

from src.common.config import load_config, load_rules, schema
from src.common.logging import get_logger

log = get_logger("rules")

OPS = {
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "eq": operator.eq,
}


class CauseActionRuleEngine:
    def __init__(self, config: dict | None = None, rules: dict | None = None):
        self.config = config or load_config()
        self.rules_doc = rules or load_rules(config=self.config)
        self.rules: list[dict] = self.rules_doc.get("rules", [])
        self.fallback: list[dict] = self.rules_doc.get("fallback_actions", [])

    def evaluate(self, features: dict) -> list[dict]:
        """Return matched {cause, actions} entries for a customer feature dict."""
        matched = []
        for rule in self.rules:
            cond = rule.get("condition", {})
            feat, op, thr = cond.get("feature"), cond.get("operator"), cond.get("threshold")
            if feat not in features or op not in OPS:
                continue
            try:
                val = float(features[feat])
                thr_f = float(thr)
            except (TypeError, ValueError):
                continue
            if OPS[op](val, thr_f):
                matched.append(
                    {
                        "cause": rule["cause"],
                        "trigger_value": val,
                        "actions": rule.get("actions", []),
                    }
                )
        log.info("rules_evaluated", matched=[m["cause"] for m in matched])
        return matched

    def reload(self, rules: dict) -> None:
        self.rules_doc = rules
        self.rules = rules.get("rules", [])
        self.fallback = rules.get("fallback_actions", [])

    def feature_to_cause_map(self) -> dict[str, str]:
        """Derive {condition feature -> cause} straight from the YAML rules."""
        mapping: dict[str, str] = {}
        for rule in self.rules:
            feat = rule.get("condition", {}).get("feature")
            if feat and rule.get("cause"):
                mapping.setdefault(feat, rule["cause"])
        return mapping

    def condition_features(self) -> list[str]:
        """Every feature referenced by any rule condition, in first-seen order."""
        seen: list[str] = []
        for rule in self.rules:
            feat = rule.get("condition", {}).get("feature")
            if feat and feat not in seen:
                seen.append(feat)
        return seen


def extra_rule_features(config: dict | None = None) -> list[str]:
    """Rule-referenced features absent from the config data schema.

    Single runtime source for "columns the rules need but the schema doesn't
    define" — used both by feature fusion (model inputs) and the synthetic
    generator (data emission), so no column name is hardcoded in code.
    """
    cfg = config or load_config()
    s = schema(cfg)
    fusion_cfg = cfg.get("feature_fusion", {})
    known = (
        set(s.get("behavioral_features", []))
        | set(s.get("transactional_features", []))
        | set(s.get("categorical_features", []))
        | {s.get("id_column"), s.get("target_column"), s.get("clv_column")}
    )
    known.add(fusion_cfg.get("image_score_column", fusion_cfg.get("image_score_col")))
    try:
        rules_doc = load_rules(config=cfg)
    except Exception:
        return []
    return [
        feat
        for rule in rules_doc.get("rules", [])
        if (feat := rule.get("condition", {}).get("feature")) and feat not in known
    ]
