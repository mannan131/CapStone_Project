"""Recommender orchestrator: SHAP causes + rules -> RetentionRecommendation."""

from __future__ import annotations

import ast
import operator as _operators
from dataclasses import asdict, dataclass

from src.common.config import load_config, schema
from src.common.logging import get_logger
from src.recommendation_engine.rules import CauseActionRuleEngine

log = get_logger("recommender")


def shap_feature_to_cause(engine: CauseActionRuleEngine) -> dict[str, str]:
    """SHAP feature -> cause map derived at runtime from the rules YAML.

    Fused SHAP names carry a transformer prefix (e.g. ``num__x``); callers strip
    it before lookup, so the map keys are the raw rule condition features.
    """
    return engine.feature_to_cause_map()


_ALLOWED_OPS = {
    ast.Add: _operators.add,
    ast.Sub: _operators.sub,
    ast.Mult: _operators.mul,
    ast.Div: _operators.truediv,
    ast.USub: _operators.neg,
}


def eval_priority_formula(formula: str, variables: dict[str, float]) -> float:
    """Safely evaluate an arithmetic priority formula over named variables."""
    tree = ast.parse(formula, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"Unknown variable in priority formula: {node.id}")
            return float(variables[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression in priority formula: {formula!r}")

    return float(_eval(tree))


@dataclass
class RetentionRecommendation:
    customer_id: str
    churn_probability: float
    top_causes: list[str]
    recommended_actions: list[dict]
    confidence_score: float
    priority_score: float

    def to_dict(self):
        return asdict(self)


class RetentionRecommender:
    def __init__(self, config: dict | None = None, engine: CauseActionRuleEngine | None = None):
        self.config = config or load_config()
        self.engine = engine or CauseActionRuleEngine(self.config)
        rec = self.config.get("recommendation_engine", self.config.get("recommendation", {}))
        self.max_actions = int(rec.get("max_actions_per_customer", 10))
        self.priority_formula = str(
            rec.get("priority_formula", "churn_probability * customer_lifetime_value")
        )
        s = schema(self.config)
        self.clv_col = s.get("clv_column", "customer_lifetime_value")

    def recommend(
        self,
        customer_id: str,
        churn_probability: float,
        features: dict,
        shap_top_features: list[dict] | None = None,
    ) -> RetentionRecommendation:
        matched = self.engine.evaluate(features)
        # Boost causes corroborated by SHAP (risk-increasing features), using the
        # feature->cause map derived from the rules YAML at runtime.
        feature_to_cause = shap_feature_to_cause(self.engine)
        shap_causes: list[str] = []
        if shap_top_features:
            for f in shap_top_features:
                name = f.get("feature", "")
                base = name.split("__")[-1]
                if f.get("direction") == "increases_churn" and base in feature_to_cause:
                    shap_causes.append(feature_to_cause[base])

        causes = [m["cause"] for m in matched]
        for c in shap_causes:
            if c not in causes:
                causes.append(c)

        # Rank actions: rule priority (1 best) + SHAP corroboration bonus
        rule_actions: dict[str, list[dict]] = {}
        for rule in self.engine.rules:
            rule_actions.setdefault(rule["cause"], []).extend(rule.get("actions", []))
        ranked: list[dict] = []
        matched_causes = {m["cause"] for m in matched}
        for m in matched:
            bonus = 1 if m["cause"] in shap_causes else 0
            for a in m["actions"]:
                ranked.append(
                    {
                        "cause": m["cause"],
                        "action": a["action"],
                        "detail": a.get("detail", ""),
                        "rank_score": float(a.get("priority", 3)) - 0.5 * bonus,
                        "shap_corroborated": bool(bonus),
                    }
                )
        # SHAP-flagged causes whose thresholds didn't fire still get their playbook actions.
        for cause in shap_causes:
            if cause not in matched_causes:
                for a in rule_actions.get(cause, []):
                    ranked.append(
                        {
                            "cause": cause,
                            "action": a["action"],
                            "detail": a.get("detail", ""),
                            "rank_score": float(a.get("priority", 3)) - 0.5,
                            "shap_corroborated": True,
                        }
                    )
        if not ranked:
            for a in self.engine.fallback:
                ranked.append(
                    {
                        "cause": "general",
                        "action": a["action"],
                        "detail": a.get("detail", ""),
                        "rank_score": float(a.get("priority", 3)),
                        "shap_corroborated": False,
                    }
                )
        ranked.sort(key=lambda r: r["rank_score"])
        ranked = ranked[: max(1, self.max_actions)]

        clv = float(features.get(self.clv_col, features.get("customer_lifetime_value", 0)) or 0)
        try:
            priority = round(
                eval_priority_formula(
                    self.priority_formula,
                    {"churn_probability": float(churn_probability), "customer_lifetime_value": clv},
                ),
                2,
            )
        except ValueError as e:
            log.info("priority_formula_fallback", error=str(e))
            priority = round(float(churn_probability) * clv, 2)
        confidence = round(min(0.95, 0.5 + 0.1 * len(matched) + 0.05 * len(shap_causes)), 3)
        rec = RetentionRecommendation(
            customer_id=customer_id,
            churn_probability=round(float(churn_probability), 2),
            top_causes=causes,
            recommended_actions=ranked,
            confidence_score=confidence,
            priority_score=priority,
        )
        log.info("recommended", customer_id=customer_id, causes=causes, priority=priority)
        return rec
