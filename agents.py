from __future__ import annotations

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from models import Portfolio, ClientProfile, Scenario
from llm_client import LLMClient


class MonitorAgent:
    """Rule-based monitoring: drift, thresholds, etc."""

    def __init__(self, drift_threshold: float = 0.05):
        self.drift_threshold = drift_threshold

    def compute_allocation_by_class(self, portfolio: Portfolio) -> Dict[str, float]:
        df = portfolio.to_dataframe()
        allocation = df.groupby("asset_class")["market_value"].sum()
        allocation = allocation / allocation.sum()
        return allocation.to_dict()

    def analyze_drift(
        self,
        portfolio: Portfolio,
        client_profile: ClientProfile
    ) -> Dict[str, Any]:
        current_alloc = self.compute_allocation_by_class(portfolio)
        target_alloc = client_profile.target_allocation

        drift = {}
        total_drift = 0.0
        classes = set(current_alloc.keys()) | set(target_alloc.keys())
        for c in classes:
            cur = current_alloc.get(c, 0.0)
            tgt = target_alloc.get(c, 0.0)
            diff = cur - tgt
            drift[c] = diff
            total_drift += abs(diff)

        needs_rebalancing = total_drift > self.drift_threshold

        return {
            "needs_rebalancing": needs_rebalancing,
            "total_drift": total_drift,
            "drift_by_class": drift,
            "current_allocation": current_alloc,
            "target_allocation": target_alloc,
        }


class AnalystAgent:
    """LLM-powered reasoning: tax, risk, market context, constraints."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def analyze(
        self,
        portfolio: Portfolio,
        client_profile: ClientProfile,
        monitor_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        portfolio_df = portfolio.to_dataframe()

        prompt = self._build_prompt(portfolio_df, client_profile, monitor_output)
        content = self.llm.chat(
            role="analyst",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior portfolio analyst. "
                        "Given a portfolio, drift metrics, and a client profile, "
                        "you must output a concise JSON object with fields: "
                        "{'market_context': {...}, 'risk_bounds': {...}, "
                        "'tax_considerations': {...}, 'notes': '...'}."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        # Light JSON-ish parsing: expect the model to output valid JSON.
        try:
            import json
            parsed = json.loads(content)
        except Exception:
            # fallback: minimal structure
            parsed = {
                "market_context": {"comment": "LLM response could not be parsed as JSON."},
                "risk_bounds": self._derive_risk_bounds(client_profile),
                "tax_considerations": {"comment": "Tax logic fallback: use tax bracket only."},
                "notes": content,
            }

        # Ensure minimum fields exist
        if "risk_bounds" not in parsed:
            parsed["risk_bounds"] = self._derive_risk_bounds(client_profile)

        parsed["portfolio_df"] = portfolio_df
        return parsed

    def _build_prompt(
        self,
        portfolio_df: pd.DataFrame,
        client_profile: ClientProfile,
        monitor_output: Dict[str, Any],
    ) -> str:
        portfolio_summary = portfolio_df.to_dict(orient="records")
        return (
            "Portfolio positions (list of dicts):\n"
            f"{portfolio_summary}\n\n"
            "Client profile:\n"
            f"{client_profile}\n\n"
            "Drift metrics:\n"
            f"{monitor_output}\n\n"
            "Think step by step about:\n"
            "- Market context (volatility relevance, sector concentration, any obvious red flags)\n"
            "- Risk bounds (max equity %, concentration constraints, etc.)\n"
            "- Tax considerations (high brackets should prefer fewer realized gains, etc.)\n"
            "Return ONLY a JSON object as described, no extra text."
        )

    def _derive_risk_bounds(self, client_profile: ClientProfile) -> Dict[str, Any]:
        profile = client_profile.risk_profile
        if profile == "conservative":
            max_equity = 0.5
        elif profile == "moderate":
            max_equity = 0.7
        else:
            max_equity = 0.9
        return {"max_equity_weight": max_equity}


class StrategistAgent:
    """
    LLM planning + deterministic trade construction.

    LLM decides scenario styles and aggressiveness; we convert that into
    concrete target allocations and trades.
    """

    def __init__(self, llm: LLMClient, analyst_agent: AnalystAgent):
        self.llm = llm
        self.analyst_agent = analyst_agent

    def generate_scenarios(
        self,
        portfolio: Portfolio,
        client_profile: ClientProfile,
        monitor_output: Dict[str, Any],
        analyst_output: Dict[str, Any],
    ) -> List[Scenario]:
        portfolio_df = analyst_output["portfolio_df"]
        risk_bounds = analyst_output.get("risk_bounds", {})

        prompt = self._build_prompt(portfolio_df, client_profile, monitor_output, risk_bounds)
        raw = self.llm.chat(
            role="strategist",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a portfolio strategist. "
                        "Given drift information, client profile, and risk bounds, "
                        "propose 3 scenarios in JSON: "
                        "{'scenarios': ["
                        "{'name': 'A - Conservative', 'aggressiveness': 0.3},"
                        "{'name': 'B - Moderate', 'aggressiveness': 0.6},"
                        "{'name': 'C - Aggressive', 'aggressiveness': 1.0}"
                        "]}. "
                        "aggressiveness is a float between 0 and 1 representing fraction of drift corrected."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        import json
        try:
            parsed = json.loads(raw)
            scenario_specs = parsed.get("scenarios", [])
        except Exception:
            scenario_specs = [
                {"name": "A - Conservative", "aggressiveness": 0.3},
                {"name": "B - Moderate", "aggressiveness": 0.6},
                {"name": "C - Aggressive", "aggressiveness": 1.0},
            ]

        drift_by_class = monitor_output["drift_by_class"]
        scenarios: List[Scenario] = []

        for spec in scenario_specs:
            name = spec.get("name", "Unnamed Scenario")
            level = float(spec.get("aggressiveness", 0.5))
            target_alloc = self._build_target_allocation(monitor_output, level)
            trades = self._build_trades_for_allocation(portfolio_df, target_alloc)
            estimated_tax = self._estimate_tax_for_trades(portfolio_df, trades, client_profile)
            risk_reduction_pct = self._estimate_risk_reduction(level, drift_by_class)

            scenarios.append(Scenario(
                name=name,
                target_allocation=target_alloc,
                trades=trades,
                estimated_tax=estimated_tax,
                risk_reduction_pct=risk_reduction_pct,
            ))

        return scenarios

    def _build_prompt(
        self,
        portfolio_df: pd.DataFrame,
        client_profile: ClientProfile,
        monitor_output: Dict[str, Any],
        risk_bounds: Dict[str, Any],
    ) -> str:
        portfolio_summary = portfolio_df.to_dict(orient="records")
        return (
            "Design 3 rebalancing scenarios for this client.\n\n"
            f"Portfolio:\n{portfolio_summary}\n\n"
            f"Client profile:\n{client_profile}\n\n"
            f"Drift metrics:\n{monitor_output}\n\n"
            f"Risk bounds:\n{risk_bounds}\n\n"
            "Each scenario should differ by how aggressively it corrects drift. "
            "Return ONLY JSON as described in the system message; no explanations."
        )

    def _build_target_allocation(self, monitor_output: Dict[str, Any], level: float) -> Dict[str, float]:
        current = monitor_output["current_allocation"]
        target = monitor_output["target_allocation"]
        classes = set(current.keys()) | set(target.keys())
        new_alloc = {}
        for c in classes:
            cur = current.get(c, 0.0)
            tgt = target.get(c, 0.0)
            drift = cur - tgt
            new_alloc[c] = cur - level * drift
        total = sum(new_alloc.values())
        if total > 0:
            for c in new_alloc:
                new_alloc[c] = new_alloc[c] / total
        return new_alloc

    def _build_trades_for_allocation(
        self,
        portfolio_df: pd.DataFrame,
        target_alloc: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        total_value = portfolio_df["market_value"].sum()
        trades: List[Dict[str, Any]] = []

        current_alloc = (
            portfolio_df.groupby("asset_class")["market_value"].sum() / total_value
        ).to_dict()

        desired_values = {c: w * total_value for c, w in target_alloc.items()}
        current_values = {
            c: v * total_value for c, v in current_alloc.items()
        }

        class_deltas = {
            c: desired_values.get(c, 0.0) - current_values.get(c, 0.0)
            for c in set(desired_values.keys()) | set(current_values.keys())
        }

        for asset_class, delta_value in class_deltas.items():
            class_df = portfolio_df[portfolio_df["asset_class"] == asset_class]
            if class_df.empty:
                continue
            class_total_value = class_df["market_value"].sum()
            if class_total_value == 0:
                continue

            for _, row in class_df.iterrows():
                weight_in_class = row["market_value"] / class_total_value
                trade_value = delta_value * weight_in_class
                if abs(trade_value) < 1e-6:
                    continue
                qty = trade_value / row["price"]
                action = "BUY" if trade_value > 0 else "SELL"
                trades.append({
                    "ticker": row["ticker"],
                    "asset_class": asset_class,
                    "action": action,
                    "quantity": abs(qty),
                    "value": abs(trade_value),
                })

        return trades

    def _estimate_tax_for_trades(
        self,
        portfolio_df: pd.DataFrame,
        trades: List[Dict[str, Any]],
        client_profile: ClientProfile,
    ) -> float:
        df = portfolio_df.set_index("ticker")
        tax = 0.0
        for t in trades:
            if t["action"] != "SELL":
                continue
            ticker = t["ticker"]
            qty = t["quantity"]
            if ticker not in df.index:
                continue
            price = df.loc[ticker, "price"]
            cost_basis = df.loc[ticker, "cost_basis"]
            gain_per_share = price - cost_basis
            taxable_gain = max(0.0, gain_per_share) * qty
            tax += taxable_gain * client_profile.tax_bracket
        return tax

    def _estimate_risk_reduction(self, level: float, drift_by_class: Dict[str, float]) -> float:
        return level * 0.2  # same heuristic as before


class ExplainerAgent:
    """LLM NLG: advisor-friendly natural language summaries from structured scenario data."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def build_summary_for_scenario(
        self,
        portfolio: Portfolio,
        client_profile: ClientProfile,
        monitor_output: Dict[str, Any],
        scenario: Scenario,
    ) -> str:
        prompt = self._build_prompt(portfolio, client_profile, monitor_output, scenario)
        summary = self.llm.chat(
            role="explainer",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant for financial advisors. "
                        "Explain portfolio rebalancing scenarios in clear, concise language. "
                        "Avoid jargon, be concrete, and keep it under 180 words."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return summary

    def _build_prompt(
        self,
        portfolio: Portfolio,
        client_profile: ClientProfile,
        monitor_output: Dict[str, Any],
        scenario: Scenario,
    ) -> str:
        portfolio_df = portfolio.to_dataframe()
        drift = monitor_output["drift_by_class"]
        portfolio_summary = portfolio_df.to_dict(orient="records")

        return (
            "You are explaining a rebalancing recommendation to a human financial advisor.\n\n"
            f"Client risk profile: {client_profile.risk_profile}\n"
            f"Portfolio positions: {portfolio_summary}\n\n"
            f"Current drift by asset class: {drift}\n"
            f"Scenario name: {scenario.name}\n"
            f"Scenario target allocation (by asset class): {scenario.target_allocation}\n"
            f"Scenario trades: {scenario.trades}\n"
            f"Estimated tax: {scenario.estimated_tax:.2f}\n"
            f"Estimated risk reduction (fraction): {scenario.risk_reduction_pct:.3f}\n\n"
            "Write a short explanation like you would say in a meeting:\n"
            "- Start by summarizing the issue (drift).\n"
            "- Describe in plain language what the trades do.\n"
            "- Mention risk change and tax impact.\n"
            "- End with why this fits the client's risk profile.\n"
        )