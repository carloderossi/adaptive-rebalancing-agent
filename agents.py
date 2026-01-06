from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from models import Portfolio, ClientProfile, Scenario


class MonitorAgent:
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
    ) -> Dict:
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
    def __init__(self, long_term_tax_rate: float = 0.15, short_term_tax_rate: float = 0.25):
        self.long_term_tax_rate = long_term_tax_rate
        self.short_term_tax_rate = short_term_tax_rate

    def estimate_tax_for_trades(self, portfolio_df: pd.DataFrame, trades: List[Dict]) -> float:
        """
        Very simplified tax model:
        - Assume all gains are long-term.
        - Tax = max(0, (price - cost_basis)) * quantity_sold * long_term_tax_rate
        """
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
            tax += taxable_gain * self.long_term_tax_rate
        return tax

    def assess_market_context(self) -> Dict:
        """
        Placeholder for market volatility, sector trends etc.
        For MVP, return a dummy structure.
        """
        return {
            "volatility_level": "normal",
            "comment": "Market volatility within normal range; no special timing adjustments applied."
        }

    def analyze(
        self,
        portfolio: Portfolio,
        client_profile: ClientProfile
    ) -> Dict:
        portfolio_df = portfolio.to_dataframe()
        market_context = self.assess_market_context()
        # In a more advanced version, we compute beta, volatility, sector concentration etc.
        risk_bounds = self._derive_risk_bounds(client_profile)
        return {
            "portfolio_df": portfolio_df,
            "market_context": market_context,
            "risk_bounds": risk_bounds,
        }

    def _derive_risk_bounds(self, client_profile: ClientProfile) -> Dict:
        profile = client_profile.risk_profile
        if profile == "conservative":
            max_equity = 0.5
        elif profile == "moderate":
            max_equity = 0.7
        else:
            max_equity = 0.9
        return {"max_equity_weight": max_equity}


class StrategistAgent:
    def __init__(self, analyst_agent: AnalystAgent):
        self.analyst_agent = analyst_agent

    def generate_scenarios(
        self,
        portfolio: Portfolio,
        client_profile: ClientProfile,
        monitor_output: Dict,
        analyst_output: Dict,
    ) -> List[Scenario]:
        portfolio_df = analyst_output["portfolio_df"]
        drift_by_class = monitor_output["drift_by_class"]

        # We build three simple scenarios:
        # A) Light: correct 30% of drift
        # B) Moderate: correct 60% of drift
        # C) Aggressive: correct 100% of drift
        correction_levels = {
            "A - Conservative": 0.3,
            "B - Moderate": 0.6,
            "C - Aggressive": 1.0,
        }

        scenarios = []
        for name, level in correction_levels.items():
            target_alloc = self._build_target_allocation(monitor_output, level)
            trades = self._build_trades_for_allocation(portfolio_df, target_alloc)
            estimated_tax = self.analyst_agent.estimate_tax_for_trades(portfolio_df, trades)
            risk_reduction_pct = self._estimate_risk_reduction(level, drift_by_class)
            scenarios.append(Scenario(
                name=name,
                target_allocation=target_alloc,
                trades=trades,
                estimated_tax=estimated_tax,
                risk_reduction_pct=risk_reduction_pct,
            ))
        return scenarios

    def _build_target_allocation(self, monitor_output: Dict, level: float) -> Dict[str, float]:
        """
        Move current allocation toward original target by 'level' fraction.
        new = current - level * drift
        """
        current = monitor_output["current_allocation"]
        target = monitor_output["target_allocation"]
        classes = set(current.keys()) | set(target.keys())
        new_alloc = {}
        for c in classes:
            cur = current.get(c, 0.0)
            tgt = target.get(c, 0.0)
            drift = cur - tgt
            new_alloc[c] = cur - level * drift
        # Normalize to sum to 1.0
        total = sum(new_alloc.values())
        if total > 0:
            for c in new_alloc:
                new_alloc[c] = new_alloc[c] / total
        return new_alloc

    def _build_trades_for_allocation(
        self,
        portfolio_df: pd.DataFrame,
        target_alloc: Dict[str, float]
    ) -> List[Dict]:
        total_value = portfolio_df["market_value"].sum()
        trades = []

        # For simplicity, we assume:
        # - Each asset_class has many tickers; we rebalance proportionally inside each class.
        current_alloc = (
            portfolio_df.groupby("asset_class")["market_value"].sum() / total_value
        ).to_dict()

        # Desired value per asset class
        desired_values = {c: w * total_value for c, w in target_alloc.items()}
        current_values = {
            c: v * total_value for c, v in current_alloc.items()
        }

        # For each asset class, compute buy/sell at class level
        class_deltas = {
            c: desired_values.get(c, 0.0) - current_values.get(c, 0.0)
            for c in set(desired_values.keys()) | set(current_values.keys())
        }

        # Translate class-level deltas to ticker-level trades (pro-rata)
        for asset_class, delta_value in class_deltas.items():
            class_mask = portfolio_df["asset_class"] == asset_class
            class_df = portfolio_df[class_mask]
            if class_df.empty:
                continue
            class_total_value = class_df["market_value"].sum()
            if class_total_value == 0:
                continue

            # If delta_value < 0: we sell proportionally; if >0: we buy proportionally.
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

    def _estimate_risk_reduction(self, level: float, drift_by_class: Dict[str, float]) -> float:
        """
        Heuristic: more correction level -> more risk reduction if drift is equity-heavy.
        For MVP: risk reduction = level * 0.2 (up to 20%).
        """
        return level * 0.2  # e.g., 0.3 -> 6%, 1.0 -> 20%


class ExplainerAgent:
    def build_summary_for_scenario(
        self,
        portfolio: Portfolio,
        client_profile: ClientProfile,
        monitor_output: Dict,
        scenario: Scenario,
    ) -> str:
        drift = monitor_output["drift_by_class"]
        over_weights = {k: v for k, v in drift.items() if v > 0.01}  # >1% overweight
        under_weights = {k: v for k, v in drift.items() if v < -0.01}

        parts = []

        total_drift_pct = monitor_output["total_drift"] * 100
        parts.append(
            f"Your client's portfolio has drifted approximately {total_drift_pct:.1f}% from the target allocation."
        )

        if over_weights:
            ow_desc = ", ".join(
                f"{k} by {v*100:.1f}%" for k, v in over_weights.items()
            )
            parts.append(f"Overweight asset classes: {ow_desc}.")
        if under_weights:
            uw_desc = ", ".join(
                f"{k} by {abs(v)*100:.1f}%" for k, v in under_weights.items()
            )
            parts.append(f"Underweight asset classes: {uw_desc}.")

        sell_trades = [t for t in scenario.trades if t["action"] == "SELL"]
        buy_trades = [t for t in scenario.trades if t["action"] == "BUY"]
        sell_value = sum(t["value"] for t in sell_trades)
        buy_value = sum(t["value"] for t in buy_trades)

        parts.append(
            f"In scenario '{scenario.name}', the system recommends selling approximately ${sell_value:,.0f} "
            f"and buying approximately ${buy_value:,.0f} across the portfolio."
        )

        parts.append(
            f"Estimated tax impact is about ${scenario.estimated_tax:,.0f}, and the expected risk reduction "
            f"is roughly {scenario.risk_reduction_pct*100:.1f}%."
        )

        parts.append(
            f"This scenario is aligned with the client's '{client_profile.risk_profile}' risk profile."
        )

        return " ".join(parts)