from dataclasses import dataclass
from typing import Dict, List, Literal, Optional
import pandas as pd


RiskProfile = Literal["conservative", "moderate", "aggressive"]


@dataclass
class ClientProfile:
    client_id: str
    risk_profile: RiskProfile
    tax_bracket: float  # e.g. 0.25 for 25%
    target_allocation: Dict[str, float]  # e.g. {"equity": 0.6, "bonds": 0.3, "cash": 0.1}


@dataclass
class PortfolioPosition:
    ticker: str
    quantity: float
    price: float
    cost_basis: float
    asset_class: str  # "equity", "bonds", "cash", etc.


@dataclass
class Portfolio:
    client_id: str
    positions: List[PortfolioPosition]

    @property
    def value(self) -> float:
        return sum(p.quantity * p.price for p in self.positions)

    def to_dataframe(self) -> pd.DataFrame:
        data = [{
            "ticker": p.ticker,
            "quantity": p.quantity,
            "price": p.price,
            "cost_basis": p.cost_basis,
            "asset_class": p.asset_class,
            "market_value": p.quantity * p.price
        } for p in self.positions]
        df = pd.DataFrame(data)
        df["weight"] = df["market_value"] / df["market_value"].sum()
        return df


@dataclass
class Scenario:
    name: str
    target_allocation: Dict[str, float]          # by asset class
    trades: List[Dict[str, float]]               # list of {"ticker", "action", "quantity", "value"}
    estimated_tax: float
    risk_reduction_pct: float                    # e.g. 0.15 -> 15% reduction
    description: Optional[str] = None