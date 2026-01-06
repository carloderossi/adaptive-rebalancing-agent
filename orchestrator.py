from typing import Dict, List

from models import Portfolio, ClientProfile, Scenario
from agents import MonitorAgent, AnalystAgent, StrategistAgent, ExplainerAgent


class Orchestrator:
    def __init__(self, drift_threshold: float = 0.05):
        self.monitor_agent = MonitorAgent(drift_threshold=drift_threshold)
        self.analyst_agent = AnalystAgent()
        self.strategist_agent = StrategistAgent(self.analyst_agent)
        self.explainer_agent = ExplainerAgent()

    def run(self, portfolio: Portfolio, client_profile: ClientProfile) -> Dict:
        # Step 1: Monitor
        monitor_output = self.monitor_agent.analyze_drift(portfolio, client_profile)

        if not monitor_output["needs_rebalancing"]:
            summary = (
                f"Portfolio drift ({monitor_output['total_drift']*100:.1f}%) is below the "
                f"{self.monitor_agent.drift_threshold*100:.1f}% threshold. "
                f"No rebalancing is recommended at this time."
            )
            return {
                "needs_rebalancing": False,
                "monitor_output": monitor_output,
                "scenarios": [],
                "summaries": [],
                "no_action_summary": summary,
            }

        # Step 2: Analyst
        analyst_output = self.analyst_agent.analyze(portfolio, client_profile)

        # Step 3: Strategist
        scenarios: List[Scenario] = self.strategist_agent.generate_scenarios(
            portfolio=portfolio,
            client_profile=client_profile,
            monitor_output=monitor_output,
            analyst_output=analyst_output,
        )

        # Step 4: Explainer
        summaries = [
            self.explainer_agent.build_summary_for_scenario(
                portfolio, client_profile, monitor_output, scenario
            )
            for scenario in scenarios
        ]

        return {
            "needs_rebalancing": True,
            "monitor_output": monitor_output,
            "scenarios": scenarios,
            "summaries": summaries,
        }