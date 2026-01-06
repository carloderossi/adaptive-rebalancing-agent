from __future__ import annotations

from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

from models import Portfolio, ClientProfile, Scenario
from agents import MonitorAgent, AnalystAgent, StrategistAgent, ExplainerAgent
from llm_client import LLMClient
from config_loader import Config


class AgentState(TypedDict, total=False):
    portfolio: Portfolio
    client_profile: ClientProfile

    needs_rebalancing: bool
    monitor_output: Dict[str, Any]
    analyst_output: Dict[str, Any]
    scenarios: List[Scenario]
    summaries: List[str]
    no_action_summary: str


class OrchestratorLangGraph:
    def __init__(self, drift_threshold: float = 0.05, config: Config | None = None):
        self.config = config or Config()
        self.llm_client = LLMClient(self.config)

        self.monitor_agent = MonitorAgent(drift_threshold=drift_threshold)
        self.analyst_agent = AnalystAgent(self.llm_client)
        self.strategist_agent = StrategistAgent(self.llm_client, self.analyst_agent)
        self.explainer_agent = ExplainerAgent(self.llm_client)

        self.graph = self._build_graph()

    # --- Node functions ---

    def _monitor_node(self, state: AgentState) -> AgentState:
        portfolio = state["portfolio"]
        client_profile = state["client_profile"]
        monitor_output = self.monitor_agent.analyze_drift(portfolio, client_profile)

        new_state: AgentState = dict(state)
        new_state["monitor_output"] = monitor_output
        new_state["needs_rebalancing"] = monitor_output["needs_rebalancing"]

        if not monitor_output["needs_rebalancing"]:
            new_state["no_action_summary"] = (
                f"Portfolio drift ({monitor_output['total_drift']*100:.1f}%) is below the "
                f"{self.monitor_agent.drift_threshold*100:.1f}% threshold. "
                "No rebalancing is recommended at this time."
            )
        return new_state

    def _analyst_node(self, state: AgentState) -> AgentState:
        portfolio = state["portfolio"]
        client_profile = state["client_profile"]
        monitor_output = state["monitor_output"]

        analyst_output = self.analyst_agent.analyze(
            portfolio=portfolio,
            client_profile=client_profile,
            monitor_output=monitor_output,
        )

        new_state: AgentState = dict(state)
        new_state["analyst_output"] = analyst_output
        return new_state

    def _strategist_node(self, state: AgentState) -> AgentState:
        portfolio = state["portfolio"]
        client_profile = state["client_profile"]
        monitor_output = state["monitor_output"]
        analyst_output = state["analyst_output"]

        scenarios = self.strategist_agent.generate_scenarios(
            portfolio=portfolio,
            client_profile=client_profile,
            monitor_output=monitor_output,
            analyst_output=analyst_output,
        )

        new_state: AgentState = dict(state)
        new_state["scenarios"] = scenarios
        return new_state

    def _explainer_node(self, state: AgentState) -> AgentState:
        portfolio = state["portfolio"]
        client_profile = state["client_profile"]
        monitor_output = state["monitor_output"]
        scenarios = state["scenarios"]

        summaries: List[str] = []
        for scenario in scenarios:
            summary = self.explainer_agent.build_summary_for_scenario(
                portfolio=portfolio,
                client_profile=client_profile,
                monitor_output=monitor_output,
                scenario=scenario,
            )
            summaries.append(summary)

        new_state: AgentState = dict(state)
        new_state["summaries"] = summaries
        return new_state

    # --- Conditional routing ---

    def _should_rebalance(self, state: AgentState) -> str:
        if state.get("needs_rebalancing"):
            return "rebalance"
        return "no_rebalance"

    # --- Graph definition ---

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("monitor", self._monitor_node)
        workflow.add_node("analyst", self._analyst_node)
        workflow.add_node("strategist", self._strategist_node)
        workflow.add_node("explainer", self._explainer_node)

        workflow.set_entry_point("monitor")

        workflow.add_conditional_edges(
            "monitor",
            self._should_rebalance,
            {
                "rebalance": "analyst",
                "no_rebalance": END,
            },
        )

        workflow.add_edge("analyst", "strategist")
        workflow.add_edge("strategist", "explainer")

        return workflow.compile()

    # --- Public API ---

    def run(self, portfolio: Portfolio, client_profile: ClientProfile) -> AgentState:
        initial_state: AgentState = {
            "portfolio": portfolio,
            "client_profile": client_profile,
        }
        final_state = self.graph.invoke(initial_state)
        return final_state