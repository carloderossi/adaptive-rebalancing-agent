import streamlit as st
import pandas as pd

from models import Portfolio, PortfolioPosition, ClientProfile
from orchestrator_langgraph import OrchestratorLangGraph


def parse_portfolio_csv(uploaded_file) -> Portfolio:
    df = pd.read_csv(uploaded_file)

    required_cols = {"ticker", "quantity", "price", "cost_basis", "asset_class"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    positions = []
    for _, row in df.iterrows():
        positions.append(
            PortfolioPosition(
                ticker=row["ticker"],
                quantity=float(row["quantity"]),
                price=float(row["price"]),
                cost_basis=float(row["cost_basis"]),
                asset_class=str(row["asset_class"]),
            )
        )

    return Portfolio(client_id="demo-client", positions=positions)


def default_target_allocation(risk_profile: str):
    if risk_profile == "conservative":
        return {"equity": 0.4, "bonds": 0.5, "cash": 0.1}
    elif risk_profile == "moderate":
        return {"equity": 0.6, "bonds": 0.3, "cash": 0.1}
    else:
        return {"equity": 0.8, "bonds": 0.15, "cash": 0.05}


def main():
    st.title("Adaptive Portfolio Rebalancing Agent (Ollama + LangGraph)")

    st.sidebar.header("Client configuration")

    risk_profile = st.sidebar.selectbox(
        "Risk profile",
        options=["conservative", "moderate", "aggressive"],
        index=1,
    )

    tax_bracket = st.sidebar.slider(
        "Tax bracket (%)",
        min_value=0,
        max_value=50,
        value=25,
        step=1,
    ) / 100.0

    st.sidebar.markdown("### Target allocation (by asset class)")
    default_alloc = default_target_allocation(risk_profile)
    equity_weight = st.sidebar.slider(
        "Equity weight", 0.0, 1.0, default_alloc["equity"], 0.05
    )
    bonds_weight = st.sidebar.slider(
        "Bonds weight", 0.0, 1.0 - equity_weight, default_alloc["bonds"], 0.05
    )
    cash_weight = 1.0 - equity_weight - bonds_weight

    st.sidebar.write(f"Cash weight (auto): {cash_weight:.2f}")

    target_alloc = {
        "equity": equity_weight,
        "bonds": bonds_weight,
        "cash": cash_weight,
    }

    uploaded_file = st.file_uploader(
        "Upload portfolio CSV",
        type=["csv"],
        help="Columns required: ticker, quantity, price, cost_basis, asset_class",
    )

    if uploaded_file is None:
        st.info("Upload a portfolio CSV to begin.")
        return

    try:
        portfolio = parse_portfolio_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error parsing portfolio: {e}")
        return

    client_profile = ClientProfile(
        client_id="demo-client",
        risk_profile=risk_profile,  # type: ignore
        tax_bracket=tax_bracket,
        target_allocation=target_alloc,
    )

    if st.button("Run rebalancing analysis"):
        orchestrator = OrchestratorLangGraph(drift_threshold=0.05)
        result = orchestrator.run(portfolio, client_profile)

        st.subheader("Current portfolio")

        df = portfolio.to_dataframe()
        st.dataframe(df)

        current_alloc = result["monitor_output"]["current_allocation"]
        st.write("Current allocation by asset class:")
        st.bar_chart(pd.Series(current_alloc))

        st.write("Target allocation by asset class:")
        st.bar_chart(pd.Series(client_profile.target_allocation))

        if not result.get("needs_rebalancing"):
            st.success(result["no_action_summary"])
            return

        st.subheader("Rebalancing scenarios")
        scenarios = result["scenarios"]
        summaries = result["summaries"]

        for scenario, summary in zip(scenarios, summaries):
            with st.expander(f"Scenario: {scenario.name}", expanded=False):
                st.write(summary)

                trades_df = pd.DataFrame(scenario.trades)
                if not trades_df.empty:
                    st.write("Trades:")
                    st.dataframe(trades_df)

                st.write(
                    f"Estimated tax impact: ${scenario.estimated_tax:,.0f} | "
                    f"Estimated risk reduction: {scenario.risk_reduction_pct*100:.1f}%"
                )

                col1, col2, col3 = st.columns(3)
                if col1.button("Approve", key=f"approve_{scenario.name}"):
                    st.success(f"Scenario '{scenario.name}' approved (logged).")
                if col2.button("Modify", key=f"modify_{scenario.name}"):
                    st.info("Modification flow not yet implemented in this version.")
                if col3.button("Reject", key=f"reject_{scenario.name}"):
                    st.warning(f"Scenario '{scenario.name}' rejected (logged).")


if __name__ == "__main__":
    main()