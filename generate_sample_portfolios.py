import os
from sample_portfolios import (
    generate_random_portfolio,
    generate_conservative_portfolio,
    generate_balanced_portfolio,
    generate_aggressive_portfolio,
    save_portfolio_to_csv,
)

OUTPUT_DIR = "sample_portfolios"

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def main():
    ensure_output_dir()

    # Generate fixed-style portfolios
    save_portfolio_to_csv(
        generate_conservative_portfolio(),
        os.path.join(OUTPUT_DIR, "conservative.csv")
    )

    save_portfolio_to_csv(
        generate_balanced_portfolio(),
        os.path.join(OUTPUT_DIR, "balanced.csv")
    )

    save_portfolio_to_csv(
        generate_aggressive_portfolio(),
        os.path.join(OUTPUT_DIR, "aggressive.csv")
    )

    # Generate several random portfolios
    for i in range(1, 6):
        save_portfolio_to_csv(
            generate_random_portfolio(client_id=f"random-{i}"),
            os.path.join(OUTPUT_DIR, f"random_{i}.csv")
        )

    print("Sample portfolios generated in:", OUTPUT_DIR)

# runit via
# uv run python generate_sample_portfolios.py
if __name__ == "__main__":
    main()