import pandas as pd
from pandas import DataFrame
from typing import TypedDict, List
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

from main import getMyPosition as getPosition

# CONSTANTS #######################################################################################
RAW_PRICES_FILEPATH: str = "./prices.txt"
START_DAY:int = 0
END_DAY:int = 0
INSTRUMENT_POSITION_LIMIT: int = 10000
COMMISSION_RATE: float = 0.0010
NUMBER_OF_INSTRUMENTS: int = 50

# TYPE DECLARATIONS ###############################################################################
class InstrumentPriceEntry(TypedDict):
    day: int
    instrument: int
    price: float

class BacktesterResults(TypedDict):
    daily_pnl: ndarray
    daily_capital_utilisation: ndarray
    instrument_traded: ndarray

# BACKTESTER CLASS ################################################################################
class Backtester:
    def __init__(self, enable_commission: bool) -> None:
        self.enable_commission: bool = enable_commission

        # Load prices data
        self.raw_prices_df: DataFrame = pd.read_csv(RAW_PRICES_FILEPATH, sep=r"\s+", header=None)

        # Transpose the raw prices such that every index represents an instrument number and each
        # row is a list of prices
        self.price_history: ndarray = self.raw_prices_df.to_numpy().T


    def run(self, start_day: int, end_day: int) -> BacktesterResults:
        # Initialise current positions, cash and portfolio value
        current_positions: ndarray = np.zeros(NUMBER_OF_INSTRUMENTS)
        cash: float = 0
        portfolio_value: float = 0

        # Initialise list of daily PnL's and capital utilisation
        daily_pnl_list: List[float] = []
        daily_capital_utilisation_list: List[float] = []

        # Iterate through specified timeline
        for day in range(start_day, end_day + 1):
            # Get the prices so far
            prices_so_far: ndarray = self.price_history[:, start_day - 1:day]

            # Get desired positions from strategy
            new_positions: ndarray = getPosition(prices_so_far)

            # Get today's prices
            current_prices: ndarray = prices_so_far[:, -1]

            # Calculate position limits
            position_limits: ndarray = np.array([int(x) for x in INSTRUMENT_POSITION_LIMIT /
                current_prices])

            # Adjust specified positions considering the position limit
            adjusted_positions: ndarray = np.clip(new_positions, -position_limits, position_limits)

            # Calculate volume
            delta_positions: ndarray = adjusted_positions - current_positions
            volumes: ndarray = current_prices * np.abs(delta_positions)
            total_volume: float = np.sum(volumes)

            # Calculate capital utilisation
            capital_utilisation: float = total_volume / (INSTRUMENT_POSITION_LIMIT
                 * NUMBER_OF_INSTRUMENTS)
            daily_capital_utilisation_list.append(capital_utilisation)

            # If commission is enabled, calculate it
            commission: float = total_volume * COMMISSION_RATE if self.enable_commission else 0

            # Subtract money spent on new positions from cash
            cash -= current_prices.dot(delta_positions) + commission

            # Update current positions
            current_positions = np.array(adjusted_positions)

            # Get total value of all positions
            positions_value: float = current_positions.dot(current_prices)

            # Calculate today's PnL and append it to list
            profit_and_loss: float = cash + positions_value - portfolio_value
            daily_pnl_list.append(profit_and_loss)

            # Update portfolio value
            portfolio_value = cash + positions_value

        backtester_results: BacktesterResults = BacktesterResults()
        backtester_results["daily_pnl"] = np.array(daily_pnl_list)
        backtester_results["daily_capital_utilisation"] = np.array(daily_capital_utilisation_list)

        return backtester_results

    def show_dashboard(self, backtester_results: BacktesterResults, start_day: int,
       end_day: int) -> None:
        daily_pnl:ndarray = backtester_results["daily_pnl"]
        daily_capital_utilisation:ndarray = backtester_results["daily_capital_utilisation"]

        fig,axs = plt.subplots(2, 2, figsize=(18,8))
        # Show Stats
        axs[0][0].axis("off")

        stats_text: str = (
            f"Ran from day {start_day} to {end_day}\n"
            r"$\bf{Commission \ Turned \ On:}$" + f"{self.enable_commission}\n\n"
            r"$\bf{Backtester \ Stats}$" + "\n\n"
            f"Mean PnL: ${daily_pnl.mean():.2f}\n"
            f"Std Dev: ${daily_pnl.std():.2f}\n"
            f"Annualised Sharpe Ratio: {np.sqrt(250) * daily_pnl.mean() / daily_pnl.std():.2f}\n"
            f"Score: {daily_pnl.mean() - 0.1*daily_pnl.std():.2f}"
        )

        axs[0][0].text(0.05, 0.95, stats_text, fontsize=14, va="top", ha="left")

        days: ndarray = np.arange(start_day, end_day + 1)

        # Plot Cumulative PnL over timeline
        cumulative_pnl: ndarray = np.cumsum(daily_pnl)

        axs[0][1].set_title(f"Cumulative Profit and Loss from day {start_day} to {end_day}")
        axs[0][1].set_xlabel("Days")
        axs[0][1].set_ylabel("Total PnL ($)")
        axs[0][1].grid(True)
        axs[0][1].plot(days, cumulative_pnl, linestyle="-")

        # Plot PnL over timeline
        axs[1][0].set_title(f"Daily Profit and Loss (PnL) from day {start_day} to {end_day}")
        axs[1][0].set_xlabel("Days")
        axs[1][0].set_ylabel("PnL ($)")
        axs[1][0].grid(True)
        axs[1][0].plot(days, daily_pnl, linestyle="-")

        # Plot daily capital utilisation
        daily_capital_utilisation_pct: ndarray = daily_capital_utilisation * 100

        axs[1][1].set_title(f"Daily capital utilisation from day {start_day} to {end_day}")
        axs[1][1].set_xlabel("Days")
        axs[1][1].set_ylabel("Capital Utilisation %")
        axs[1][1].grid(True)
        axs[1][1].plot(days, daily_capital_utilisation_pct, linestyle="-")

        plt.tight_layout()
        plt.show()




# MAIN EXECUTION #################################################################################
def main() -> None:
    backtester: Backtester = Backtester(True)
    backtester_results: BacktesterResults = backtester.run(1, 500)
    backtester.show_dashboard(backtester_results, 1, 500)

main()














