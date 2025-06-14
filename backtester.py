import pandas as pd
from pandas import DataFrame
from typing import TypedDict, List, Dict, Any
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import importlib.util
import sys
import os
from importlib.machinery import ModuleSpec
from types import ModuleType


# CONSTANTS #######################################################################################
RAW_PRICES_FILEPATH: str = "./prices.txt"
START_DAY:int = 0
END_DAY:int = 0
INSTRUMENT_POSITION_LIMIT: int = 10000
COMMISSION_RATE: float = 0.0005
NUMBER_OF_INSTRUMENTS: int = 50

PLOT_COLORS: Dict[str, str] = {
    "pnl": "#2ca02c",
    "cum_pnl": "#1f77b4",
    "utilisation": "#ff7f0e"
}

default_strategy_filepath: str = "./main.py"
strategy_function_name: str = "getMyPosition"
strategy_file_not_found_message: str = "Strategy file not found"
could_not_load_spec_message: str = "Could not load spec for module from strategy file"
strategy_function_does_not_exist_message:str = ("getMyPosition function does not exist in strategy "
                                                "file")
strategy_function_not_callable_message: str = "getMyPosition function is not callable"

usage_error: str = """
    Usage: backtester.py [OPTIONS]
    
    OPTIONS:
    --path [filepath: string] supply a custom filepath to your .py file that holds your
        getMyPosition() function. If not specified, it will use the filepath "./main.py"
    --timeline [start_day: int] [end_day: int] supply a custom start day and end day to run the
        backtester in. start day >= 1 and end day <= 750. If not specified, backtester will run
        throughout days 1-750
    --disable-comms disable commission on trades
"""

# TYPE DECLARATIONS ###############################################################################
class InstrumentPriceEntry(TypedDict):
    day: int
    instrument: int
    price: float

class BacktesterResults(TypedDict):
    daily_pnl: ndarray
    daily_capital_utilisation: ndarray
    instrument_traded: ndarray

class Params(TypedDict):
    strategy_filepath: str
    start_day: int
    end_day: int
    enable_commission: bool

# HELPER FUNCTIONS ###############################################################################
def parse_command_line_args() -> Params:
    total_args: int = len(sys.argv)
    params:Params = Params()
    params["strategy_filepath"] = default_strategy_filepath
    params["start_day"] = 1
    params["end_day"] = 750
    params["enable_commission"] = True

    if total_args > 1:
        i: int = 1
        while (i < total_args):
            current_arg: str = sys.argv[i]

            if current_arg == "--path":
                if i + 1 >= total_args:
                    raise Exception(usage_error)
                else:
                    i += 1
                    params["strategy_filepath"] = sys.argv[i]
            elif current_arg == "--timeline":
                if i + 2 >= total_args:
                    raise Exception(usage_error)
                else:
                    params["start_day"] = int(sys.argv[i + 1])
                    params["end_day"] = int(sys.argv[i + 2])
                    i += 2

                    if (params["start_day"] > params["end_day"] or params["start_day"] < 1 or
                            params["end_day"] > 750):
                        raise Exception(usage_error)
            elif current_arg == "--disable-comms":
                params["enable_commission"] = False
            else:
                raise Exception(usage_error)

            i += 1

    return params

def load_get_positions_function(strategy_filepath: str) -> Any:
    # Make sure file path is absolute and normalised
    filepath: str = os.path.abspath(strategy_filepath)

    # Check if file exists
    if not os.path.isfile(filepath):
        raise FileNotFoundError(strategy_file_not_found_message)

    # Get module name
    module_name: str = os.path.splitext(os.path.basename(filepath))[0]

    # Load the module spec
    spec: ModuleSpec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None:
        raise ImportError(could_not_load_spec_message)

    # Create a new module based on the spec
    module: ModuleType = importlib.util.module_from_spec(spec)

    # Create a new module based on the spec
    sys.modules[module_name] = module

    # Execute the module
    spec.loader.exec_module(module)

    # Get the strategy function from module
    if not hasattr(module, strategy_function_name):
        raise AttributeError(strategy_function_does_not_exist_message)
    function = getattr(module, strategy_function_name)

    # Verify that it's callable
    if not callable(function):
        raise TypeError(strategy_function_not_callable_message)

    return function

# BACKTESTER CLASS ################################################################################
class Backtester:
    def __init__(self, params: Params) -> None:
        self.enable_commission: bool = params["enable_commission"]
        self.getMyPosition = load_get_positions_function(params["strategy_filepath"])

        # Load prices data
        self.raw_prices_df: DataFrame = pd.read_csv(RAW_PRICES_FILEPATH, sep=r"\s+", header=None)

        # Transpose the raw prices such that every index represents an instrument number and each
        # row is a list of prices
        self.price_history: ndarray = self.raw_prices_df.to_numpy().T

    def run(self, start_day: int, end_day: int) -> BacktesterResults:
        """
        Run the backtest through specified timeline and keep track of daily PnL and capital usage
        :param start_day: day that the backtester should start running on
        :param end_day: day that the backtester should end running on (inclusive)
        :return: a BacktesterResults() class that contains daily PnL data and capital usage per day
        """
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
            new_positions: ndarray = self.getMyPosition(prices_so_far)

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
        """
        Generates and shows a dashboard that summarises a backtest's results. Shows stats such
        as mean PnL and sharpe ratio and plots cumulative PnL, Daily PnL and capital utilisation
        :param backtester_results: contains data on a backtest
        :param start_day: start day of the backtest
        :param end_day: end day of the backtest
        :return: None
        """
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

        axs[0][0].text(0.05, 0.95, stats_text, fontsize=14, va="top", ha="left", linespacing=1.5)

        days: ndarray = np.arange(start_day, end_day + 1)

        # Plot Cumulative PnL over timeline
        cumulative_pnl: ndarray = np.cumsum(daily_pnl)

        axs[0][1].set_title(f"Cumulative Profit and Loss from day {start_day} to {end_day}",
                            fontsize=12, fontweight="bold")
        axs[0][1].set_xlabel("Days", fontsize=10)
        axs[0][1].set_ylabel("Total PnL ($)", fontsize=10)
        axs[0][1].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        axs[0][1].spines["top"].set_visible(False)
        axs[0][1].spines["right"].set_visible(False)
        axs[0][1].plot(days, cumulative_pnl, linestyle="-", color=PLOT_COLORS["cum_pnl"],
                       linewidth=2)

        # Plot PnL over timeline
        axs[1][0].set_title(f"Daily Profit and Loss (PnL) from day {start_day} to {end_day}",
                            fontsize=12, fontweight="bold")
        axs[1][0].set_xlabel("Days", fontsize=10)
        axs[1][0].set_ylabel("PnL ($)", fontsize=10)
        axs[1][0].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        axs[1][0].spines["top"].set_visible(False)
        axs[1][0].spines["right"].set_visible(False)
        axs[1][0].plot(days, daily_pnl, linestyle="-", color=PLOT_COLORS["pnl"])

        # Plot daily capital utilisation
        daily_capital_utilisation_pct: ndarray = daily_capital_utilisation * 100

        axs[1][1].set_title(f"Daily capital utilisation from day {start_day} to {end_day}",
                            fontsize=12, fontweight="bold")
        axs[1][1].set_xlabel("Days", fontsize=10)
        axs[1][1].set_ylabel("Capital Utilisation %", fontsize=10)
        axs[1][1].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        axs[1][1].spines["top"].set_visible(False)
        axs[1][1].spines["right"].set_visible(False)
        axs[1][1].set_ylim(0, 100)
        axs[1][1].plot(days, daily_capital_utilisation_pct, linestyle="-", color=PLOT_COLORS[
            "utilisation"])

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.suptitle("Backtest Performance Summary", fontsize=16, fontweight="bold")
        plt.show()

# MAIN EXECUTION #################################################################################
def main() -> None:
    params: Params = parse_command_line_args()
    backtester: Backtester = Backtester(params)
    backtester_results: BacktesterResults = backtester.run(params["start_day"],
                                                           params["end_day"])
    backtester.show_dashboard(backtester_results, params["start_day"], params["end_day"])

main()