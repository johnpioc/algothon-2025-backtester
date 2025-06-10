import pandas as pd
from pandas import DataFrame
from typing import TypedDict, List, Dict
import numpy as np
from numpy import ndarray

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

# BACKTESTER CLASS ################################################################################
class Backtester:
    def __init__(self, enable_commission: bool) -> None:
        self.enable_commission: bool = enable_commission

        # Load prices data
        self.raw_prices_df: DataFrame = pd.read_csv(RAW_PRICES_FILEPATH, sep=r"\s+", header=None)

        # Transpose the raw prices such that every index represents an instrument number and each
        # row is a list of prices
        self.price_history: ndarray = self.raw_prices_df.to_numpy().T

    def get_market_data(self, instruments: List[int], start_day: int, end_day: int) -> DataFrame:
        # initialise a list of price entries
        price_entries: List[InstrumentPriceEntry] = []

        # Iterate through specified timeline
        for day in range(start_day, end_day + 1):
            # iterate through specified list of instruments
            for instrument_no in instruments:
                new_entry: InstrumentPriceEntry = InstrumentPriceEntry()

                new_entry["day"] = day
                new_entry["instrument"] = instrument_no
                new_entry["price"] = self.raw_prices_df.iloc[day - 1, instrument_no]

                price_entries.append(new_entry)

        market_data: DataFrame = pd.DataFrame(price_entries)
        return market_data


    def run(self, start_day: int, end_day: int) -> None:
        # Initialise current positions, cash and portfolio value
        current_positions: ndarray = np.zeros(NUMBER_OF_INSTRUMENTS)
        cash: float = 0
        portfolio_value: float = 0

        # Initialise list of daily PnL's
        daily_pnl_list: List[float] = []

        # Iterate through specified timeline
        for day in range(start_day + 1, end_day + 1):
            # Get the prices so far
            prices_so_far: ndarray = self.price_history[:, :day]

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

        daily_pnl: ndarray = np.array(daily_pnl_list)
        average_pnl = np.mean(daily_pnl)
        standard_dev = np.std(daily_pnl)

        print(f"Mean Profit and Loss per day: {average_pnl:.2f}")
        print(f"StdDev (PnL): {standard_dev:.2f}")

# MAIN EXECUTION #################################################################################
def main() -> None:
    backtester: Backtester = Backtester(True)
    backtester.run(0, 100)

main()














