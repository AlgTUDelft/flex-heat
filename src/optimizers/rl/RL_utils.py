from dataclasses import dataclass
from pathlib import *
import pandas as pd
from time import time
from util.util import *


@dataclass
class DataPaths:
    """
    Path to data folder.
    """

    data: Path = Path(__file__).parents[3] / "data"


@dataclass
class Data:
    """
    Paths to training and testing datasets of Q-learning algorithm.
    """

    data = pd.read_csv(
        DataPaths.data / "processed_data_discrete.csv",
        usecols=["Season", "Time_day", "Price", "Heat_demand"],
    ).values.tolist()
    data_len: int = len(data)
    heat_demand_data, electricity_price_data = [], []
    for i in range(data_len):
        heat_demand_data.append(data[i][3])
        electricity_price_data.append(data[i][2])
    heat_demand_data_len: int = len(heat_demand_data)
    electricity_price_data_len: int = len(electricity_price_data)
    max_heat_demand: float = max(heat_demand_data)
    min_heat_demand: float = min(heat_demand_data)
    max_electricity_price: float = max(electricity_price_data)
    min_electricity_price: float = min(electricity_price_data)
    season_train = pd.read_csv(DataPaths.data / "season_train.csv").values.tolist()
    time_of_the_day_train = pd.read_csv(
        DataPaths.data / "time_of_the_day_train.csv"
    ).values.tolist()
    heat_demand_train = pd.read_csv(
        DataPaths.data / "heat_demand_train.csv"
    ).values.tolist()
    electricity_price_train = pd.read_csv(
        DataPaths.data / "day_ahead_electricity_price_train.csv"
    ).values.tolist()

    season_test = pd.read_csv(DataPaths.data / "season_test.csv").values.tolist()
    time_of_the_day_test = pd.read_csv(
        DataPaths.data / "time_of_the_day_test.csv"
    ).values.tolist()
    heat_demand_test = pd.read_csv(
        DataPaths.data / "heat_demand_test.csv"
    ).values.tolist()
    electricity_price_test = pd.read_csv(
        DataPaths.data / "day_ahead_electricity_price_test.csv"
    ).values.tolist()
    train_len: int = len(season_train)
    test_len: int = len(season_test)


@dataclass
class RewardFunParams:
    """
    Reward functions parameters.
    """

    operation_cost_reshape_up: int = 20
    operation_cost_reshape_down: int = -10
    operation_cost_gradient: float = 0.2
    underdeliver_heat_reshape_up: int = 0
    underdeliver_heat_reshape_down: int = -40
    underdeliver_heat_gradient: float = 0.4
    overdeliver_heat_reshape_up: int = 0
    overdeliver_heat_reshape_down: int = -5
    overdeliver_heat_gradient: float = 0.4
    constraint_violation_up: int = TIME_HORIZON * (
        underdeliver_heat_reshape_down + operation_cost_reshape_down
    )
    constraint_violation_down: int = (
        2
        * TIME_HORIZON
        * (underdeliver_heat_reshape_down + operation_cost_reshape_down)
    )
    constraint_violation_gradient: float = 0.1


@dataclass
class RLParams:
    """
    Parameters of the reinforcement learning algorithm.
    """

    # future demands taken into account for state creation
    future_heat_demand: int = 1
    future_electricity_price: int = 1
    season_combinations = [0, 1, 2, 3]
    time_of_the_day_combinations = [0, 1, 2, 3]
    action_space = [
        (5, 48.92857),
        (10, 47.85714),
        (15, 46.7857),
        (20, 45.7142857),
        (25, 44.642857),
        (30, 43.5714),
        (40, 41.42857),
        (50, 39.2857),
        (70, 35),
        (50, 25),
        (40, 20),
        (30, 15),
        (25, 12.5),
        (20, 10),
        (15, 7.5),
        (10, 5),
        (5, 7.5),
    ]


@dataclass
class EnvParams:
    """
    Parameters of the environment for training Q-learning algorithm.
    """

    min_return_T: int = 45  # minimal return temperature for reward creation
    number_condition_flags: int = 4
    base_return_T: int = 48
    heat_demand_tolerance: float = 0.5  # [MW/h]


def pipe_disc(x, base) -> float:
    """
    Discretization of the element according to the base.
    """
    return round(base * round(x / base))


def to_tuple(lst) -> tuple:
    """
    Transformation of list of lists into tuples.
    """
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)


def operation_cost_upper_bound() -> float:
    """
    Extreme CHP points define reward function rescaling. As they are temporary unavailable due to
    (0,10) and (0,50) immediate lead to error states, rescaling is done based on the available action points.
    These action points preserve the order of the CHP extreme points in the reward shaping.
    """
    if Data.max_electricity_price > a[1]:
        operation_cost_up = (
            -a[0] * EXTREME_POINTS[3][0]
            + (Data.max_electricity_price - a[1]) * EXTREME_POINTS[3][1]
        )
    elif -2 * a[0] + a[1] < Data.max_electricity_price <= a[1]:
        operation_cost_up = (
            -a[0] * EXTREME_POINTS[0][0]
            + (Data.max_electricity_price - a[1]) * EXTREME_POINTS[0][1]
        )
    else:
        operation_cost_up = (
            -a[0] * EXTREME_POINTS[1][0]
            + (Data.max_electricity_price - a[1]) * EXTREME_POINTS[1][1]
        )
    return operation_cost_up


def operation_cost_lower_bound() -> float:
    """
    Determining lower bound on the operation cost based on the placement of
    minimal electricity price.
    """
    if Data.min_electricity_price < 0:
        operation_cost_down = (
            -a[0] * EXTREME_POINTS[3][0]
            + (Data.min_electricity_price - a[1]) * EXTREME_POINTS[3][1]
        )
    elif 0 <= Data.min_electricity_price < 2 * a[0] + a[1]:
        operation_cost_down = (
            -a[0] * EXTREME_POINTS[2][0]
            + (Data.min_electricity_price - a[1]) * EXTREME_POINTS[2][1]
        )
    else:
        operation_cost_down = (
            -a[0] * EXTREME_POINTS[1][0]
            + (Data.min_electricity_price - a[1]) * EXTREME_POINTS[1][1]
        )
    return operation_cost_down


def heat_delivered_upper_bound(max_flow_speed, pipe_diameter, max_supply_temp) -> float:
    """
    Upper bound on delivered heat (maximum possible amount of heat that can be delivered).
    """
    return (
        C
        * max_flow_speed
        * WATER_DENSITY
        * pow(pipe_diameter / 2, 2)
        * PI
        * (max_supply_temp - EnvParams.min_return_T)
    )


class Timer:
    """Simple timer"""

    def __init__(self):
        """Creates a new, uninitialized timer"""
        self.logged_time = 0
        self.reset()

    def get_time_minutes(self):
        """Return elapsed time in seconds"""
        curr_time = time()
        elapsed = curr_time - self.logged_time
        self.logged_time = curr_time
        return elapsed / 60

    def reset(self):
        """Reset the timer"""
        self.logged_time = time()
