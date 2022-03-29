import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from util.util import *
from typing import Tuple
from sklearn.model_selection import train_test_split
from copy import deepcopy

from .DT_utils import (
    DataPaths,
    HeatDemand,
    DataLen,
    ProcessContinuousDataParams,
    ProcessDiscreteDataParams,
    TrainTestParams,
)


class Data:
    def __init__(
        self,
        delta_electricity,
        delta_heat,
        train_ratio,
        test_ratio,
    ):
        self.data_paths: DataPaths = DataPaths
        self.continuous_params: ProcessContinuousDataParams = (
            ProcessContinuousDataParams
        )
        self.discrete_params: ProcessDiscreteDataParams = ProcessDiscreteDataParams(
            delta_electricity, delta_heat
        )
        self.heat_demand_params: HeatDemand = HeatDemand
        self.train_test_params: TrainTestParams = TrainTestParams(
            train_ratio, test_ratio
        )
        self.data_len: int = DataLen.data_len
        self.daily_data_len: int = DataLen.daily_data_len
        self.heat_demand: list = self.rescale_heat_demand()
        self.continuous_data: list = self.create_dataset()
        self.discrete_data: list = self.discretize_data()
        self.discrete_day_data: list = self.create_daily_dataset()
        self.season: Tuple[list, list] = self.create_train_test_dataset(index=0)
        self.time_of_the_day: Tuple[list, list] = self.create_train_test_dataset(
            index=1
        )
        self.electricity_price: Tuple[list, list] = self.create_train_test_dataset(
            index=2
        )
        self._heat_demand: Tuple[list, list] = self.create_train_test_dataset(index=3)
        self.eliminate_test_overlap()

    def rescale_heat_demand(self) -> list:
        heat_data = []
        for i in range(self.data_len):
            heat_data.append(
                round(
                    self.continuous_params.heat_demand_upper_bound
                    * (
                        self.heat_demand_params.heat_demand[i]
                        - self.heat_demand_params.min_heat_demand
                    )
                    / (
                        self.heat_demand_params.max_heat_demand
                        - self.heat_demand_params.min_heat_demand
                    ),
                    2,
                )
            )
        return heat_data

    def remove_special_characters(self, price) -> float:
        alphanumeric = ""
        for character in price:
            if character.isalnum() or character == "." or character == "-":
                alphanumeric += character
        return float(alphanumeric)

    def create_dataset(self) -> list:
        num = 0
        data = []
        with os.scandir(self.data_paths.ELECTRICITY_PRICE) as it:
            for entry in it:
                day_ahead_electricity_prices = np.array(
                    pd.DataFrame(pd.read_csv(entry.path, delimiter=";"))
                )
                mul = (
                    0
                    if num == 0
                    else (
                        self.continuous_params.multi_factor[0]
                        if num == 1
                        else self.continuous_params.multi_factor[0]
                        + (num - 1) * self.continuous_params.multi_factor[1]
                    )
                )
                for i in range(len(day_ahead_electricity_prices)):
                    day_ahead_electricity_price = str(day_ahead_electricity_prices[i])
                    month = day_ahead_electricity_price[
                        self.continuous_params.month_start_index : self.continuous_params.month_end_index
                    ]
                    hour = day_ahead_electricity_price[
                        self.continuous_params.day_time_start_index : self.continuous_params.day_time_end_index
                    ]
                    price = day_ahead_electricity_price[
                        self.continuous_params.price_start_index : self.continuous_params.price_end_index
                    ]
                    price = self.remove_special_characters(price)
                    season = self.discrete_params.season_dict.get(month)
                    day_time = self.discrete_params.day_time_dict.get(hour)
                    heat_demand_data = self.heat_demand[mul + i]
                    temp = [season, day_time, price, heat_demand_data]
                    data.append(temp)
                num += 1
        return data

    def discretize_element(self, x, dict) -> int:
        for key, value in dict.items():
            if x >= value[0] and x < value[1]:
                return key

    def discretize_data(self) -> list:
        discrete_data = deepcopy(self.continuous_data)
        for i in range(self.data_len):
            discrete_data[i][2] = self.discretize_element(
                self.continuous_data[i][2],
                self.discrete_params.electricity_price_discrete,
            )
            discrete_data[i][3] = self.discretize_element(
                self.continuous_data[i][3], self.discrete_params.heat_demand_discrete
            )
        return discrete_data

    def create_daily_dataset(self) -> list:
        dataset = []
        for i in range(self.daily_data_len):
            dataset.append(
                self.discrete_data[i * TIME_HORIZON : (i + 1) * TIME_HORIZON]
            )
        random.shuffle(dataset)
        return dataset

    def create_train_test_dataset(self, index) -> Tuple[list, list]:
        x = [
            [self.discrete_day_data[i][j][index] for j in range(TIME_HORIZON)]
            for i in range(self.daily_data_len)
        ]
        x_train, x_test = train_test_split(
            x, train_size=self.train_test_params.TRAIN_SIZE, shuffle=False
        )
        return x_train, x_test

    def eliminate_test_overlap(self):
        temp_heat_electricity = []
        for i in range(self.train_test_params.TEST_SIZE):
            temp_heat_electricity.append(
                self.electricity_price[1][i] + self._heat_demand[1][i]
            )
        temp_heat_electricity_new = []
        for i, e in reversed(list(enumerate(temp_heat_electricity))):
            if e not in temp_heat_electricity_new:
                temp_heat_electricity_new.append(e)
            else:
                self.season[1].pop(i)
                self.time_of_the_day[1].pop(i)
                self.electricity_price[1].pop(i)
                self._heat_demand[1].pop(i)

    def save_continuous_data(self) -> None:
        df = pd.DataFrame(self.continuous_data, columns=self.continuous_params.columns)
        df.to_csv(self.data_paths.PROCESSED_DATA, index=False)

    def save_discrete_data(self) -> None:
        df = pd.DataFrame(self.discrete_data, columns=self.continuous_params.columns)
        df.to_csv(self.data_paths.PROCESSED_DATA_DISCRETE, index=False)

    def save_train_test_data(self) -> None:
        for i in range(len(TRAIN_TEST_COL)):
            pd.DataFrame(self.season[i]).to_csv(
                os.path.join(self.data_paths.DATA, "season" + TRAIN_TEST_COL[i]),
                index=False,
            )
            pd.DataFrame(self.time_of_the_day[i]).to_csv(
                os.path.join(
                    self.data_paths.DATA, "time_of_the_day" + TRAIN_TEST_COL[i]
                ),
                index=False,
            )
            pd.DataFrame(self.electricity_price[i]).to_csv(
                os.path.join(
                    self.data_paths.DATA,
                    "day_ahead_electricity_price" + TRAIN_TEST_COL[i],
                ),
                index=False,
            )
            pd.DataFrame(self._heat_demand[i]).to_csv(
                os.path.join(self.data_paths.DATA, "heat_demand" + TRAIN_TEST_COL[i]),
                index=False,
            )

    def plot(self) -> None:
        fig, ax = plt.subplots()
        title = "Scaled heat demand density"
        ax.set_xlabel("Scaled heat demand [MWh]")
        ax.set_ylabel("Density")
        ax.set_title(title)
        sns.kdeplot(list(list(zip(*self.continuous_data))[3]), shade=True)
        plt.show()

        fig, ax = plt.subplots()
        title = "Day-ahead electricity price density"
        ax.set_xlabel("Day-ahead electricity price [e/MWh]")
        ax.set_ylabel("Density")
        ax.set_title(title)
        sns.kdeplot(list(list(zip(*self.continuous_data))[2]), shade=True)
        plt.show()

        fig, ax = plt.subplots()
        title = "Heat demand histogram"
        ax.set_label("Heat demand")
        ax.set_ylabel("Number of samples")
        ax.set_title(title)
        plt.hist(list(list(zip(*self.discrete_data))[3]))
        plt.show()

        fig, ax = plt.subplots()
        title = "Electricity price histogram"
        ax.set_label("Electricity price")
        ax.set_ylabel("Number of samples")
        ax.set_title(title)
        plt.hist(list(list(zip(*self.discrete_data))[2]))
        plt.show()


if __name__ == "__main__":
    TRAIN_TEST_COL = ["_train.csv", "_test.csv"]
    data = Data(9, 9, 0.9, 0.1)
    df = pd.DataFrame(data.continuous_data, columns=ProcessContinuousDataParams.columns)
    df_discrete = pd.DataFrame(
        data.discrete_data, columns=ProcessContinuousDataParams.columns
    )
    data.plot()
