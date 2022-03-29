import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import re

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple
from util.util import *
from scipy.stats import ttest_rel
from pathlib import *


@dataclass
class DataPaths:
    """
    Paths to data files.
    """

    PARENT: str = Path(__file__).parents[1]
    CURRENT: str = PARENT / "plots"
    DATA: str = PARENT / "data"
    IMAGES: str = CURRENT / "images"
    RESULT: str = PARENT / "results"
    HEAT_DEMAND: str = DATA / "heat_demand_test.csv"
    ELECTRICITY_PRICE: str = DATA / "day_ahead_electricity_price_test.csv"
    BOUND: str = RESULT / "upper_bound_basic_strategy"
    LP: str = RESULT / "abdollahi_2015"
    NLMIP_DAY_AHEAD: str = RESULT / "li_2016"
    NLMIP_OPT: str = NLMIP_DAY_AHEAD / "li_2016_day_ahead"
    NLMIP_SIM_FLAG: str = NLMIP_DAY_AHEAD / "li_2016_day_ahead_sim"
    RL_FULL_STATE: str = RESULT / "rl_full_state/rl_full_state"
    RL_ABSTRACT_STATE: str = RESULT / "rl_abstract_state/rl_abstract_state"
    RL_FULL_STATE_SIM_FLAG: str = RESULT / "rl_full_state/rl_full_state_sim"


class Data(DataPaths):
    """
    Loads and processes the data.
    """

    def __init__(self, N, L, Q_init, number_of_episodes):
        self.N = N
        self.L = L
        self.Q_init = Q_init
        self.number_of_episodes = number_of_episodes
        self.__dir = self.IMAGES / "unstable_index"
        self.__heat_demand: list = pd.read_csv(self.HEAT_DEMAND).values.tolist()
        self.__electricity_price: list = pd.read_csv(
            self.ELECTRICITY_PRICE
        ).values.tolist()
        self.__heat_demand_var: list = [
            round(np.var(self.__heat_demand[i]), 1) for i in range(self.N)
        ]
        self.__electricity_price_var: list = [
            round(np.var(self.__electricity_price[i]), 1) for i in range(self.N)
        ]
        self.__covariance: list = [
            np.cov(self.__heat_demand[i], self.__electricity_price[i])[0][1]
            for i in range(self.N)
        ]
        self.__upper_bound_sum: list = Data.read_pickle(
            self.regex(self.BOUND, bound=True, up=True)
        )
        self.__basic_strategy_sum: list = Data.read_pickle(
            self.regex(self.BOUND, bound=True, up=False)
        )
        self.__lp_sum: list = Data.read_pickle(self.regex(self.LP, lp=True))
        self.__nlmip_sum: list = Data.read_pickle(self.regex(self.NLMIP_OPT))
        self.__rl_full_state: dict = self.read_rl_state(Data.RL_FULL_STATE)
        self.__condition_flags: dict = self.read_rl_infeasible_indices()
        self.nlmip_unstable_indices: list = Data.read_pickle(
            os.path.join(self.NLMIP_OPT, "nlmip_unstable_indices_L_{}".format(self.L))
        )
        self.q_function_full = self.find_q_function(self.RL_FULL_STATE)
        self.q_function_abstract = self.find_q_function(self.RL_ABSTRACT_STATE)
        self.reward_function_full: list = self.find_reward_function(self.RL_FULL_STATE)
        self.reward_function_abstract: list = self.find_reward_function(
            self.RL_ABSTRACT_STATE
        )
        self.__plot: Plots = Plots(
            self.N,
            self.L,
            self.Q_init,
            self.number_of_episodes,
            self.__dir,
            self.__heat_demand_var,
            self.__electricity_price_var,
            self.__covariance,
            self.__upper_bound_sum,
            self.__basic_strategy_sum,
            self.__lp_sum,
            self.__nlmip_sum,
            self.nlmip_unstable_indices,
            self.__rl_full_state,
            self.__condition_flags,
            self.q_function_full,
            self.q_function_abstract,
            self.reward_function_full,
            self.reward_function_abstract,
        )

    def regex(
        self,
        path,
        bound=False,
        up=False,
        lp=False,
        Q_init=-1,
        condition_flag=False,
        condition_flag_type="",
    ) -> Path:
        """
        Finds the path to the specific file.
        """
        rx = ("upper" if up else "basic_strategy") if bound else ".*?"
        if condition_flag:
            rx += condition_flag_type + ".*?"
        else:
            rx += ".*?" + "sum" + ".*?"
        if not lp:
            rx += "_L_" + str(self.L) + ".*?"
        if Q_init in self.Q_init:
            rx += "_Q_" + str(Q_init) + ".*?"
        for file in os.listdir(path):
            match = re.match(rx, file)
            if match:
                return os.path.join(path, file)

    def read_rl_state(self, path) -> dict:
        """
        Read reinforcement learning (full state space) results.
        """
        dict = {}
        for i in range(len(self.Q_init)):
            dict["Q_init_" + str(abs(self.Q_init[i]))] = Data.read_pickle(
                self.regex(path, Q_init=self.Q_init[i])
            )
        return dict

    def read_nlmip_infeasible_indices(self) -> dict:
        """
        Read MINLP infeasible indices.
        """
        condition_flags_ = {}
        for k, v in condition_flags.items():
            condition_flags_[k] = Data.read_pickle(
                self.regex(
                    self.NLMIP_SIM_FLAG, condition_flag=True, condition_flag_type=k
                )
            )
        return condition_flags_

    def read_rl_infeasible_indices(self) -> dict:
        """
        Read reinforcement learning infeasible indices.
        """
        condition_flags_ = {}
        for k, v in condition_flags.items():
            dict = {}
            for j in range(len(self.Q_init)):
                dict["Q_init_" + str(abs(self.Q_init[j]))] = Data.read_pickle(
                    self.regex(
                        self.RL_FULL_STATE,
                        Q_init=self.Q_init[j],
                        condition_flag=True,
                        condition_flag_type=k,
                    )
                )
            condition_flags_[k] = dict
        return condition_flags_

    def find_q_function(self, path) -> dict:
        """
        Finds summer/winter q-function updates.
        """
        rx = {
            "summer": "q_function_summer_ep_all_L_" + str(self.L) + ".*?",
            "winter": "q_function_winter_ep_all_L_" + str(self.L) + ".*?",
        }
        files = {}
        for file in os.listdir(path):
            for key, value in rx.items():
                match = re.match(value, file)
                if match:
                    files[key] = Data.read_pickle(os.path.join(path, file))
        return files

    def find_reward_function(self, path) -> list:
        """
        Finds the reward function for specified path and pipe length.
        """
        rx = "reward_function_training_ep_all_L_" + str(self.L) + "_Q_0_" ".*?"
        for file in os.listdir(path):
            match = re.match(rx, file)
            if match:
                reward_function = Data.read_pickle(os.path.join(path, file))
                return reward_function

    def __create_visual_csv(self) -> None:
        """
        Generates and stores csv file of results.
        """
        cumulative_profit = zip(
            self.__upper_bound_sum,
            self.__basic_strategy_sum,
            self.__lp_sum,
            self.__nlmip_sum,
            *self.__rl_full_state.values()
        )
        with open(
            os.path.join(self.CURRENT, "cumulative_profit_L=" + str(self.L) + ".txt"),
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)
            for row in cumulative_profit:
                if row:
                    writer.writerow(row)

    def __generate_plot(self):
        self.__plot.plot()

    def __plot_single_day(self, day_index):
        self.__plot.plot_single_day(day_index)

    def __plot_q_function(self):
        self.__plot.plot_q_function()

    def __plot_reward_function(self):
        self.__plot.plot_reward_function()

    @staticmethod
    def read_pickle(path):
        """
        Reads pickle file based on the path.
        """
        return pickle.load(open(path, "rb"))


class NoUnstableIndices(Data):
    def __init__(self, N, L, Q_init, number_of_episodes):
        super().__init__(N=N, L=L, Q_init=Q_init, number_of_episodes=number_of_episodes)
        self.__dir = self.IMAGES / "stable_index"
        self.__heat_demand: list = self.eliminate_unstable_indices(
            deepcopy(self._Data__heat_demand)
        )
        self.__electricity_price: list = self.eliminate_unstable_indices(
            deepcopy(self._Data__electricity_price)
        )
        self.__upper_bound_sum: list = self.eliminate_unstable_indices(
            deepcopy(self._Data__upper_bound_sum)
        )
        self.__basic_strategy_sum: list = self.eliminate_unstable_indices(
            deepcopy(self._Data__basic_strategy_sum)
        )
        self.__heat_demand_var: list = self.eliminate_unstable_indices(
            deepcopy(self._Data__heat_demand_var)
        )
        self.__electricity_price_var: list = self.eliminate_unstable_indices(
            deepcopy(self._Data__electricity_price_var)
        )
        self.__covariance: list = self.eliminate_unstable_indices(
            deepcopy(self._Data__covariance)
        )
        self.__lp_sum: list = self.eliminate_unstable_indices(
            deepcopy(self._Data__lp_sum)
        )
        self.__nlmip_sum: list = self.eliminate_unstable_indices(
            deepcopy(self._Data__nlmip_sum)
        )
        self.__rl_full_state: dict = self.eliminate_unstable_indices(
            deepcopy(self._Data__rl_full_state)
        )
        self.__condition_flags: dict = self.eliminate_unstable_indices(
            deepcopy(self._Data__condition_flags), condition_flag=True
        )
        self.condition_flags_nlmip: dict = self.read_nlmip_infeasible_indices()
        self.violation_percentage_nlmip: dict = Data.read_pickle(
            os.path.join(
                self.NLMIP_SIM_FLAG, "percentage_violation_ep_all_L_" + str(self.L)
            )
        )
        self.violation_percentage_rl_full_state: dict = (
            self.read_percentage_violation_rl_state(Data.RL_FULL_STATE_SIM_FLAG)
        )
        self.eliminate_data_overlap()
        self.__plot: Plots = Plots(
            self.N,
            self.L,
            self.Q_init,
            self.number_of_episodes,
            self.__dir,
            self.__heat_demand_var,
            self.__electricity_price_var,
            self.__covariance,
            self.__upper_bound_sum,
            self.__basic_strategy_sum,
            self.__lp_sum,
            self.__nlmip_sum,
            self.nlmip_unstable_indices,
            self.__rl_full_state,
            self.__condition_flags,
            self.condition_flags_nlmip,
            self.violation_percentage_nlmip,
            self.violation_percentage_rl_full_state,
            self.q_function_full,
            self.q_function_abstract,
            self.reward_function_full,
            self.reward_function_abstract,
        )

    def read_percentage_violation_rl_state(self, path):
        """
        Reading percentage violation for rl, and eliminating infeasible indices from nlmip.
        """
        dict = {}
        for Q in self.Q_init:
            dict["Q_init_" + str(abs(Q))] = Data.read_pickle(
                os.path.join(
                    path,
                    "percentage_violation_ep_all_L_{}_Q_{}".format(self.L, Q),
                )
            )
        for k1, v1 in dict.items():
            for k2, v2 in condition_flags.items():
                self.delete(dict[k1][k2])
        return dict

    def delete(self, data) -> None:
        """
        Deletes data according to the MINLP unstable indices.
        """
        for i in reversed(self.nlmip_unstable_indices):
            del data[i]

    def eliminate_unstable_indices(self, data, condition_flag=False):
        """
        As MINLP has no solution for some episodes, for the comparison purposes
        we eliminate those episodes from all the results.
        """
        if condition_flag:
            for k1, v1 in condition_flags.items():
                for k2, v2 in data[k1].items():
                    self.delete(data[k1][k2])
        elif type(data) is dict:
            for k, v in data.items():
                self.delete(data[k])
        else:
            self.delete(data)
        return data

    def eliminate_data_overlap(self):
        """
        Eliminates data overlap.
        """
        temp_heat_electricity = []
        for i in range(len(self.__heat_demand)):
            temp_heat_electricity.append(
                self.__electricity_price[i] + self.__heat_demand[i]
            )
        temp_heat_electricity_new = []
        for i, e in reversed(list(enumerate(temp_heat_electricity))):
            if e not in temp_heat_electricity_new:
                temp_heat_electricity_new.append(e)
            else:
                self.__heat_demand_var.pop(i)
                self.__electricity_price_var.pop(i)
                self.__covariance.pop(i)
                self.__upper_bound_sum.pop(i)
                self.__basic_strategy_sum.pop(i)
                self.__lp_sum.pop(i)
                self.__nlmip_sum.pop(i)
                for j in range(len(self.Q_init)):
                    self.__rl_full_state["Q_init_" + str(abs(self.Q_init[j]))].pop(i)
                    for k, v in condition_flags.items():
                        self.__condition_flags[k][
                            "Q_init_" + str(abs(self.Q_init[j]))
                        ].pop(i)
                        self.violation_percentage_rl_full_state[
                            "Q_init_" + str(abs(self.Q_init[j]))
                        ][k].pop(i)
                for k, v in condition_flags.items():
                    self.condition_flags_nlmip[k].pop(i)
                    self.violation_percentage_nlmip[k].pop(i)

    def __generate_plot(self):
        self.__plot.plot()

    def __plot_single_day(self, day_index):
        self.__plot.plot_single_day(day_index)

    def __plot_q_function(self):
        self.__plot.plot_q_function()

    def __plot_reward_function(self):
        self.__plot.plot_reward_function()

    def __paired_t_test(self):
        """
        Performs two paired t-test: The first between LP and MINLP,
        and the second between MINLP and Q-learning full state space (zero initial).
        """
        t = []
        for i in self.__nlmip_sum:
            if type(i) == np.ndarray:
                t.append(i[0])
            else:
                t.append(i)
        return (
            ttest_rel(self.__lp_sum, t),
            ttest_rel(self.__lp_sum, self.__rl_full_state["Q_init_0"]),
        )


class Plots:
    def __init__(
        self,
        N,
        L,
        Q_init,
        number_of_episodes,
        path,
        heat_demand_var,
        electricity_price_var,
        covariance,
        upper_bound,
        basic_strategy,
        lp_sum,
        nlmip_sum,
        nlmip_unstable_indices,
        rl_full_state,
        condition_flags_,
        condition_flags_nlmip={},
        violation_percentage_nlmip={},
        violation_percentage_rl_full_state={},
        q_function_full={},
        q_function_abstract={},
        reward_function_full=[],
        reward_function_abstract=[],
    ):
        self.N = N
        self.M = len(lp_sum)
        self.L = L
        self.L_km = int(self.L / 1000)
        self.Q_init = Q_init
        self.number_of_episodes = number_of_episodes
        self.dir = path
        self.heat_demand_var = heat_demand_var
        self.heat_demand_var_unique = list(set(self.heat_demand_var))
        self.electricity_price_var = electricity_price_var
        self.electricity_price_var_unique = list(set(self.electricity_price_var))
        self.covariance = covariance
        self.upper_bound = upper_bound
        self.basic_strategy = basic_strategy
        self.lp_sum = lp_sum
        self.nlmip_sum = [elem.item() for elem in nlmip_sum]
        self.nlmip_unstable_indices = nlmip_unstable_indices
        self.rl_full_state = rl_full_state
        self.condition_flags = condition_flags_
        self.condition_flags_nlmip = condition_flags_nlmip
        self.violation_percentage_nlmip = violation_percentage_nlmip
        self.violation_percentage_rl_full_state = violation_percentage_rl_full_state
        self.q_function_full = q_function_full
        self.q_function_abstract = q_function_abstract
        self.reward_function_full: list = reward_function_full
        self.reward_function_abstract: list = reward_function_abstract
        self.colors = {
            "Upper bound": "r",
            "LP": "b",
            "Q$^{full}$": "g",
            "MINLP": "y",
            "Basic strategy": "k",
        }

    def reformat_condition_flag(self, condition_flag) -> Tuple[list, list]:
        """
        Reformation of condition flags.
        """
        temp = {
            k: round((v / self.M) * 100, 2)
            for k, v in dict(
                sorted(dict(Counter(condition_flag)).items(), key=lambda item: item[0])
            ).items()
        }
        x = list(np.cumsum(list(temp.values())))
        y = list(temp.keys())
        x.insert(0, 0)
        y.insert(0, 0)
        return x, y

    def group_profit(self, variance_unique, variance, profit) -> list:
        """
        Grouping profit based on the variance.
        """
        profit_group = []
        variance_unique.sort()
        for i in variance_unique:
            profit_group_unique = []
            for j in range(self.M):
                if i == variance[j]:
                    profit_group_unique.append(profit[j])
            profit_group.append(profit_group_unique)
        return profit_group

    def set_box_color(self, bp, color):
        """
        Setting colors for box plots according to the algorithm category.
        """
        plt.setp(bp["boxes"], color=color)
        plt.setp(bp["whiskers"], color=color)
        plt.setp(bp["caps"], color=color)
        plt.setp(bp["medians"], color=color)

    def plot(self) -> None:
        """
        Generation of all plots.
        Plots include ones from the paper, but also additional plots that provide more insight.
        """
        # Performance
        # Cumulative profit
        fig, ax = plt.subplots()
        title = "Cumulative profit for L={} km".format(int(self.L_km))
        ax.plot(self.upper_bound, label="Upper bound")
        ax.plot(self.basic_strategy, label="Basic strategy")
        ax.plot(self.lp_sum, label="LP")
        ax.plot(self.nlmip_sum, label="MINLP")
        ax.plot(
            self.rl_full_state["Q_init_" + str(abs(self.Q_init[0]))],
            label="Q$^{full}$",
        )
        ax.set_title(title)
        ax.set_xlabel("Day")
        ax.set_ylabel("Profit [\u20ac]")
        ax.legend()
        plt.subplots_adjust(
            wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.96, top=0.92
        )
        fig.savefig(os.path.join(self.dir, title + ".png"))
        plt.show()

        # Box-plots
        fig, ax = plt.subplots()
        title = "Profit for L={} km".format(int(self.L_km))
        boxplot_ = [
            self.upper_bound,
            self.basic_strategy,
            self.lp_sum,
            self.nlmip_sum,
        ]
        labels_ = ["UB", "BS", "LP", "MINLP"]
        boxplot_.append(self.rl_full_state["Q_init_" + str(abs(self.Q_init[0]))])
        labels_.append("Q$^{full}$")
        ax.set_title(title)
        ax.set_ylabel("Profit [\u20ac]")
        ax.boxplot(boxplot_, labels=labels_, notch=True)
        plt.subplots_adjust(
            wspace=0.8, hspace=0.6, left=0.15, bottom=0.1, right=0.96, top=0.92
        )
        fig.savefig(os.path.join(self.dir, title + ".png"))
        plt.show()

        # Scatter profit plot
        fig, ax = plt.subplots()
        title = "Scatter profit LP - MINLP&RL for L = {} km".format(int(self.L_km))
        ax.scatter(self.lp_sum, self.nlmip_sum, s=15, marker="*", label="MINLP")
        ax.scatter(
            self.lp_sum,
            self.rl_full_state["Q_init_" + str(abs(self.Q_init[0]))],
            s=15,
            marker="*",
            label="Q$^{full}$",
        )
        ax.set_title(title)
        ax.set_xlabel("LP")
        ax.set_ylabel("Profit [\u20ac]")
        ax.legend()
        plt.subplots_adjust(
            wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.96, top=0.92
        )
        fig.savefig(os.path.join(self.dir, title + ".png"))
        plt.show()

        # Statistics
        fig, ax = plt.subplots()
        title = "Variance of heat demand and MINLP&RL for L = {} km".format(self.L_km)
        ax.scatter(
            self.heat_demand_var, self.nlmip_sum, s=15, marker="*", label="MINLP"
        )
        ax.scatter(
            self.heat_demand_var,
            self.rl_full_state["Q_init_" + str(abs(self.Q_init[0]))],
            s=15,
            marker="*",
            label="Q-learning " + str(self.Q_init[0]),
        )
        ax.set_title(title)
        ax.set_xlabel("Heat demand variance")
        ax.set_ylabel("Profit [\u20ac]")
        ax.legend()
        plt.subplots_adjust(
            wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.96, top=0.92
        )
        fig.savefig(os.path.join(self.dir, title + ".png"))
        plt.show()

        fig, ax = plt.subplots()
        title = "Heat demand variance for L={} km".format(int(self.L_km))
        nlmip_heat_var = self.group_profit(
            self.heat_demand_var_unique, self.heat_demand_var, self.nlmip_sum
        )
        rl_heat_var = self.group_profit(
            self.heat_demand_var_unique,
            self.heat_demand_var,
            self.rl_full_state["Q_init_" + str(abs(self.Q_init[0]))],
        )
        bpl = plt.boxplot(
            nlmip_heat_var,
            positions=np.array(range(len(self.heat_demand_var_unique))) * 2.0 - 0.4,
            sym="",
            widths=0.6,
        )
        bpr = plt.boxplot(
            rl_heat_var,
            positions=np.array(range(len(self.heat_demand_var_unique))) * 2.0 + 0.4,
            sym="",
            widths=0.6,
        )
        self.set_box_color(bpl, "y")  # colors are from http://colorbrewer2.org/
        self.set_box_color(bpr, "g")
        ax.plot([], c="y", label="MINLP")
        ax.plot([], c="g", label="Q$^{full}$")
        ax.set_title(title)
        ax.set_xlabel("Variance")
        ax.set_ylabel("Profit [\u20ac]")
        ax.legend()
        plt.xticks(
            range(0, len(self.heat_demand_var_unique) * 2, 2),
            self.heat_demand_var_unique,
        )
        plt.subplots_adjust(
            wspace=0.7, hspace=0.7, left=0.15, bottom=0.1, right=0.96, top=0.92
        )
        fig.savefig(os.path.join(self.dir, title + ".png"))
        plt.show()

        fig, ax = plt.subplots()
        title = "Variance of electricity price and MINLP&RL for L = {} km".format(
            self.L_km
        )
        ax.scatter(
            self.electricity_price_var, self.nlmip_sum, s=15, marker="*", label="MINLP"
        )
        ax.scatter(
            self.electricity_price_var,
            self.rl_full_state["Q_init_" + str(abs(self.Q_init[0]))],
            s=15,
            marker="*",
            label="Q$^{full}$",
        )
        ax.set_title(title)
        ax.set_xlabel("Electricity price variance")
        ax.set_ylabel("Profit [\u20ac]")
        ax.legend()
        plt.subplots_adjust(
            wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.96, top=0.92
        )
        fig.savefig(os.path.join(self.dir, title + ".png"))
        plt.show()

        fig, ax = plt.subplots()
        title = "Electricity price variance for L={} km".format(int(self.L_km))
        nlmip_heat_var = self.group_profit(
            self.electricity_price_var_unique,
            self.electricity_price_var,
            self.nlmip_sum,
        )
        rl_heat_var = self.group_profit(
            self.electricity_price_var_unique,
            self.electricity_price_var,
            self.rl_full_state["Q_init_" + str(abs(self.Q_init[0]))],
        )
        bpl = plt.boxplot(
            nlmip_heat_var,
            positions=np.array(range(len(self.electricity_price_var_unique))) * 2.0
            - 0.4,
            sym="",
            widths=0.6,
        )
        bpr = plt.boxplot(
            rl_heat_var,
            positions=np.array(range(len(self.electricity_price_var_unique))) * 2.0
            + 0.4,
            sym="",
            widths=0.6,
        )
        self.set_box_color(bpl, "y")  # colors are from http://colorbrewer2.org/
        self.set_box_color(bpr, "g")
        ax.plot([], c="y", label="MINLP")
        ax.plot([], c="g", label="Q$^{full}$")
        ax.set_title(title)
        ax.set_xlabel("Variance")
        ax.set_ylabel("Profit [\u20ac]")
        ax.legend()
        plt.xticks(
            range(0, len(self.electricity_price_var_unique) * 2, 2),
            self.electricity_price_var_unique,
        )
        plt.subplots_adjust(
            wspace=0.7, hspace=0.7, left=0.15, bottom=0.1, right=0.96, top=0.92
        )
        fig.savefig(os.path.join(self.dir, title + ".png"))
        plt.show()

        fig = plt.figure()
        ax = Axes3D(fig)
        title = "Variance of heat demand, electricity and MINLP&RL for L = {}".format(
            self.L
        )
        ax.scatter(
            self.heat_demand_var,
            self.electricity_price_var,
            self.nlmip_sum,
            s=15,
            marker="*",
            label="MINLP",
        )
        ax.scatter(
            self.heat_demand_var,
            self.electricity_price_var,
            self.rl_full_state["Q_init_" + str(abs(self.Q_init[0]))],
            s=15,
            marker="*",
            label="Q$^{full}$",
        )
        ax.set_title(title)
        ax.set_xlabel("Heat demand variance")
        ax.set_ylabel("Electricity price variance")
        ax.set_zlabel("Profit [\u20ac]")
        ax.legend()
        fig.savefig(os.path.join(self.dir, title + ".png"))
        plt.show()

        fig, ax = plt.subplots()
        title = "Covariance and MINLP&RL with L = {} km".format(self.L_km)
        ax.scatter(self.covariance, self.nlmip_sum, s=15, marker="*", label="MINLP")
        ax.scatter(
            self.covariance,
            self.rl_full_state["Q_init_" + str(abs(self.Q_init[0]))],
            s=15,
            marker="*",
            label="Q$^{full}$",
        )
        ax.set_title(title)
        ax.set_xlabel("Covariance")
        ax.set_ylabel("Profit [\u20ac]")
        ax.legend()
        plt.subplots_adjust(
            wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.96, top=0.92
        )
        fig.savefig(os.path.join(self.dir, title + ".png"))
        plt.show()

        # Stability
        fig, ax = plt.subplots()
        title = "MINLP (un)stable operation for L={} km".format(int(self.L / 1000))
        ax.bar(
            [0, 1],
            height=[
                self.N - len(self.nlmip_unstable_indices),
                len(self.nlmip_unstable_indices),
            ],
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Stable", "Unstable"])
        ax.set_title(title)
        ax.set_ylabel("Number of days")
        fig.savefig(os.path.join(self.dir, title + ".png"))
        plt.show()

        # Feasibility
        for k, v in condition_flags.items():
            title = v + " for L = {} km".format(self.L_km)
            fig, ax = plt.subplots()
            x1, y1 = self.reformat_condition_flag(self.condition_flags_nlmip[k])
            x2, y2 = self.reformat_condition_flag(
                self.condition_flags[k]["Q_init_" + str(abs(self.Q_init[0]))]
            )
            ax.plot(x1, y1, self.colors["MINLP"], label="MINLP")
            Plots.plot_stars(ax, x1, y1, color=self.colors["MINLP"])
            ax.plot(
                x2,
                y2,
                self.colors["Q$^{full}$"],
                label="Q$^{full}$" + str(self.Q_init[0]),
            )
            Plots.plot_stars(ax, x2, y2, color=self.colors["Q$^{full}$"])
            ax.set_xlabel("Percentage of days [%]")
            ax.set_ylabel("Violation occurrence with respect to the episode step")
            ax.legend()
            ax.set_title(title)
            fig.savefig(os.path.join(self.dir, title + ".png"))
            plt.show()
            title = v + " for L={} km".format(int(self.L / 1000))
            fig, ax = plt.subplots()
            x1, y1 = self.reformat_condition_flag(self.violation_percentage_nlmip[k])
            x2, y2 = self.reformat_condition_flag(
                self.violation_percentage_rl_full_state[
                    "Q_init_" + str(abs(self.Q_init[0]))
                ][k]
            )
            ax.plot(x1, y1, self.colors["MINLP"], label="MINLP")
            Plots.plot_stars(ax, x1, y1, color=self.colors["MINLP"])
            ax.plot(
                x2,
                y2,
                self.colors["Q$^{full}$"],
                label="Q$^{full}$ ",  # + str(self.Q_init[0])
            )
            Plots.plot_stars(ax, x2, y2, color=self.colors["Q$^{full}$"])
            ax.set_xlabel("Percentage of days [%]")
            ax.set_ylabel("Violation percentage [%]")
            ax.legend()
            ax.set_title(title)
            fig.savefig(os.path.join(self.dir, title + ".png"))
            plt.show()

    def find_file(self, path, start_string, day_index, lp=False, rl=False, Q_init=0):
        """
        Find file according to the specifications.
        """
        base_search = start_string + "_ep_" + str(day_index)
        if not lp:
            base_search += "_L_" + str(self.L)
        if rl:
            base_search += "_Q_" + str(Q_init)
        for filename in os.listdir(path):
            if filename.startswith(base_search):
                break
        return os.path.join(path, filename)

    def get_single_day(self, day_index) -> dict:
        """
        Extract upper bound, basic strategy, Q^{full}, MINLP and LP data for the specified day.
        """
        upper_bound = pickle.load(
            open(self.find_file(DataPaths.BOUND, "upper_bound", day_index), "rb")
        )
        upper_bound_sum = Plots.find_sum(upper_bound)
        basic_strategy = pickle.load(
            open(self.find_file(DataPaths.BOUND, "basic_strategy", day_index), "rb")
        )
        basic_strategy_sum = Plots.find_sum(basic_strategy)
        lp = pickle.load(
            open(
                self.find_file(DataPaths.LP, "data_episode", day_index, lp=True),
                "rb",
            )
        )["objective_function"]
        lp_sum = Plots.find_sum(lp)
        nlmip = pickle.load(
            open(
                self.find_file(DataPaths.NLMIP_OPT, "data_episode", day_index),
                "rb",
            )
        )["objective_function"]
        nlmip_sum = Plots.find_sum(nlmip)
        rl = pickle.load(
            open(
                self.find_file(
                    DataPaths.RL_FULL_STATE, "data_episode", day_index, rl=True
                ),
                "rb",
            )
        )["objective_function"]
        rl_sum = Plots.find_sum(rl)
        return {
            "Upper bound": upper_bound_sum,
            "Basic strategy": basic_strategy_sum,
            "LP": lp_sum,
            "MINLP": nlmip_sum,
            "Q$^{full}$": rl_sum,
        }

    def plot_single_day(self, day_index) -> None:
        """
        Plot cumulative profit for a single day based on the index of that day.
        """
        title = "Cumulative profit (single day) for L={} km".format(int(self.L_km))
        fig, ax = plt.subplots()
        data = self.get_single_day(day_index)
        y2, y3, temp1, temp2 = [], [], [], []
        for i in range(len(data["MINLP"])):
            y2.append(data["MINLP"][i][0])
            y3.append(data["Q$^{full}$"][i])
        for i in range(len(y2)):
            if y2[i] > data["LP"][i]:
                temp1.append(1)
            else:
                temp1.append(0)
            if y3[i] > data["LP"][i]:
                temp2.append(1)
            else:
                temp2.append(0)
        for k, v in data.items():
            ax.plot(v, self.colors[k], label=k)
            Plots.plot_stars(ax, list(range(TIME_HORIZON)), v, color=self.colors[k])
        ax.fill_between(
            list(range(TIME_HORIZON)),
            data["LP"],
            y2,
            where=temp1,
            facecolor="none",
            hatch="X",
            edgecolor="y",
            linewidth=0.2,
        )
        ax.fill_between(
            list(range(TIME_HORIZON)),
            data["LP"],
            y3,
            where=temp2,
            facecolor="none",
            hatch="X",
            edgecolor="g",
            linewidth=0.2,
        )
        ax.set_xlabel("Hour")
        ax.set_ylabel("Profit [\u20ac]")
        ax.set_title(title)
        ax.legend()
        fig.savefig(os.path.join(self.dir, title + ".png"))
        plt.show()

    def plot_q_function(self):
        """
        Plot updates to Q-function for randomly selected day from summer and winter season
        (external part of the state space).
        """
        plt.plot(self.q_function_full["summer"], c="red", label="$Q^{full}_{summer}$")
        plt.plot(self.q_function_full["winter"], c="green", label="$Q^{full}_{winter}$")
        plt.plot(
            self.q_function_abstract["summer"], c="blue", label="$Q^{par}_{summer}$"
        )
        plt.plot(
            self.q_function_abstract["winter"], c="yellow", label="$Q^{par}_{winter}$"
        )
        plt.xlabel("Episode")
        plt.ylabel("Q-value")
        plt.title("Q-value function updates for L={} km".format(self.L_km))
        plt.legend()
        plt.show()

    def plot_reward_function(self):
        """
        Plot cumulative reward function.
        """
        x = list(
            range(
                int(self.number_of_episodes / 100),
                self.number_of_episodes + int(self.number_of_episodes / 100),
                int(self.number_of_episodes / 100),
            )
        )
        plt.plot(x, self.reward_function_full, c="red", label="$Q^{full}$")
        plt.plot(x, self.reward_function_abstract, c="blue", label="$Q^{par}$")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(
            "Cumulative reward on training dataset for L= {} km".format(self.L_km)
        )
        plt.legend()
        plt.show()

    @staticmethod
    def plot_stars(ax, x, y, color):
        """
        Plot star on every full hour of the cumulative profit function.
        """
        for i in range(len(x)):
            ax.plot(x[i], y[i], color + "*", markersize=7)
        return ax

    @staticmethod
    def find_sum(data) -> list:
        """
        Find cumulative sum.
        """
        data_sum = []
        for i in range(1, len(data) + 1):
            data_sum.append(sum(data[:i]))
        return data_sum
