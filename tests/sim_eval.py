import os
import numpy as np
import pickle
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from ..src.simulator.interfaces.keypts import Sim_keypts
from util.util import *


class SimEval(ABC):
    """
    Simulator evaluation.
    """

    def __init__(
        self,
        pipe_len,
        pipe_diameter,
        max_flow_speed,
        min_flow_speed,
        max_supply_temp,
        min_supply_temp,
        max_return_temp,
        min_return_temp,
        historical_t_supply,
        historical_t_return,
    ):
        self.path: Path = Path(__file__).parents[1]
        self.data: Path = self.path / "data"
        self.heat_demand: np.array = np.array(
            pd.read_csv(self.data / "heat_demand_test.csv")
        )
        self.electricity_price: np.array = np.array(
            pd.read_csv(self.data / "day_ahead_electricity_price_test.csv")
        )
        self.pipe_len: int = pipe_len
        self.max_flow_speed = max_flow_speed
        self.min_flow_speed = min_flow_speed
        self.max_supply_temp = max_supply_temp
        self.min_supply_temp = min_supply_temp
        self.max_return_temp = max_return_temp
        self.min_return_temp = min_return_temp
        self.condition_flags: list = list(condition_flags.keys())
        self.violations_percent_: dict = {}
        for k in self.condition_flags:
            self.violations_percent_[k] = []
        self.nlmip_unstable_index: list = []
        self.sim_interface = Sim_keypts(
            heat_demand=self.heat_demand[0] / (10 ** 6),
            pipe_len=self.pipe_len,
            pipe_diameter=pipe_diameter,
            historical_t_supply=historical_t_supply,
            historical_t_return=historical_t_return,
            max_flow_speed=self.max_flow_speed,
            min_flow_speed=self.min_flow_speed,
            max_supply_temp=self.max_supply_temp,
            min_supply_temp=self.min_supply_temp,
            min_return_temp=self.min_return_temp,
            control_with_temp=False,
        )

    @abstractmethod
    def load_data(self):
        """
        Load the data.
        """
        pass

    @abstractmethod
    def run(self):
        """
        Run the evaluation.
        """
        pass

    def get_add_violation(self, grid, N, ep) -> list:
        """
        Part of the violations is provided by the simulator,
        and part of the violations -- minimal supply inlet temperature
        and minimal return inlet temperatures are calculated here.
        """
        for producer in grid.producers:
            sup_temp = producer.temp[0]
        for consumer in grid.consumers:
            ret_in_temp = consumer.temp[1]
        for i in range(len(sup_temp[:N])):
            if np.isnan(sup_temp[i]):
                print("Length " + str(self.pipe_len))
                print("Ep " + str(ep))
                print("Episodes step " + str(i))
        violations = [
            np.minimum(sup_temp - self.min_supply_temp, 0)[:N],
            np.minimum(ret_in_temp - self.min_return_temp, 0)[:N],
        ]
        return violations

    def cal_violation_percent(self, viol1, viol2, heat_demand) -> dict:
        """
        Transforming violation into percentage.
        """
        percent = {}
        percent[self.condition_flags[0]] = [
            abs(viol1[0][i]) * 100 / (10 ** 6) / heat_demand[i]
            for i in range(len(viol1[0]))
        ]
        percent[self.condition_flags[1]] = [
            abs(i) * 100 / (self.max_supply_temp - self.min_supply_temp)
            for i in viol1[2]
        ]
        percent[self.condition_flags[2]] = [
            abs(i) * 100 / (self.max_flow_speed - self.min_flow_speed) for i in viol1[1]
        ]
        percent[self.condition_flags[3]] = [
            abs(i) * 100 / (self.max_supply_temp - self.min_supply_temp)
            for i in viol2[0]
        ]
        percent[self.condition_flags[4]] = [
            abs(i) * 100 / (self.max_return_temp - self.min_return_temp)
            for i in viol2[1]
        ]
        for k in self.condition_flags:
            self.violations_percent_[k].append(np.mean(percent[k]))
        return percent

    @staticmethod
    def get_ep(filename) -> int:
        """
        Extract episode from the file name.
        """
        ep_start_idx = filename.find("ep_") + len("ep_")
        ep_end_idx = filename.find("_", ep_start_idx)
        ep = int(filename[ep_start_idx:ep_end_idx])
        return ep

    @staticmethod
    def get_len(filename) -> int:
        """
        Extract pipe length from the filename.
        """
        length_start_idx = filename.find("L_") + len("L_")
        length_end_idx = filename.find("_", length_start_idx)
        length = int(filename[length_start_idx:length_end_idx])
        return length


class MinlpEval(SimEval):
    """
    Simulator evaluation of the MINLP algorithm.
    """

    def __init__(
        self,
        pipe_len,
        pipe_diameter,
        max_flow_speed,
        min_flow_speed,
        max_supply_temp,
        min_supply_temp,
        max_return_temp,
        min_return_temp,
        historical_t_supply,
        historical_t_return,
    ):
        super().__init__(
            pipe_len,
            pipe_diameter,
            max_flow_speed,
            min_flow_speed,
            max_supply_temp,
            min_supply_temp,
            max_return_temp,
            min_return_temp,
            historical_t_supply,
            historical_t_return,
        )
        self.result_p: Path = self.path / "results/li_2016/li_2016_day_ahead"
        self.store_p: Path = self.path / "results/li_2016/li_2016_day_ahead_sim"
        self.result_file_form = "data_episode_ep_{}_L_" + str(self.pipe_len)
        self.percent_file_form = "percentage_violation_ep_{}_L_" + str(self.pipe_len)
        self.results: dict = self.load_data()

    def load_data(self) -> dict:
        """
        Load the data.
        """
        result_dict = {}
        for (dirpath, dirnames, filenames) in os.walk(self.result_p):
            for filename in filenames:
                if filename.endswith(".pickle") & (filename[:4] == "data"):
                    ep = self.get_ep(filename=filename)
                    len = self.get_len(filename=filename)
                    if self.pipe_len == len:
                        with open(os.path.join(dirpath, filename), "rb") as f:
                            x = pickle.load(f)
                            result_dict[ep] = x
        return result_dict

    def run(self):
        """
        Run the evaluation, and save final percentage of violations.
        """
        for i, (demand, e_price) in enumerate(
            zip(self.heat_demand, self.electricity_price)
        ):
            ep = i + 1
            self.sim_interface.update(demand, e_price)
            action = self.results.get(ep)["action"]
            if np.any((np.count_nonzero(action, axis=1)) == 0):
                self.nlmip_unstable_index.append(ep - 1)
                continue
            (
                cost,
                heat_delivered,
                electricity_delivered,
                violations,
            ) = self.sim_interface.run(action)

            add_violations = self.get_add_violation(
                grid=self.sim_interface.grid, N=len(action), ep=ep
            )
            violations_percent = self.cal_violation_percent(
                viol1=violations, viol2=add_violations, heat_demand=demand
            )
            """
            with open(
                os.path.join(self.store_p, self.percent_file_form.format(ep)),
                "wb",
            ) as handle:
                pickle.dump(
                    violations_percent, handle, protocol=pickle.HIGHEST_PROTOCOL
                )
        with open(
            os.path.join(
                self.store_p, "nlmip_unstable_indices_L_" + str(self.pipe_len)
            ),
            "wb",
        ) as handle:
            pickle.dump(
                self.nlmip_unstable_index, handle, protocol=pickle.HIGHEST_PROTOCOL
            )
        with open(
            os.path.join(
                self.store_p, "percentage_violation_ep_all_L_" + str(self.pipe_len)
            ),
            "wb",
        ) as handle:
            pickle.dump(
                self.violations_percent_, handle, protocol=pickle.HIGHEST_PROTOCOL
            )
        """


class RlEval(SimEval):
    """
    Simulator evaluation of the full state space Q-learning algorithm.
    """

    def __init__(
        self,
        pipe_len,
        pipe_diameter,
        max_flow_speed,
        min_flow_speed,
        max_supply_temp,
        min_supply_temp,
        max_return_temp,
        min_return_temp,
        historical_t_supply,
        historical_t_return,
        Q_init,
    ):
        super().__init__(
            pipe_len,
            pipe_diameter,
            max_flow_speed,
            min_flow_speed,
            max_supply_temp,
            min_supply_temp,
            max_return_temp,
            min_return_temp,
            historical_t_supply,
            historical_t_return,
        )
        self.result_p: Path = self.path / "results/rl_full_state/rl_full_state"
        self.store_p: Path = self.path / "results/rl_full_state/rl_full_state_sim"
        self.Q_init = Q_init
        self.result_file_form = (
            "data_episode_ep_{}_L_" + str(self.pipe_len) + "_Q_" + str(self.Q_init)
        )
        self.percent_file_form = (
            "percentage_violation_ep_{}_L_"
            + str(self.pipe_len)
            + "_Q_"
            + str(self.Q_init)
        )
        self.results: dict = self.load_data()

    def load_data(self) -> dict:
        """
        Load the data.
        """
        result_dict = {}
        for (dirpath, dirnames, filenames) in os.walk(self.result_p):
            for filename in filenames:
                if filename.endswith(".pickle") & (filename[:4] == "data"):
                    ep = self.get_ep(filename=filename)
                    length = self.get_len(filename=filename)
                    q_start_idx = filename.find("Q_") + len("Q_")
                    q_end_idx = filename.find("_", q_start_idx)
                    q = int(filename[q_start_idx:q_end_idx])
                    if self.pipe_len == length and q == self.Q_init:
                        with open(os.path.join(dirpath, filename), "rb") as f:
                            x = pickle.load(f)
                            result_dict[ep] = x
        return result_dict

    def run(self):
        """
        Run the evaluation by the simulator, and save final percentage of violations.
        """
        for i, (demand, e_price) in enumerate(
            zip(self.heat_demand, self.electricity_price)
        ):
            ep = i + 1
            self.sim_interface.update(demand, e_price)
            action = self.results.get(ep)["action"]

            (
                cost,
                heat_delivered,
                electricity_delivered,
                violation_heat,
                violation_mass_flow,
                violation_max_supply_in_temp,
            ) = ([], [], [], [], [], [])
            for act in action:
                (
                    cost_one_step,
                    heat_delivered_one_step,
                    electricity_delivered_one_step,
                    violations_one_step,
                ) = self.sim_interface.run_one_step(act)
                cost.append(cost_one_step)
                heat_delivered.append(heat_delivered_one_step)
                electricity_delivered.append(electricity_delivered_one_step)
                violation_heat.append(violations_one_step[0])
                violation_mass_flow.append(violations_one_step[1])
                violation_max_supply_in_temp.append(violations_one_step[2])

            violations = [
                violation_heat,
                violation_mass_flow,
                violation_max_supply_in_temp,
            ]
            add_violations = self.get_add_violation(
                self.sim_interface.grid, len(action), ep
            )
            violations_percent = self.cal_violation_percent(
                viol1=violations, viol2=add_violations, heat_demand=demand
            )
            """
            with open(
                os.path.join(self.store_p, self.percent_file_form.format(ep)),
                "wb",
            ) as handle:
                pickle.dump(
                    violations_percent, handle, protocol=pickle.HIGHEST_PROTOCOL
                )
        with open(
            os.path.join(
                self.store_p, "percentage_violation_ep_all_L_" + str(self.pipe_len)
            ),
            "wb",
        ) as handle:
            pickle.dump(
                self.violations_percent_, handle, protocol=pickle.HIGHEST_PROTOCOL
            )
        """
