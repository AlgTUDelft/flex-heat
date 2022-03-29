import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from datetime import datetime
from pathlib import *
from pyscipopt import Model, exp, quicksum
from util.util import *


class Abdollahi2015:
    """
    CHP optimal dispatch (without the grid dynamics)
    day-ahead (certain) and ancillary (uncertain) electricity markets are the same
    """

    def __init__(self, pipe_len):
        self.data_p = Path(__file__).parents[2] / "data"
        self.store_p = Path(__file__).parents[2] / "results/abdollahi_2015"
        self.pipe_len = pipe_len
        self.heat_demand = np.array(pd.read_csv(self.data_p / "heat_demand_test.csv"))
        self.electricity_price = np.array(
            pd.read_csv(self.data_p / "day_ahead_electricity_price_test.csv")
        )
        self.number_of_episodes = len(self.heat_demand)
        self.now = datetime.now().strftime("%H_%M_%S")
        self.average_heat_loss = 0.28
        self.keys = [(0, 0), (0, 1), (0, 2), (0, 3)]
        self.num_extreme_points: int = len(self.keys)
        self.extreme_points_heat: dict = dict(
            [
                (self.keys[0], 0),
                (self.keys[1], 10),
                (self.keys[2], 70),
                (self.keys[3], 0),
            ]
        )
        self.extreme_points_power: dict = dict(
            [
                (self.keys[0], 10),
                (self.keys[1], 5),
                (self.keys[2], 30),
                (self.keys[3], 50),
            ]
        )

    def run(self):
        objective_function = []
        for i in range(self.number_of_episodes):
            (
                objective_function_episode,
                heat_delivered_episode,
                electricity_delivered_episode,
            ) = ([], [], [])
            m = self.solve_chpd_model(self.heat_demand[i], self.electricity_price[i])
            alpha = self.get_alpha(m)
            for t in range(TIME_HORIZON):
                objective_function_episode.append(
                    self.obj_fun(t, alpha, self.electricity_price[i][t])
                )
                heat_delivered_episode.append(
                    sum(
                        alpha[t][k] * self.extreme_points_heat.get((0, k))
                        for k in range(self.num_extreme_points)
                    )
                )
                electricity_delivered_episode.append(
                    sum(
                        alpha[t][k] * self.extreme_points_power.get((0, k))
                        for k in range(self.num_extreme_points)
                    )
                )
            dict = {
                "action": alpha,
                "objective_function": objective_function_episode,
                "heat_delivered": heat_delivered_episode,
                "electricity_delivered": electricity_delivered_episode,
            }
            """
            save_to_pickle(
                data_path_store=self.store_p,
                variable=dict,
                variable_name="data_episode",
                now=self.now,
                pipe_len=self.pipe_len,
                ep=i + 1,
            )
            """
            objective_function.append(sum(objective_function_episode))
        """
        save_to_pickle(
            data_path_store=self.store_p,
            variable=objective_function,
            variable_name="objective_function_sum",
            now=self.now,
            pipe_len=self.pipe_len,
        )
        """

    def obj_fun(self, t, alpha, electricity_price):
        return (
            -a[0]
            * sum(
                alpha[t][k] * self.extreme_points_heat.get((0, k))
                for k in range(self.num_extreme_points)
            )
            - a[0] * self.average_heat_loss
            - a[1]
            * sum(
                alpha[t][k] * self.extreme_points_power.get((0, k))
                for k in range(self.num_extreme_points)
            )
            + float(electricity_price)
            * (
                sum(
                    alpha[t][k] * self.extreme_points_power.get((0, k))
                    for k in range(self.num_extreme_points)
                )
            )
        )

    def solve_chpd_model(self, heat_demand, electricity_price):
        m = Model("CHPED")
        # variable connected to CHP
        (alpha) = {}
        for t in range(TIME_HORIZON):
            for k in range(self.num_extreme_points):
                alpha[t, k] = m.addVar(
                    lb=0, ub=1, vtype="C", name="alpha(%s,%s)" % (t, k)
                )

        # defining constraints
        for t in range(TIME_HORIZON):
            m.addCons(
                quicksum(alpha[t, k] for k in range(self.num_extreme_points)) == 1,
                "alpha_sum_constraint(%s)" % (t),
            )
        for t in range(TIME_HORIZON):
            m.addCons(
                quicksum(
                    alpha[t, k] * self.extreme_points_heat.get((0, k))
                    for k in range(self.num_extreme_points)
                )
                == heat_demand[t],
                "heat_demand(%s)" % (t),
            )
        objvar = m.addVar(name="objvar", vtype="C", lb=None, ub=None)
        m.setObjective(objvar, "minimize")
        m.addCons(
            objvar
            >= (
                quicksum(
                    a[0]
                    * quicksum(
                        alpha[t, k] * self.extreme_points_heat.get((0, k))
                        for k in range(self.num_extreme_points)
                    )
                    + a[1]
                    * quicksum(
                        alpha[t, k] * self.extreme_points_power.get((0, k))
                        for k in range(self.num_extreme_points)
                    )
                    for t in range(TIME_HORIZON)
                )
                - quicksum(
                    float(electricity_price[t])
                    * (
                        quicksum(
                            alpha[t, k] * self.extreme_points_power.get((0, k))
                            for k in range(self.num_extreme_points)
                        )
                    )
                    for t in range(TIME_HORIZON)
                )
            ),
            name="objconst",
        )
        m.optimize()
        return m

    def get_alpha(self, m) -> list:
        alpha = []
        for v in m.getVars():
            if "alpha" in v.name:
                alpha.append(m.getVal(v))
        alpha = np.array(alpha).reshape(TIME_HORIZON, self.num_extreme_points)
        return alpha
