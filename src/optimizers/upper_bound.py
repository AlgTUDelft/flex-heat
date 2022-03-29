import os
import numpy as np
import pandas as pd
import pickle

from datetime import datetime
from pathlib import *
from pyscipopt import Model, exp, quicksum
from util.util import *


class UpperBound:
    """
    Upper bound cost calculation.
    """

    def __init__(self, upper_lower_bound, pipe_len):
        self.data_p = Path(__file__).parents[2] / "data"
        self.store_p = Path(__file__).parents[2] / "results/upper_bound_trial"
        self.heat_demand = np.array(pd.read_csv(self.data_p / "heat_demand_test.csv"))
        self.electricity_price = np.array(
            pd.read_csv(self.data_p / "day_ahead_electricity_price_test.csv")
        )
        self.number_of_episodes: int = len(self.heat_demand)
        self.now = datetime.now().strftime("%H_%M_%S")
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
        self.upper_lower_bound = upper_lower_bound
        self.pipe_len = pipe_len

    def calculate_upper_bound_cost(self):
        """
        Calculate cost on the upper bound.
        """
        obj_fun_up_sum = []
        for i in range(self.number_of_episodes):
            obj_fun_up = []
            heat_demand_episode = self.heat_demand[i]
            electricity_price_episode = self.electricity_price[i]
            for t in range(TIME_HORIZON):
                obj_fun_time_step = []
                for j in range(self.upper_lower_bound):
                    if t + j <= TIME_HORIZON - 1:
                        k = t + j
                    else:
                        k = TIME_HORIZON - 1
                    m = self.solve_chpd_model(
                        heat_demand_episode[t], electricity_price_episode[k]
                    )
                    alpha = UpperBound.get_alpha(m)
                    obj_fun_time_step.append(
                        self.obj_fun(alpha, electricity_price_episode[k])
                    )
                obj_fun_up.append(max(obj_fun_time_step))
                save_to_pickle(
                    data_path_store=self.store_p,
                    variable=obj_fun_up,
                    variable_name="upper_bound",
                    pipe_len=self.pipe_len,
                    now=self.now,
                    ep=i + 1,
                )
            obj_fun_up_sum.append(sum(obj_fun_up))
            save_to_pickle(
                data_path_store=self.store_p,
                variable=obj_fun_up_sum,
                variable_name="upper_bound_sum",
                pipe_len=self.pipe_len,
                now=self.now,
            )
            print(obj_fun_up_sum)

    def solve_chpd_model(self, heat_demand, electricity_price):
        """
        Solve combined heat and power economic dispatch model.
        """
        m = Model("CHPED")
        # variable connected to CHP
        (alpha) = {}
        for k in range(len(self.keys)):
            alpha[k] = m.addVar(lb=0, ub=1, vtype="C", name="alpha(%s)" % (k))

        # defining constraints
        m.addCons(
            quicksum(alpha[k] for k in range(self.num_extreme_points)) == 1,
            "alpha_sum_constraint",
        )
        m.addCons(
            quicksum(
                alpha[k] * self.extreme_points_heat.get((0, k))
                for k in range(self.num_extreme_points)
            )
            == heat_demand,
            "heat_demand",
        )
        objvar = m.addVar(name="objvar", vtype="C", lb=None, ub=None)
        m.setObjective(objvar, "minimize")
        m.addCons(
            objvar
            >= (
                a[0]
                * quicksum(
                    alpha[k] * self.extreme_points_heat.get((0, k))
                    for k in range(self.num_extreme_points)
                )
                + a[1]
                * quicksum(
                    alpha[k] * self.extreme_points_power.get((0, k))
                    for k in range(self.num_extreme_points)
                )
                - float(electricity_price)
                * (
                    quicksum(
                        alpha[k] * self.extreme_points_power.get((0, k))
                        for k in range(self.num_extreme_points)
                    )
                )
            ),
            name="objconst",
        )
        m.optimize()
        return m

    def obj_fun(self, alpha, electricity_price):
        """
        Calculate objective function.
        """
        return (
            -a[0]
            * sum(
                alpha[k] * self.extreme_points_heat.get((0, k))
                for k in range(self.num_extreme_points)
            )
            - a[1]
            * sum(
                alpha[k] * self.extreme_points_power.get((0, k))
                for k in range(self.num_extreme_points)
            )
            + float(electricity_price)
            * sum(
                alpha[k] * self.extreme_points_power.get((0, k))
                for k in range(self.num_extreme_points)
            )
        )

    @staticmethod
    def get_alpha(m):
        """
        Get alpha parameters of the model.
        """
        alpha = []
        for v in m.getVars():
            if "alpha" in v.name:
                alpha.append(m.getVal(v))
        return alpha
