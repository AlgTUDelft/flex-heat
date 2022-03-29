import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import *
from pyscipopt import Model, exp, quicksum
from util.util import *
from ..simulator.interfaces.keypts import Sim_keypts


class Li_2016(ABC):
    def __init__(
        self,
        max_iter,
        time_limit,
        pipe_len,
        pipe_diameter,
        time_interval,
        max_flow_speed,
        min_flow_speed,
        max_mass_flow,
        min_mass_flow,
        max_supply_temp,
        min_supply_temp,
        max_return_temp,
        min_return_temp,
        p_hes,
        water_pump_efficiency,
        max_power_consumption_water_pump,
        min_power_consumption_water_pump,
    ):
        self.data_p = Path(__file__).parents[2] / "data"
        self.max_iter = max_iter
        self.time_limit = time_limit
        self.heat_demand = np.array(pd.read_csv(self.data_p / "heat_demand_test.csv"))
        self.electricity_price = np.array(
            pd.read_csv(self.data_p / "day_ahead_electricity_price_test.csv")
        )
        self.number_of_episodes = len(self.heat_demand)
        self.time_interval = time_interval
        self.now = datetime.now().strftime("%H_%M_%S")
        # number of elements
        self.i_hs = 1  # number of heat stations
        self.i_chp = self.i_hs  # number of CHP units
        self.i_nd = 2  # number of nodes
        self.i_hes = 1  # number of heat exchanger stations
        self.i_pipe = 1  # number of pipes

        # connections
        self.s_hs = {0: [0]}  # set of HS connected to node (key-node, value-HS)
        self.Nd_hs = {
            0: 0
        }  # set of indices of nodes connected to HS (key-HS, value-node)
        self.s_hes = {1: [0]}  # set of HES connected to node (key-node, value-HES)
        self.Nd_hes = {0: 1}  # index of node connected to HES (key-HES, value-node)
        self.Nd_pf = {0: 0}  # index of starting node of pipeline (key-pipe, value-node)
        self.Nd_pt = {0: 1}  # index of ending node of pipeline (key-pipe, value-node)
        self.s_pipe_supply_in = {
            0: [0]
        }  # set of indices of pipelines starting at the certain node of supply network (key-node, pipe-value)
        self.s_pipe_supply_out = {
            1: [0]
        }  # set of indices of pipelines ending at the certain node of supply network (key-node, pipe-value)
        self.s_pipe_return_in = {
            1: [0]
        }  # set of indices of pipelines starting at the certain node of return network (key-node, pipe-value)
        self.s_pipe_return_out = {
            0: [0]
        }  # set of indices of pipelines ending at the certain node of return network (key-node, pipe-value)
        self.a = np.array([a])
        self.pipe_len = np.array([pipe_len])  # [m]
        self.cross_sectional_area = np.array([pipe_diameter])  # [m]
        self.cross_sectional_area_surface = np.array(
            [pow(self.cross_sectional_area[0], 2) * PI / 4]
        )
        self.max_flow_speed = max_flow_speed  # [m/s]
        self.max_flow_rate_limit = max_mass_flow
        self.min_flow_rate_limit = min_mass_flow
        self.max_node_t_supply_network = {
            0: max_supply_temp,
            1: max_supply_temp,
        }  # max supply T of the node connected to heat station 1
        self.min_node_t_supply_network = {
            0: min_supply_temp,
            1: min_supply_temp,
        }  # min supply T of the node connected to heat station 1
        self.p_hes = {
            0: p_hes
        }  # minimum heat load pressure of a certain heat exchanger station
        self.max_node_t_return_network = {
            0: max_return_temp,
            1: max_return_temp,
        }  # max return T of the node
        self.min_node_t_return_network = {
            0: min_return_temp,
            1: min_return_temp,
        }  # min return T of the node
        self.water_pump_efficiency = {0: water_pump_efficiency}
        self.max_power_consumption_water_pump = {0: max_power_consumption_water_pump}
        self.min_power_consumption_water_pump = {0: min_power_consumption_water_pump}

        self.coefficient_of_pressure_loss = {
            0: (8 * FRICTION_COEFFICIENT * self.pipe_len[0])
            / (WATER_DENSITY * pow(self.cross_sectional_area[0], 5) * pow(PI, 2))
        }  # [1/(kg*m)]

        # CHP operation region
        self.keys = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
        ]  # One heat station, in case of a two heat station (1,0),...(1,3)
        self.extreme_points_power = dict(
            [
                (self.keys[0], 10),
                (self.keys[1], 5),
                (self.keys[2], 35),
                (self.keys[3], 50),
            ]
        )
        self.extreme_points_heat = dict(
            [
                (self.keys[0], 0),
                (self.keys[1], 10),
                (self.keys[2], 70),
                (self.keys[3], 0),
            ]
        )

        # ambient temperature
        self.tau_am = np.array([T_env] * TIME_HORIZON)

        # inequality buffer
        self.delta_temperature = 0.01
        self.delta_mass_flow = 0.01
        self.delta_pressure = 0.01
        self.delta_heat_demand = 0.5

        # parameters to be saved
        self.alpha: list = []
        self.d_pump: list = []
        self.ms_pipe: list = []
        self.mr_pipe: list = []
        self.m_hs: list = []
        self.m_hes: list = []
        self.p_ns: list = []
        self.p_nr: list = []
        self.tau_ns: list = []
        self.tau_nr: list = []
        self.tau_PS_in: list = []
        self.tau_PS_out: list = []
        self.tau_PR_in: list = []
        self.tau_PR_out: list = []
        self.tau_PS_no_out: list = []
        self.tau_PR_no_out: list = []

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def solve_chpd_model(self):
        pass

    @abstractmethod
    def update_complicating_variables(self):
        pass

    def reset_param(self):
        self.alpha: list = []
        self.d_pump: list = []
        self.ms_pipe: list = []
        self.mr_pipe: list = []
        self.m_hs: list = []
        self.m_hes: list = []
        self.p_ns: list = []
        self.p_nr: list = []
        self.tau_ns: list = []
        self.tau_nr: list = []
        self.tau_PS_in: list = []
        self.tau_PS_out: list = []
        self.tau_PR_in: list = []
        self.tau_PR_out: list = []
        self.tau_PS_no_out: list = []
        self.tau_PR_no_out: list = []

    def process_param(self, m):
        for v in m.getVars():
            if "alpha" in v.name:
                self.alpha.append(m.getVal(v))
            elif "d_pump" in v.name:
                self.d_pump.append(m.getVal(v))
            elif "m_hs" in v.name:
                self.m_hs.append(m.getVal(v))
            elif "m_hes" in v.name:
                self.m_hes.append(m.getVal(v))
            elif "ms_pipe" in v.name:
                self.ms_pipe.append(m.getVal(v))
            elif "mr_pipe" in v.name:
                self.mr_pipe.append(m.getVal(v))
            elif "tau_ns" in v.name:
                self.tau_ns.append(m.getVal(v))
            elif "tau_nr" in v.name:
                self.tau_nr.append(m.getVal(v))
            elif "tau_PS_in" in v.name:
                self.tau_PS_in.append(m.getVal(v))
            elif "tau_PS_out" in v.name:
                self.tau_PS_out.append(m.getVal(v))
            elif "tau_PR_in" in v.name:
                self.tau_PR_in.append(m.getVal(v))
            elif "tau_PR_out" in v.name:
                self.tau_PR_out.append(m.getVal(v))
            elif "tau_PS_no_out" in v.name:
                self.tau_PS_no_out.append(m.getVal(v))
            elif "tau_PR_no_out" in v.name:
                self.tau_PR_no_out.append(m.getVal(v))
            elif "p_ns" in v.name:
                self.p_ns.append(m.getVal(v))
            elif "p_nr" in v.name:
                self.p_nr.append(m.getVal(v))

            self.alpha = np.array(self.alpha).reshape(TIME_HORIZON, len(self.keys))
            self.d_pump = np.array(self.d_pump).flatten()
            self.m_hs = np.array(self.m_hs).flatten()
            self.m_hes = np.array(self.m_hes).flatten()
            self.ms_pipe = np.array(self.ms_pipe).flatten()
            self.mr_pipe = np.array(self.mr_pipe).flatten()
            self.tau_ns = np.array(self.tau_ns).flatten()
            self.tau_nr = np.array(self.tau_nr).flatten()
            self.tau_PS_in = np.array(self.tau_PS_in).flatten()
            self.tau_PS_out = np.array(self.tau_PS_out).flatten()
            self.tau_PR_in = np.array(self.tau_PR_in).flatten()
            self.tau_PR_out = np.array(self.tau_PR_out).flatten()
            self.tau_PS_no_out = np.array(self.tau_PS_no_out).flatten()
            self.tau_PR_no_out = np.array(self.tau_PR_no_out).flatten()
            self.p_ns = np.array(self.p_ns).flatten()
            self.p_nr = np.array(self.p_nr).flatten()

    def mass_flow_pipe_in(self, m, i, t, S_PIPE_in):
        if i in S_PIPE_in.keys():
            return quicksum(
                m[S_PIPE_in.get(i)[j], t] for j in range(len(S_PIPE_in.get(i)))
            )
        else:
            return 0

    def mass_flow_pipe_out(self, m, i, t, S_PIPE_out):
        if i in S_PIPE_out.keys():
            return quicksum(
                m[S_PIPE_out.get(i)[j], t] for j in range(len(S_PIPE_out.get(i)))
            )
        else:
            return 0

    def mass_flow_hs(self, m_hs, i, t):
        if i in self.s_hs.keys():
            return quicksum(
                m_hs[self.s_hs.get(i)[j], t] for j in range(len(self.s_hs.get(i)))
            )
        else:
            return 0

    def mass_flow_hes(self, m_hes, i, t):
        if i in self.s_hes.keys():
            return quicksum(
                m_hes[self.s_hes.get(i)[j], t] for j in range(len(self.s_hes.get(i)))
            )
        else:
            return 0

    def temp_mixing_outlet(self, tau_out, m_pipe, i, t, S_PIPE_out):
        if i in S_PIPE_out.keys():
            return quicksum(
                tau_out[S_PIPE_out.get(i)[j], t] * m_pipe[S_PIPE_out.get(i)[j], t]
                for j in range(len(S_PIPE_out.get(i)))
            )
        else:
            return 0

    def temp_mixing_inlet(self, m, tau_in, tau, S_PIPE_in, i, t, name):
        if i in S_PIPE_in.keys():
            for j in range(len(S_PIPE_in.get(i))):
                m.addCons(
                    tau_in[S_PIPE_in.get(i)[j], t] - tau[i, t]
                    <= self.delta_temperature,
                    name="L " + name % (i, S_PIPE_in.get(i)[j], t),
                )
                m.addCons(
                    tau_in[S_PIPE_in.get(i)[j], t] - tau[i, t]
                    >= -self.delta_temperature,
                    name="G " + name % (i, S_PIPE_in.get(i)[j], t),
                )

    def C_chp(self, alpha, i, t):
        return self.a[i, 1] * quicksum(
            alpha[i, t, k] * self.extreme_points_power.get((i, k))
            for k in range(len(self.keys))
        ) + self.a[i, 0] * quicksum(
            alpha[i, t, k] * self.extreme_points_heat.get((i, k))
            for k in range(len(self.keys))
        )

    def electricity_sell(self, alpha, t, d_pump):
        return quicksum(
            (
                quicksum(
                    alpha[i, t, k] * self.extreme_points_heat.get((i, k))
                    for k in range(len(self.keys))
                )
                # - d_pump[i, t]
            )
            for i in range(self.i_chp)
        )

    def delivered_electricity_fun(self, t, alpha, d_pump):
        return (
            sum(
                alpha[t][k] * self.extreme_points_power.get((0, k))
                for k in range(len(self.keys))
            )
            # - d_pump
        )

    def obj_fun(self, t, alpha, day_ahead_electricity_price_episode, d_pump):
        return (
            -self.a[0, 0]
            * sum(
                alpha[t][k] * self.extreme_points_heat.get((0, k))
                for k in range(len(self.keys))
            )
            - self.a[0, 1]
            * sum(
                alpha[t][k] * self.extreme_points_power.get((0, k))
                for k in range(len(self.keys))
            )
            + float(day_ahead_electricity_price_episode[t])
            * (self.delivered_electricity_fun(t, alpha, d_pump[t]))
        )


class Li_2016_day_ahead(Li_2016):
    def __init__(
        self,
        max_iter,
        time_limit,
        pipe_len,
        pipe_diameter,
        time_interval,
        max_flow_speed,
        min_flow_speed,
        max_mass_flow,
        min_mass_flow,
        max_supply_temp,
        min_supply_temp,
        max_return_temp,
        min_return_temp,
        p_hes,
        water_pump_efficiency,
        max_power_consumption_water_pump,
        min_power_consumption_water_pump,
    ):
        super().__init__(
            max_iter=max_iter,
            time_limit=time_limit,
            pipe_len=pipe_len,
            pipe_diameter=pipe_diameter,
            time_interval=time_interval,
            max_flow_speed=max_flow_speed,
            min_flow_speed=min_flow_speed,
            max_mass_flow=max_mass_flow,
            min_mass_flow=min_mass_flow,
            max_supply_temp=max_supply_temp,
            min_supply_temp=min_supply_temp,
            max_return_temp=max_return_temp,
            min_return_temp=min_return_temp,
            p_hes=p_hes,
            water_pump_efficiency=water_pump_efficiency,
            max_power_consumption_water_pump=max_power_consumption_water_pump,
            min_power_consumption_water_pump=min_power_consumption_water_pump,
        )
        self.result_p = self.data_p / "results/li_2016/li_2016_day_ahead"

    def run(self):
        objective_function, heat_delivered, electricity_delivered = (
            [],
            [],
            [],
        )
        for i in range(1, self.number_of_episodes):
            (
                objective_function_episode,
                heat_delivered_episode,
                electricity_delivered_episode,
            ) = ([], [], [])
            ITER_COUNT = 0
            self.reset_param()
            TIME_DELAY_I = np.zeros(
                (self.i_pipe, TIME_HORIZON), dtype=int
            )  # time delays associating changes in temperature
            TIME_DELAY_II = np.zeros(
                (self.i_pipe, TIME_HORIZON), dtype=int
            )  # time delays associating changes in temperature
            COEFFICIENT_VARIABLE_R = np.full(
                (self.i_pipe, TIME_HORIZON),
                WATER_DENSITY * self.cross_sectional_area_surface[0] * self.pipe_len[0],
            )  # coefficient variables R associated with the historic mass flow
            COEFFICIENT_VARIABLE_S = np.full(
                (self.i_pipe, TIME_HORIZON),
                WATER_DENSITY * self.cross_sectional_area_surface[0] * self.pipe_len[0],
            )  # coefficient variables S associated with the historic mass flow
            complicating_variables = [  # first iteration, initialized
                TIME_DELAY_I,
                TIME_DELAY_II,
                COEFFICIENT_VARIABLE_R,
                COEFFICIENT_VARIABLE_S,
            ]
            while ITER_COUNT < self.max_iter:
                ITER_COUNT += 1
                m = self.solve_chpd_model(
                    complicating_variables=complicating_variables,
                    heat_demand_episode=self.heat_demand[i],
                    electricity_price_episode=self.electricity_price[i],
                )
                complicating_variables = self.update_complicating_variables(
                    m,
                    TIME_DELAY_I,
                    TIME_DELAY_II,
                    COEFFICIENT_VARIABLE_S,
                    COEFFICIENT_VARIABLE_R,
                )
            self.process_param(m)
            for t in range(TIME_HORIZON):
                objective_function_episode.append(
                    self.obj_fun(t, self.alpha, self.electricity_price[i], self.d_pump)
                )
                heat_delivered_episode.append(
                    C * self.ms_pipe[t] * (self.tau_PS_out[t] - self.tau_PR_in[t])
                )
                electricity_delivered_episode.append(
                    self.delivered_electricity_fun(t, self.alpha, self.d_pump[t])
                )
            dict_ = {
                "action": self.alpha,
                "d_pump": self.d_pump,
                "m_hs": self.m_hs,
                "m_hes": self.m_hes,
                "ms_pipe": self.ms_pipe,
                "mr_pipe": self.mr_pipe,
                "tau_ns": self.tau_ns,
                "tau_nr": self.tau_nr,
                "tau_PS_in": self.tau_PS_in,
                "tau_PS_out": self.tau_PS_out,
                "tau_PR_in": self.tau_PR_in,
                "tau_PR_out": self.tau_PR_out,
                "tau_PS_no_out": self.tau_PS_no_out,
                "tau_PR_no_out": self.tau_PR_no_out,
                "p_ns": self.p_ns,
                "p_nr": self.p_nr,
                "objective_function": objective_function_episode,
                "heat_delivered": heat_delivered_episode,
                "electricity_delivered": electricity_delivered_episode,
                "time_delay_I": TIME_DELAY_I,
                "time_delay_II": TIME_DELAY_II,
                "coefficient_variable_S": COEFFICIENT_VARIABLE_S,
                "coefficient_variable_R": COEFFICIENT_VARIABLE_R,
            }
            """
            save_to_pickle(
                data_path_store=self.result_p,
                variable=dict_,
                variable_name="ncl_episode_with_compl_var",
                pipe_len=self.pipe_len,
                ep=i + 1,
                now=self.now,
            )
            """
            objective_function.append(sum(objective_function_episode))
            heat_delivered.append(sum(heat_delivered_episode) / TIME_HORIZON)
            electricity_delivered.append(
                sum(electricity_delivered_episode) / TIME_HORIZON
            )

    def solve_chpd_model(
        self, complicating_variables, heat_demand_episode, electricity_price_episode
    ):
        time_delay_I = complicating_variables[0]
        time_delay_II = complicating_variables[1]
        coeff_R = complicating_variables[2]
        coeff_S = complicating_variables[3]
        m = Model("CHPED")
        m.setRealParam("limits/time", self.time_limit)
        m.setRealParam("numerics/epsilon", pow(10, -17))
        m.setRealParam("numerics/sumepsilon", pow(10, -14))
        m.setBoolParam("lp/presolving", True)
        m.setIntParam("presolving/maxrounds", 0)
        # defining variables
        (
            alpha,
            m_hs,
            tau_ns,
            tau_nr,
            p_ns,
            p_nr,
            m_hes,
            ms_pipe,
            mr_pipe,
            d_pump,
            tau_PS_no_out,
            tau_PR_no_out,
            tau_PS_out,
            tau_PR_out,
            tau_PS_in,
            tau_PR_in,
        ) = ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})

        # variables connected to heat station
        for i in range(self.i_hs):
            for t in range(TIME_HORIZON):
                for k in range(len(self.keys)):
                    alpha[i, t, k] = m.addVar(
                        lb=0, ub=1, vtype="C", name="alpha(%s,%s,%s)" % (i, t, k)
                    )
                m_hs[i, t] = m.addVar(
                    vtype="C",
                    name="m_hs(%s,%s)" % (i, t),
                    lb=0,
                    ub=self.max_flow_rate_limit,
                )
                d_pump[i, t] = m.addVar(
                    lb=self.min_power_consumption_water_pump.get(i),
                    ub=self.max_power_consumption_water_pump.get(i),
                    vtype="C",
                    name="d_pump(%s, %s)" % (i, t),
                )
        # variables connected to network nodes
        for i in range(self.i_nd):
            for t in range(TIME_HORIZON):
                tau_ns[i, t] = m.addVar(
                    vtype="C",
                    name="tau_ns(%s,%s)" % (i, t),
                    lb=self.min_node_t_supply_network.get(i),
                    ub=self.max_node_t_supply_network.get(i),
                )
                tau_nr[i, t] = m.addVar(
                    vtype="C",
                    name="tau_nr(%s,%s)" % (i, t),
                    lb=self.min_node_t_return_network.get(i),
                    ub=self.max_node_t_return_network.get(i),
                )
                p_ns[i, t] = m.addVar(vtype="C", name="p_ns(%s,%s)" % (i, t))
                p_nr[i, t] = m.addVar(vtype="C", name="p_nr(%s, %s)" % (i, t))

        # variables connected to heat exchanger
        for i in range(self.i_hes):
            for t in range(TIME_HORIZON):
                m_hes[i, t] = m.addVar(
                    vtype="C",
                    name="m_hes(%s,%s)" % (i, t),
                    lb=0,
                    ub=self.max_flow_rate_limit,
                )  # mass flow rate of heat exchanger station at period t
        # variables connected to pipes
        for i in range(self.i_pipe):
            for t in range(TIME_HORIZON):
                ms_pipe[i, t] = m.addVar(
                    ub=self.max_flow_rate_limit,
                    lb=self.min_flow_rate_limit,
                    vtype="C",
                    name="ms_pipe(%s,%s)" % (i, t),
                )  # mass flow rate of pipeline in the supply network
                mr_pipe[i, t] = m.addVar(
                    ub=self.max_flow_rate_limit,
                    lb=self.min_flow_rate_limit,
                    vtype="C",
                    name="mr_pipe(%s,%s)" % (i, t),
                )  # mass flow rate of pipeline in the return network
                tau_PS_out[i, t] = m.addVar(
                    vtype="C",
                    name="tau_PS_out(%s,%s)" % (i, t),
                )  # mass flow temperature considering T drop at the outlet of pipeline in supply network
                tau_PR_out[i, t] = m.addVar(
                    vtype="C",
                    name="tau_PR_out(%s,%s)" % (i, t),
                )  # mass flow temperature considering T drop at the outlet of pipeline in return network
                tau_PS_in[i, t] = m.addVar(
                    vtype="C",
                    name="tau_PS_in(%s,%s)" % (i, t),
                )  # mass flow temperature at the inlet of pipeline in supply network
                tau_PR_in[i, t] = m.addVar(
                    vtype="C",
                    name="tau_PR_in(%s,%s)" % (i, t),
                )  # mass flow temperature at the inlet of pipeline in return network
                tau_PS_no_out[i, t] = m.addVar(
                    vtype="C",
                    name="tau_PS_no_out(%s,%s)" % (i, t),
                )  # outlet T without heat loss
                tau_PR_no_out[i, t] = m.addVar(
                    vtype="C",
                    name="tau_PR_no_out(%s,%s)" % (i, t),
                )  # outlet T without heat loss

        # defining constraints
        # CHP unit
        for i in range(self.i_chp):
            for t in range(TIME_HORIZON):
                m.addCons(
                    quicksum(alpha[i, t, k] for k in range(len(self.keys))) == 1,
                    "alpha_sum_constraint(%s, %s)" % (i, t),
                )
                m.addCons(
                    d_pump[i, t] * MW_W
                    == m_hs[i, t]
                    * (p_ns[self.Nd_hs.get(i), t] - p_nr[self.Nd_hs.get(i), t])
                    / (self.water_pump_efficiency.get(i) * WATER_DENSITY),
                )
                m.addCons(
                    quicksum(
                        alpha[i, t, k] * self.extreme_points_heat.get((i, k))
                        for k in range(len(self.keys))
                    )
                    == C
                    * m_hs[i, t]
                    * (tau_ns[self.Nd_hs.get(i), t] - tau_nr[self.Nd_hs.get(i), t]),
                    name="heat_output_CHP_unit_constraint_G(%s, %s)" % (i, t),
                )

        # Heat exchanger station
        for i in range(self.i_hes):
            for t in range(TIME_HORIZON):
                m.addCons(
                    C
                    * m_hes[i, t]
                    * (tau_ns[self.Nd_hes.get(i), t] - tau_nr[self.Nd_hes.get(i), t])
                    - heat_demand_episode[t]
                    >= -self.delta_heat_demand,
                    name="heat_exchangers_heat_loads_TDHS_G(%s, %s)" % (i, t),
                )
                m.addCons(
                    C
                    * m_hes[i, t]
                    * (tau_ns[self.Nd_hes.get(i), t] - tau_nr[self.Nd_hes.get(i), t])
                    - heat_demand_episode[t]
                    <= self.delta_heat_demand,
                    name="heat_exchangers_heat_loads_TDHS_L(%s, %s)" % (i, t),
                )
                m.addCons(
                    p_ns[self.Nd_hes.get(i), t] - p_nr[self.Nd_hes.get(i), t]
                    >= self.p_hes.get(i),
                    name="minimum_heat_load_pressure(%s, %s)" % (i, t),
                )

        # District heating network
        for i in range(self.i_nd):
            for t in range(TIME_HORIZON):
                m.addCons(
                    self.mass_flow_pipe_in(ms_pipe, i, t, self.s_pipe_supply_in)
                    - self.mass_flow_pipe_out(ms_pipe, i, t, self.s_pipe_supply_out)
                    - self.mass_flow_hs(m_hs, i, t)
                    + self.mass_flow_hes(m_hes, i, t)
                    <= self.delta_mass_flow,
                    name="continuity_of_supply_network_mass_flow_L(%s, %s)" % (i, t),
                )
                m.addCons(
                    self.mass_flow_pipe_in(ms_pipe, i, t, self.s_pipe_supply_in)
                    - self.mass_flow_pipe_out(ms_pipe, i, t, self.s_pipe_supply_out)
                    - self.mass_flow_hs(m_hs, i, t)
                    + self.mass_flow_hes(m_hes, i, t)
                    >= -self.delta_mass_flow,
                    name="continuity_of_supply_network_mass_flow_G(%s, %s)" % (i, t),
                )

                m.addCons(
                    self.mass_flow_pipe_in(mr_pipe, i, t, self.s_pipe_return_in)
                    - self.mass_flow_pipe_out(mr_pipe, i, t, self.s_pipe_return_out)
                    - self.mass_flow_hes(m_hes, i, t)
                    + self.mass_flow_hs(m_hs, i, t)
                    <= self.delta_mass_flow,
                    name="continuity_of_return_network_mass_flow_L(%s, %s)" % (i, t),
                )
                m.addCons(
                    self.mass_flow_pipe_in(mr_pipe, i, t, self.s_pipe_return_in)
                    - self.mass_flow_pipe_out(mr_pipe, i, t, self.s_pipe_return_out)
                    - self.mass_flow_hes(m_hes, i, t)
                    + self.mass_flow_hs(m_hs, i, t)
                    >= -self.delta_mass_flow,
                    name="continuity_of_return_network_mass_flow_G(%s, %s)" % (i, t),
                )
                m.addCons(
                    self.temp_mixing_outlet(
                        tau_PS_out, ms_pipe, i, t, self.s_pipe_supply_out
                    )
                    - tau_ns[i, t]
                    * self.mass_flow_pipe_out(ms_pipe, i, t, self.s_pipe_supply_out)
                    <= self.delta_mass_flow,
                    name="temperature_mixing_outlet_supply_network_L(%s, %s)" % (i, t),
                )
                m.addCons(
                    self.temp_mixing_outlet(
                        tau_PS_out, ms_pipe, i, t, self.s_pipe_supply_out
                    )
                    - tau_ns[i, t]
                    * self.mass_flow_pipe_out(ms_pipe, i, t, self.s_pipe_supply_out)
                    >= -self.delta_mass_flow,
                    name="temperature_mixing_outlet_supply_network_G(%s, %s)" % (i, t),
                )
                m.addCons(
                    self.temp_mixing_outlet(
                        tau_PR_out, mr_pipe, i, t, self.s_pipe_return_out
                    )
                    - tau_nr[i, t]
                    * self.mass_flow_pipe_out(mr_pipe, i, t, self.s_pipe_return_out)
                    <= self.delta_mass_flow,
                    name="temperature_mixing_outlet_return_network_L(%s, %s)" % (i, t),
                )
                m.addCons(
                    self.temp_mixing_outlet(
                        tau_PR_out, mr_pipe, i, t, self.s_pipe_return_out
                    )
                    - tau_nr[i, t]
                    * self.mass_flow_pipe_out(mr_pipe, i, t, self.s_pipe_return_out)
                    >= -self.delta_mass_flow,
                    name="temperature_mixing_outlet_return_network_G(%s, %s)" % (i, t),
                )
                self.temp_mixing_inlet(
                    m,
                    tau_PS_in,
                    tau_ns,
                    self.s_pipe_supply_in,
                    i,
                    t,
                    "temperature_mixing_inlet_supply_network(%s, %s, %s)",
                )
                self.temp_mixing_inlet(
                    m,
                    tau_PR_in,
                    tau_nr,
                    self.s_pipe_return_in,
                    i,
                    t,
                    "temperature_mixing_inlet_return_network(%s, %s, %s)",
                )
        for b in range(self.i_pipe):
            for t in range(TIME_HORIZON):
                m.addCons(
                    p_ns[self.Nd_pf.get(b), t]
                    - p_ns[self.Nd_pt.get(b), t]
                    - self.coefficient_of_pressure_loss.get(b)
                    * ms_pipe[b, t]
                    * ms_pipe[b, t]
                    <= self.delta_pressure,
                    name="pressure_loss_supply_net_L (%s, %s)" % (b, t),
                )
                m.addCons(
                    p_ns[self.Nd_pf.get(b), t]
                    - p_ns[self.Nd_pt.get(b), t]
                    - self.coefficient_of_pressure_loss.get(b)
                    * ms_pipe[b, t]
                    * ms_pipe[b, t]
                    >= -self.delta_pressure,
                    name="pressure_loss_supply_net_G (%s, %s)" % (b, t),
                )
                m.addCons(
                    p_nr[self.Nd_pt.get(b), t]
                    - p_nr[self.Nd_pf.get(b), t]
                    - self.coefficient_of_pressure_loss.get(b)
                    * mr_pipe[b, t]
                    * mr_pipe[b, t]
                    <= self.delta_pressure,
                    name="pressure_loss_return_net_L (%s, %s)" % (b, t),
                )
                m.addCons(
                    p_nr[self.Nd_pt.get(b), t]
                    - p_nr[self.Nd_pf.get(b), t]
                    - self.coefficient_of_pressure_loss.get(b)
                    * mr_pipe[b, t]
                    * mr_pipe[b, t]
                    >= -self.delta_pressure,
                    name="pressure_loss_return_net_G (%s, %s)" % (b, t),
                )
                m.addCons(
                    (
                        tau_PS_no_out[b, t]
                        - (
                            (
                                coeff_R[b, t]
                                - WATER_DENSITY
                                * self.cross_sectional_area_surface[b]
                                * self.pipe_len[b]
                            )
                            * tau_PS_in[b, t - time_delay_II[b, t]]
                            + quicksum(
                                ms_pipe[b, k] * HOUR_TO_SEC * tau_PS_in[b, k]
                                for k in range(
                                    t - time_delay_I[b, t] + self.time_interval,
                                    t - time_delay_II[b, t],
                                    self.time_interval,
                                )
                            )
                            + (
                                ms_pipe[b, t] * HOUR_TO_SEC
                                + WATER_DENSITY
                                * self.cross_sectional_area_surface[b]
                                * self.pipe_len[b]
                                - coeff_S[b, t]
                            )
                            * tau_PS_in[b, t - time_delay_I[b, t]]
                        )
                        / (ms_pipe[b, t] * HOUR_TO_SEC)
                    )
                    <= self.delta_temperature,
                    name="outlet_supply_T_without_heat_loss_L (%s, %s)" % (b, t),
                )
                m.addCons(
                    (
                        tau_PS_no_out[b, t]
                        - (
                            (
                                coeff_R[b, t]
                                - WATER_DENSITY
                                * self.cross_sectional_area_surface[b]
                                * self.pipe_len[b]
                            )
                            * tau_PS_in[b, t - time_delay_II[b, t]]
                            + quicksum(
                                ms_pipe[b, k] * HOUR_TO_SEC * tau_PS_in[b, k]
                                for k in range(
                                    t - time_delay_I[b, t] + self.time_interval,
                                    t - time_delay_II[b, t],
                                    self.time_interval,
                                )
                            )
                            + (
                                ms_pipe[b, t] * HOUR_TO_SEC
                                + WATER_DENSITY
                                * self.cross_sectional_area_surface[b]
                                * self.pipe_len[b]
                                - coeff_S[b, t]
                            )
                            * tau_PS_in[b, t - time_delay_I[b, t]]
                        )
                        / (ms_pipe[b, t] * HOUR_TO_SEC)
                    )
                    >= -self.delta_temperature,
                    name="outlet_supply_T_without_heat_loss_G (%s, %s)" % (b, t),
                )

                m.addCons(
                    (
                        tau_PR_no_out[b, t]
                        - (
                            (
                                coeff_R[b, t]
                                - WATER_DENSITY
                                * self.cross_sectional_area_surface[b]
                                * self.pipe_len[b]
                            )
                            * tau_PR_in[b, t - time_delay_II[b, t]]
                            + quicksum(
                                ms_pipe[b, k] * HOUR_TO_SEC * tau_PR_in[b, k]
                                for k in range(
                                    t - time_delay_I[b, t] + self.time_interval,
                                    t - time_delay_II[b, t],
                                    self.time_interval,
                                )
                            )
                            + (
                                ms_pipe[b, t] * HOUR_TO_SEC
                                + WATER_DENSITY
                                * self.cross_sectional_area_surface[b]
                                * self.pipe_len[b]
                                - coeff_S[b, t]
                            )
                            * tau_PR_in[b, t - time_delay_I[b, t]]
                        )
                        / (ms_pipe[b, t] * HOUR_TO_SEC)
                    )
                    <= self.delta_temperature,
                    name="outlet_return_T_without_heat_loss_L (%s, %s)" % (b, t),
                )
                m.addCons(
                    (
                        tau_PR_no_out[b, t]
                        - (
                            (
                                coeff_R[b, t]
                                - WATER_DENSITY
                                * self.cross_sectional_area_surface[b]
                                * self.pipe_len[b]
                            )
                            * tau_PR_in[b, t - time_delay_II[b, t]]
                            + quicksum(
                                ms_pipe[b, k] * HOUR_TO_SEC * tau_PR_in[b, k]
                                for k in range(
                                    t - time_delay_I[b, t] + self.time_interval,
                                    t - time_delay_II[b, t],
                                    self.time_interval,
                                )
                            )
                            + (
                                ms_pipe[b, t] * HOUR_TO_SEC
                                + WATER_DENSITY
                                * self.cross_sectional_area_surface[b]
                                * self.pipe_len[b]
                                - coeff_S[b, t]
                            )
                            * tau_PR_in[b, t - time_delay_I[b, t]]
                        )
                        / (ms_pipe[b, t] * HOUR_TO_SEC)
                    )
                    >= -self.delta_temperature,
                    name="outlet_return_T_without_heat_loss_G (%s, %s)" % (b, t),
                )
                """
                m.addCons(
                    tau_PS_out[b, t]
                    - tau_am[t]
                    - (tau_PS_no_out[b, t] - tau_am[t])
                    * exp(
                        -(HEAT_TRANSFER_COEFF[b] * HOUR_TO_SEC)
                        / (CROSS_SECTIONAL_AREA_SURFACE[b] * WATER_DENSITY * C)
                        * (
                            time_delay_II[b, t]
                            + 0.5
                            + (coeff_S[b, t] - coeff_R[b, t])
                            / (ms_pipe[b, t - time_delay_II[b, t]] * HOUR_TO_SEC)
                        )
                    )
                    == 0,
                    name="outlet_supply_T_with_heat_loss (%s, %s)" % (b, t),
                )
                m.addCons(
                    tau_PR_out[b, t]
                    - tau_am[t]
                    - (tau_PR_no_out[b, t] - tau_am[t])
                    * exp(
                        -(HEAT_TRANSFER_COEFF[b] * HOUR_TO_SEC)
                        / (CROSS_SECTIONAL_AREA_SURFACE[b] * WATER_DENSITY * C)
                        * (
                            time_delay_II[b, t]
                            + 0.5
                            + (coeff_S[b, t] - coeff_R[b, t])
                            / (mr_pipe[b, t - time_delay_II[b, t]] * HOUR_TO_SEC)
                        )
                    )
                    == 0,
                    name="outlet_return_T_with_heat_loss_L (%s, %s)" % (b, t),
                )
                """
                m.addCons(
                    tau_PS_out[b, t]
                    - self.tau_am[t]
                    - (tau_PS_no_out[b, t] - self.tau_am[t])
                    * exp(
                        -(HEAT_TRANSFER_COEFF[b] * self.pipe_len[b])
                        / (ms_pipe[b, t] * C)
                    )
                    == 0,
                    name="outlet_supply_T_with_heat_loss (%s, %s)" % (b, t),
                )
                m.addCons(
                    tau_PR_out[b, t]
                    - self.tau_am[t]
                    - (tau_PR_no_out[b, t] - self.tau_am[t])
                    * exp(
                        -(HEAT_TRANSFER_COEFF[b] * self.pipe_len[b])
                        / (mr_pipe[b, t] * C)
                    )
                    == 0,
                    name="outlet_return_T_with_heat_loss (%s, %s)" % (b, t),
                )
        objvar = m.addVar(name="objvar", vtype="C", lb=None, ub=None)
        m.setObjective(objvar, "minimize")
        m.addCons(
            objvar
            >= (
                quicksum(
                    quicksum(self.C_chp(alpha, i, t) for t in range(TIME_HORIZON))
                    for i in range(self.i_chp)
                )
                - quicksum(
                    float(electricity_price_episode[t])
                    * self.electricity_sell(alpha, t, d_pump)
                    for t in range(TIME_HORIZON)
                )
            ),
            name="objconst",
        )
        m.data = ms_pipe
        m.optimize()
        return m

    def update_complicating_variables(
        self,
        m,
        TIME_DELAY_I,
        TIME_DELAY_II,
        COEFFICIENT_VARIABLE_S,
        COEFFICIENT_VARIABLE_R,
    ):
        ms_pipe_sol = m.data
        ms_pipe = np.empty((self.i_pipe, TIME_HORIZON), dtype=float)
        for key, value in ms_pipe_sol.items():
            ms_pipe[list(key)[0], list(key)[1]] = m.getVal(value)

        for b in range(self.i_pipe):
            pipe_vol = (
                WATER_DENSITY * self.cross_sectional_area_surface[b] * self.pipe_len[b]
            )
            for t in range(TIME_HORIZON):
                k, cumsum = 0, ms_pipe[b, t] * HOUR_TO_SEC
                while cumsum < pipe_vol and k < t:
                    k += 1
                    cumsum += ms_pipe[b, t - k] * HOUR_TO_SEC
                TIME_DELAY_II[b, t] = k
                cumsum -= ms_pipe[b, t] * HOUR_TO_SEC
                while cumsum < pipe_vol and k < t:
                    k += 1
                    cumsum += ms_pipe[b, t - k] * HOUR_TO_SEC
                TIME_DELAY_I[b, t] = k
                COEFFICIENT_VARIABLE_R[b, t] = np.sum(
                    ms_pipe[b, (t - TIME_DELAY_II[b, t]) : (t + 1)] * HOUR_TO_SEC
                )
                if TIME_DELAY_I[b, t] >= TIME_DELAY_II[b, t] + 1:
                    COEFFICIENT_VARIABLE_S[b, t] = np.sum(
                        ms_pipe[b, (t - TIME_DELAY_I[b, t] + 1) : (t + 1)] * HOUR_TO_SEC
                    )
                else:
                    COEFFICIENT_VARIABLE_S[b, t] = COEFFICIENT_VARIABLE_R[b, t]
        return [
            TIME_DELAY_I,
            TIME_DELAY_II,
            COEFFICIENT_VARIABLE_R,
            COEFFICIENT_VARIABLE_S,
        ]


class Li_2016_intraday(Li_2016):
    def __init__(
        self,
        max_iter,
        time_limit,
        pipe_len,
        pipe_diameter,
        time_interval,
        max_flow_speed,
        min_flow_speed,
        max_mass_flow,
        min_mass_flow,
        max_supply_temp,
        min_supply_temp,
        max_return_temp,
        min_return_temp,
        p_hes,
        water_pump_efficiency,
        max_power_consumption_water_pump,
        min_power_consumption_water_pump,
    ):
        super().__init__(
            max_iter=max_iter,
            time_limit=time_limit,
            pipe_len=pipe_len,
            pipe_diameter=pipe_diameter,
            time_interval=time_interval,
            max_flow_speed=max_flow_speed,
            min_flow_speed=min_flow_speed,
            max_mass_flow=max_mass_flow,
            min_mass_flow=min_mass_flow,
            max_supply_temp=max_supply_temp,
            min_supply_temp=min_supply_temp,
            max_return_temp=max_return_temp,
            min_return_temp=min_return_temp,
            p_hes=p_hes,
            water_pump_efficiency=water_pump_efficiency,
            max_power_consumption_water_pump=max_power_consumption_water_pump,
            min_power_consumption_water_pump=min_power_consumption_water_pump,
        )
        self.result_p = self.data_p / "results/li_2016/li_2016_intraday"
        self.sim = Sim_keypts(
            heat_demand=self.heat_demand[0],
            pipe_len=self.pipe_len[0],
            pipe_diameter=self.cross_sectional_area[0],
            historical_t_supply=historical_t_supply,
            historical_t_return=historical_t_return,
            max_flow_speed=max_flow_speed,
            min_flow_speed=min_flow_speed,
            max_supply_temp=max_supply_temp,
            min_supply_temp=min_supply_temp,
            min_return_temp=min_return_temp,
            control_with_temp=False,
        )

    def run(self):
        objective_function, heat_delivered, electricity_delivered = (
            [],
            [],
            [],
        )
        for i in range(1, self.number_of_episodes):
            (
                objective_function_episode,
                heat_delivered_episode,
                electricity_delivered_episode,
            ) = ([], [], [])
            alpha_episode, d_pump_episode, ms_episode = [], [], []

            # defining intraday and day ahead electricity price
            e_price_dayahead = np.repeat(
                np.mean(self.electricity_price[i]), TIME_HORIZON
            )
            e_price_intraday = self.electricity_price[i]

            TIME_DELAY_I = np.zeros(
                (self.i_pipe, TIME_HORIZON), dtype=int
            )  # time delays associating changes in temperature
            TIME_DELAY_II = np.zeros(
                (self.i_pipe, TIME_HORIZON), dtype=int
            )  # time delays associating changes in temperature
            COEFFICIENT_VARIABLE_R = np.full(
                (self.i_pipe, TIME_HORIZON),
                WATER_DENSITY * self.cross_sectional_area_surface[0] * self.pipe_len[0],
            )  # coefficient variables R associated with the historic mass flow
            COEFFICIENT_VARIABLE_S = np.full(
                (self.i_pipe, TIME_HORIZON),
                WATER_DENSITY * self.cross_sectional_area_surface[0] * self.pipe_len[0],
            )  # coefficient variables S associated with the historic mass flow

            complicating_variables = [  # first iteration, initialized
                TIME_DELAY_I,
                TIME_DELAY_II,
                COEFFICIENT_VARIABLE_R,
                COEFFICIENT_VARIABLE_S,
            ]

            self.sim.update(self.heat_demand[i], e_price_dayahead)
            for t in range(TIME_HORIZON):
                self.reset_param()
                e_price_current = np.copy(e_price_dayahead)
                e_price_current[:t] = e_price_intraday[:t]
                ITER_COUNT = 0

                # updating the simulator
                if t > 0:
                    if np.sum(alpha_episode[-1]) == 0:
                        print(i, t)
                        break
                else:
                    producer_temp_sim = []

                while ITER_COUNT < self.max_iter:
                    ITER_COUNT += 1
                    m = self.solve_chpd_model(
                        complicating_variables,
                        self.heat_demand[i],
                        e_price_current,
                        producer_temp_sim,
                        alpha_episode,
                    )
                    complicating_variables = self.update_complicating_variables(
                        m,
                        TIME_DELAY_I,
                        TIME_DELAY_II,
                        COEFFICIENT_VARIABLE_S,
                        COEFFICIENT_VARIABLE_R,
                        t,
                    )
                self.process_param(m)
                alpha_episode.append(self.alpha[t])
                d_pump_episode.append(self.d_pump[t])
                ms_episode.append(self.ms[t])
            for t in range(TIME_HORIZON):
                objective_function_episode.append(
                    self.obj_fun(
                        t, alpha_episode, self.electricity_price[i], d_pump_episode
                    )
                )
                heat_delivered_episode.append(
                    C * ms_episode[t] * (self.tau_PS_out[t] - self.tau_PR_in[t])
                )
                electricity_delivered_episode.append(
                    self.delivered_electricity_fun(t, alpha_episode, d_pump_episode[t])
                )
            dict_ = {
                "alpha": alpha_episode,
                "d_pump": d_pump_episode,
                "ms": ms_episode,
                "objective_function": objective_function_episode,
                "heat_delivered": heat_delivered_episode,
                "electricity_delivered": electricity_delivered_episode,
            }
            """
            save_to_pickle(
                data_path_store=self.result_p,
                variable=dict_,
                variable_name="data_episode",
                pipe_len=self.pipe_len,
                ep=i + 1,
                now=self.now,
            )
            """
            objective_function.append(sum(objective_function_episode))
            heat_delivered.append(sum(heat_delivered_episode) / TIME_HORIZON)
            electricity_delivered.append(
                sum(electricity_delivered_episode) / TIME_HORIZON
            )

    def solve_chpd_model(
        self,
        complicating_variables,
        heat_demand_episode,
        electricity_price_episode,
        temp_his,
        alpha_his,
    ):
        time_delay_I = complicating_variables[0]
        time_delay_II = complicating_variables[1]
        coeff_R = complicating_variables[2]
        coeff_S = complicating_variables[3]
        m = Model("CHPED")
        m.setRealParam("limits/time", self.time_limit)
        m.setRealParam("numerics/epsilon", pow(10, -17))
        m.setRealParam("numerics/sumepsilon", pow(10, -14))
        m.setBoolParam("lp/presolving", True)
        m.setIntParam("presolving/maxrounds", 0)
        # defining variables
        (
            alpha,
            m_hs,
            tau_ns,
            tau_nr,
            p_ns,
            p_nr,
            m_hes,
            ms_pipe,
            mr_pipe,
            d_pump,
            tau_PS_no_out,
            tau_PR_no_out,
            tau_PS_out,
            tau_PR_out,
            tau_PS_in,
            tau_PR_in,
        ) = ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})

        # variables connected to heat station
        for i in range(self.i_hs):
            for t in range(TIME_HORIZON):
                for k in range(len(self.keys)):
                    alpha[i, t, k] = m.addVar(
                        lb=0, ub=1, vtype="C", name="alpha(%s,%s,%s)" % (i, t, k)
                    )
                m_hs[i, t] = m.addVar(
                    vtype="C",
                    name="m_hs(%s,%s)" % (i, t),
                    lb=0,
                    ub=self.max_flow_rate_limit,
                )
                d_pump[i, t] = m.addVar(
                    lb=self.min_power_consumption_water_pump.get(i),
                    ub=self.max_power_consumption_water_pump.get(i),
                    vtype="C",
                    name="d_pump(%s, %s)" % (i, t),
                )
        # variables connected to network nodes
        for i in range(self.i_nd):
            for t in range(TIME_HORIZON):
                tau_ns[i, t] = m.addVar(
                    vtype="C",
                    name="tau_ns(%s,%s)" % (i, t),
                    lb=self.min_node_t_supply_network.get(i),
                    ub=self.max_node_t_supply_network.get(i),
                )
                tau_nr[i, t] = m.addVar(
                    vtype="C",
                    name="tau_nr(%s,%s)" % (i, t),
                    lb=self.min_node_t_return_network.get(i),
                    ub=self.max_node_t_return_network.get(i),
                )
                p_ns[i, t] = m.addVar(vtype="C", name="p_ns(%s,%s)" % (i, t))
                p_nr[i, t] = m.addVar(vtype="C", name="p_nr(%s, %s)" % (i, t))

        # variables connected to heat exchanger
        for i in range(self.i_hes):
            for t in range(TIME_HORIZON):
                m_hes[i, t] = m.addVar(
                    vtype="C",
                    name="m_hes(%s,%s)" % (i, t),
                    lb=0,
                    ub=self.max_flow_rate_limit,
                )  # mass flow rate of heat exchanger station at period t
        # variables connected to pipes
        for i in range(self.i_pipe):
            for t in range(TIME_HORIZON):
                ms_pipe[i, t] = m.addVar(
                    ub=self.max_flow_rate_limit,
                    lb=self.min_flow_rate_limit,
                    vtype="C",
                    name="ms_pipe(%s,%s)" % (i, t),
                )  # mass flow rate of pipeline in the supply network
                mr_pipe[i, t] = m.addVar(
                    ub=self.max_flow_rate_limit,
                    lb=self.min_flow_rate_limit,
                    vtype="C",
                    name="mr_pipe(%s,%s)" % (i, t),
                )  # mass flow rate of pipeline in the return network
                tau_PS_out[i, t] = m.addVar(
                    vtype="C",
                    name="tau_PS_out(%s,%s)" % (i, t),
                )  # mass flow temperature considering T drop at the outlet of pipeline in supply network
                tau_PR_out[i, t] = m.addVar(
                    vtype="C",
                    name="tau_PR_out(%s,%s)" % (i, t),
                )  # mass flow temperature considering T drop at the outlet of pipeline in return network
                tau_PS_in[i, t] = m.addVar(
                    vtype="C",
                    name="tau_PS_in(%s,%s)" % (i, t),
                )  # mass flow temperature at the inlet of pipeline in supply network
                tau_PR_in[i, t] = m.addVar(
                    vtype="C",
                    name="tau_PR_in(%s,%s)" % (i, t),
                )  # mass flow temperature at the inlet of pipeline in return network
                tau_PS_no_out[i, t] = m.addVar(
                    vtype="C",
                    name="tau_PS_no_out(%s,%s)" % (i, t),
                )  # outlet T without heat loss
                tau_PR_no_out[i, t] = m.addVar(
                    vtype="C",
                    name="tau_PR_no_out(%s,%s)" % (i, t),
                )  # outlet T without heat loss

        # defining constraints
        # CHP unit

        # constraint from previous time steps
        for i in range(self.i_chp):
            for t in range(len(alpha_his)):
                # m.addCons(
                #     tau_ns[Nd_HS[i], t] == temp_his[t]
                # )
                for k in range(len(self.keys)):
                    m.addCons(alpha[self.Nd_hs[i], t, k] == alpha_his[t][k])

        for i in range(self.i_chp):
            for t in range(TIME_HORIZON):
                m.addCons(
                    quicksum(alpha[i, t, k] for k in range(len(self.keys))) == 1,
                    "alpha_sum_constraint(%s, %s)" % (i, t),
                )
                m.addCons(
                    d_pump[i, t] * MW_W
                    == m_hs[i, t]
                    * (p_ns[self.Nd_hs.get(i), t] - p_nr[self.Nd_hs.get(i), t])
                    / (self.water_pump_efficiency.get(i) * WATER_DENSITY),
                )
                m.addCons(
                    quicksum(
                        alpha[i, t, k] * self.extreme_points_heat.get((i, k))
                        for k in range(len(self.keys))
                    )
                    == C
                    * m_hs[i, t]
                    * (tau_ns[self.Nd_hs.get(i), t] - tau_nr[self.Nd_hs.get(i), t]),
                    name="heat_output_CHP_unit_constraint_G(%s, %s)" % (i, t),
                )

        # Heat exchanger station
        for i in range(self.i_hes):
            for t in range(TIME_HORIZON):
                m.addCons(
                    C
                    * m_hes[i, t]
                    * (tau_ns[self.Nd_hes.get(i), t] - tau_nr[self.Nd_hes.get(i), t])
                    - heat_demand_episode[t]
                    >= -self.delta_heat_demand,
                    name="heat_exchangers_heat_loads_TDHS_G(%s, %s)" % (i, t),
                )
                m.addCons(
                    C
                    * m_hes[i, t]
                    * (tau_ns[self.Nd_hes.get(i), t] - tau_nr[self.Nd_hes.get(i), t])
                    - heat_demand_episode[t]
                    <= self.delta_heat_demand,
                    name="heat_exchangers_heat_loads_TDHS_L(%s, %s)" % (i, t),
                )
                m.addCons(
                    p_ns[self.Nd_hes.get(i), t] - p_nr[self.Nd_hes.get(i), t]
                    >= self.p_hes.get(i),
                    name="minimum_heat_load_pressure(%s, %s)" % (i, t),
                )

        # District heating network
        for i in range(self.i_nd):
            for t in range(TIME_HORIZON):
                m.addCons(
                    self.mass_flow_pipe_in(ms_pipe, i, t, self.s_pipe_supply_in)
                    - self.mass_flow_pipe_out(ms_pipe, i, t, self.s_pipe_supply_out)
                    - self.mass_flow_hs(m_hs, i, t)
                    + self.mass_flow_hes(m_hes, i, t)
                    <= self.delta_mass_flow,
                    name="continuity_of_supply_network_mass_flow_L(%s, %s)" % (i, t),
                )
                m.addCons(
                    self.mass_flow_pipe_in(ms_pipe, i, t, self.s_pipe_supply_in)
                    - self.mass_flow_pipe_out(ms_pipe, i, t, self.s_pipe_supply_out)
                    - self.mass_flow_hs(m_hs, i, t)
                    + self.mass_flow_hes(m_hes, i, t)
                    >= -self.delta_mass_flow,
                    name="continuity_of_supply_network_mass_flow_G(%s, %s)" % (i, t),
                )

                m.addCons(
                    self.mass_flow_pipe_in(mr_pipe, i, t, self.s_pipe_return_in)
                    - self.mass_flow_pipe_out(mr_pipe, i, t, self.s_pipe_return_out)
                    - self.mass_flow_hes(m_hes, i, t)
                    + self.mass_flow_hs(m_hs, i, t)
                    <= self.delta_mass_flow,
                    name="continuity_of_return_network_mass_flow_L(%s, %s)" % (i, t),
                )
                m.addCons(
                    self.mass_flow_pipe_in(mr_pipe, i, t, self.s_pipe_return_in)
                    - self.mass_flow_pipe_out(mr_pipe, i, t, self.s_pipe_return_out)
                    - self.mass_flow_hes(m_hes, i, t)
                    + self.mass_flow_hs(m_hs, i, t)
                    >= -self.delta_mass_flow,
                    name="continuity_of_return_network_mass_flow_G(%s, %s)" % (i, t),
                )
                m.addCons(
                    self.temp_mixing_outlet(
                        tau_PS_out, ms_pipe, i, t, self.s_pipe_supply_out
                    )
                    - tau_ns[i, t]
                    * self.mass_flow_pipe_out(ms_pipe, i, t, self.s_pipe_supply_out)
                    <= self.delta_mass_flow,
                    name="temperature_mixing_outlet_supply_network_L(%s, %s)" % (i, t),
                )
                m.addCons(
                    self.temp_mixing_outlet(
                        tau_PS_out, ms_pipe, i, t, self.s_pipe_supply_out
                    )
                    - tau_ns[i, t]
                    * self.mass_flow_pipe_out(ms_pipe, i, t, self.s_pipe_supply_out)
                    >= -self.delta_mass_flow,
                    name="temperature_mixing_outlet_supply_network_G(%s, %s)" % (i, t),
                )
                m.addCons(
                    self.temp_mixing_outlet(
                        tau_PR_out, mr_pipe, i, t, self.s_pipe_return_out
                    )
                    - tau_nr[i, t]
                    * self.mass_flow_pipe_out(mr_pipe, i, t, self.s_pipe_return_out)
                    <= self.delta_mass_flow,
                    name="temperature_mixing_outlet_return_network_L(%s, %s)" % (i, t),
                )
                m.addCons(
                    self.temp_mixing_outlet(
                        tau_PR_out, mr_pipe, i, t, self.s_pipe_return_out
                    )
                    - tau_nr[i, t]
                    * self.mass_flow_pipe_out(mr_pipe, i, t, self.s_pipe_return_out)
                    >= -self.delta_mass_flow,
                    name="temperature_mixing_outlet_return_network_G(%s, %s)" % (i, t),
                )
                self.temp_mixing_inlet(
                    m,
                    tau_PS_in,
                    tau_ns,
                    self.s_pipe_supply_in,
                    i,
                    t,
                    "temperature_mixing_inlet_supply_network(%s, %s, %s)",
                )
                self.temp_mixing_inlet(
                    m,
                    tau_PR_in,
                    tau_nr,
                    self.s_pipe_return_in,
                    i,
                    t,
                    "temperature_mixing_inlet_return_network(%s, %s, %s)",
                )
        for b in range(self.i_pipe):
            for t in range(TIME_HORIZON):
                m.addCons(
                    p_ns[self.Nd_pf.get(b), t]
                    - p_ns[self.Nd_pt.get(b), t]
                    - self.coefficient_of_pressure_loss.get(b)
                    * ms_pipe[b, t]
                    * ms_pipe[b, t]
                    <= self.delta_pressure,
                    name="pressure_loss_supply_net_L (%s, %s)" % (b, t),
                )
                m.addCons(
                    p_ns[self.Nd_pf.get(b), t]
                    - p_ns[self.Nd_pt.get(b), t]
                    - self.coefficient_of_pressure_loss.get(b)
                    * ms_pipe[b, t]
                    * ms_pipe[b, t]
                    >= -self.delta_pressure,
                    name="pressure_loss_supply_net_G (%s, %s)" % (b, t),
                )
                m.addCons(
                    p_nr[self.Nd_pt.get(b), t]
                    - p_nr[self.Nd_pf.get(b), t]
                    - self.coefficient_of_pressure_loss.get(b)
                    * mr_pipe[b, t]
                    * mr_pipe[b, t]
                    <= self.delta_pressure,
                    name="pressure_loss_return_net_L (%s, %s)" % (b, t),
                )
                m.addCons(
                    p_nr[self.Nd_pt.get(b), t]
                    - p_nr[self.Nd_pf.get(b), t]
                    - self.coefficient_of_pressure_loss.get(b)
                    * mr_pipe[b, t]
                    * mr_pipe[b, t]
                    >= -self.delta_pressure,
                    name="pressure_loss_return_net_G (%s, %s)" % (b, t),
                )
                m.addCons(
                    (
                        tau_PS_no_out[b, t]
                        - (
                            (
                                coeff_R[b, t]
                                - WATER_DENSITY
                                * self.cross_sectional_area_surface[b]
                                * self.pipe_len[b]
                            )
                            * tau_PS_in[b, t - time_delay_II[b, t]]
                            + quicksum(
                                ms_pipe[b, k] * HOUR_TO_SEC * tau_PS_in[b, k]
                                for k in range(
                                    t - time_delay_I[b, t] + self.time_interval,
                                    t - time_delay_II[b, t],
                                    self.time_interval,
                                )
                            )
                            + (
                                ms_pipe[b, t] * HOUR_TO_SEC
                                + WATER_DENSITY
                                * self.cross_sectional_area_surface[b]
                                * self.pipe_len[b]
                                - coeff_S[b, t]
                            )
                            * tau_PS_in[b, t - time_delay_I[b, t]]
                        )
                        / (ms_pipe[b, t] * HOUR_TO_SEC)
                    )
                    <= self.delta_temperature,
                    name="outlet_supply_T_without_heat_loss_L (%s, %s)" % (b, t),
                )
                m.addCons(
                    (
                        tau_PS_no_out[b, t]
                        - (
                            (
                                coeff_R[b, t]
                                - WATER_DENSITY
                                * self.cross_sectional_area_surface[b]
                                * self.pipe_len[b]
                            )
                            * tau_PS_in[b, t - time_delay_II[b, t]]
                            + quicksum(
                                ms_pipe[b, k] * HOUR_TO_SEC * tau_PS_in[b, k]
                                for k in range(
                                    t - time_delay_I[b, t] + self.time_interval,
                                    t - time_delay_II[b, t],
                                    self.time_interval,
                                )
                            )
                            + (
                                ms_pipe[b, t] * HOUR_TO_SEC
                                + WATER_DENSITY
                                * self.cross_sectional_area_surface[b]
                                * self.pipe_len[b]
                                - coeff_S[b, t]
                            )
                            * tau_PS_in[b, t - time_delay_I[b, t]]
                        )
                        / (ms_pipe[b, t] * HOUR_TO_SEC)
                    )
                    >= -self.delta_temperature,
                    name="outlet_supply_T_without_heat_loss_G (%s, %s)" % (b, t),
                )

                m.addCons(
                    (
                        tau_PR_no_out[b, t]
                        - (
                            (
                                coeff_R[b, t]
                                - WATER_DENSITY
                                * self.cross_sectional_area_surface[b]
                                * self.pipe_len[b]
                            )
                            * tau_PR_in[b, t - time_delay_II[b, t]]
                            + quicksum(
                                ms_pipe[b, k] * HOUR_TO_SEC * tau_PR_in[b, k]
                                for k in range(
                                    t - time_delay_I[b, t] + self.time_interval,
                                    t - time_delay_II[b, t],
                                    self.time_interval,
                                )
                            )
                            + (
                                ms_pipe[b, t] * HOUR_TO_SEC
                                + WATER_DENSITY
                                * self.cross_sectional_area_surface[b]
                                * self.pipe_len[b]
                                - coeff_S[b, t]
                            )
                            * tau_PR_in[b, t - time_delay_I[b, t]]
                        )
                        / (ms_pipe[b, t] * HOUR_TO_SEC)
                    )
                    <= self.delta_temperature,
                    name="outlet_return_T_without_heat_loss_L (%s, %s)" % (b, t),
                )
                m.addCons(
                    (
                        tau_PR_no_out[b, t]
                        - (
                            (
                                coeff_R[b, t]
                                - WATER_DENSITY
                                * self.cross_sectional_area_surface[b]
                                * self.pipe_len[b]
                            )
                            * tau_PR_in[b, t - time_delay_II[b, t]]
                            + quicksum(
                                ms_pipe[b, k] * HOUR_TO_SEC * tau_PR_in[b, k]
                                for k in range(
                                    t - time_delay_I[b, t] + self.time_interval,
                                    t - time_delay_II[b, t],
                                    self.time_interval,
                                )
                            )
                            + (
                                ms_pipe[b, t] * HOUR_TO_SEC
                                + WATER_DENSITY
                                * self.cross_sectional_area_surface[b]
                                * self.pipe_len[b]
                                - coeff_S[b, t]
                            )
                            * tau_PR_in[b, t - time_delay_I[b, t]]
                        )
                        / (ms_pipe[b, t] * HOUR_TO_SEC)
                    )
                    >= -self.delta_temperature,
                    name="outlet_return_T_without_heat_loss_G (%s, %s)" % (b, t),
                )
                """
                m.addCons(
                    tau_PS_out[b, t]
                    - tau_am[t]
                    - (tau_PS_no_out[b, t] - tau_am[t])
                    * exp(
                        -(HEAT_TRANSFER_COEFF[b] * HOUR_TO_SEC)
                        / (CROSS_SECTIONAL_AREA_SURFACE[b] * WATER_DENSITY * C)
                        * (
                            time_delay_II[b, t]
                            + 0.5
                            + (coeff_S[b, t] - coeff_R[b, t])
                            / (ms_pipe[b, t - time_delay_II[b, t]] * HOUR_TO_SEC)
                        )
                    )
                    == 0,
                    name="outlet_supply_T_with_heat_loss (%s, %s)" % (b, t),
                )
                m.addCons(
                    tau_PR_out[b, t]
                    - tau_am[t]
                    - (tau_PR_no_out[b, t] - tau_am[t])
                    * exp(
                        -(HEAT_TRANSFER_COEFF[b] * HOUR_TO_SEC)
                        / (CROSS_SECTIONAL_AREA_SURFACE[b] * WATER_DENSITY * C)
                        * (
                            time_delay_II[b, t]
                            + 0.5
                            + (coeff_S[b, t] - coeff_R[b, t])
                            / (mr_pipe[b, t - time_delay_II[b, t]] * HOUR_TO_SEC)
                        )
                    )
                    == 0,
                    name="outlet_return_T_with_heat_loss_L (%s, %s)" % (b, t),
                )
                """
                m.addCons(
                    tau_PS_out[b, t]
                    - self.tau_am[t]
                    - (tau_PS_no_out[b, t] - self.tau_am[t])
                    * exp(
                        -(HEAT_TRANSFER_COEFF[b] * self.pipe_len[b])
                        / (ms_pipe[b, t] * C)
                    )
                    == 0,
                    name="outlet_supply_T_with_heat_loss (%s, %s)" % (b, t),
                )
                m.addCons(
                    tau_PR_out[b, t]
                    - self.tau_am[t]
                    - (tau_PR_no_out[b, t] - self.tau_am[t])
                    * exp(
                        -(HEAT_TRANSFER_COEFF[b] * self.pipe_len[b])
                        / (mr_pipe[b, t] * C)
                    )
                    == 0,
                    name="outlet_return_T_with_heat_loss (%s, %s)" % (b, t),
                )
        objvar = m.addVar(name="objvar", vtype="C", lb=None, ub=None)
        m.setObjective(objvar, "minimize")
        m.addCons(
            objvar
            >= (
                quicksum(
                    quicksum(self.C_chp(alpha, i, t) for t in range(TIME_HORIZON))
                    for i in range(self.i_chp)
                )
                - quicksum(
                    float(electricity_price_episode[t])
                    * self.electricity_sell(alpha, t, d_pump)
                    for t in range(TIME_HORIZON)
                )
            ),
            name="objconst",
        )
        m.data = ms_pipe
        m.optimize()
        return m

    def update_complicating_variables(
        self,
        m,
        TIME_DELAY_I,
        TIME_DELAY_II,
        COEFFICIENT_VARIABLE_S,
        COEFFICIENT_VARIABLE_R,
        start_time,
    ):
        ms_pipe_sol = m.data
        ms_pipe = np.empty((self.i_pipe, TIME_HORIZON), dtype=float)
        for key, value in ms_pipe_sol.items():
            ms_pipe[list(key)[0], list(key)[1]] = m.getVal(value)

        for b in range(self.i_pipe):
            pipe_vol = (
                WATER_DENSITY * self.cross_sectional_area_surface[b] * self.pipe_len[b]
            )
            for t in range(start_time, TIME_HORIZON):
                k, cumsum = 0, ms_pipe[b, t] * HOUR_TO_SEC
                while cumsum < pipe_vol and k < t:
                    k += 1
                    cumsum += ms_pipe[b, t - k] * HOUR_TO_SEC
                TIME_DELAY_II[b, t] = k
                cumsum -= ms_pipe[b, t] * HOUR_TO_SEC
                while cumsum < pipe_vol and k < t:
                    k += 1
                    cumsum += ms_pipe[b, t - k] * HOUR_TO_SEC
                TIME_DELAY_I[b, t] = k
                COEFFICIENT_VARIABLE_R[b, t] = np.sum(
                    ms_pipe[b, (t - TIME_DELAY_II[b, t]) : (t + 1)] * HOUR_TO_SEC
                )
                if TIME_DELAY_I[b, t] >= TIME_DELAY_II[b, t] + 1:
                    COEFFICIENT_VARIABLE_S[b, t] = np.sum(
                        ms_pipe[b, (t - TIME_DELAY_I[b, t] + 1) : (t + 1)] * HOUR_TO_SEC
                    )
                else:
                    COEFFICIENT_VARIABLE_S[b, t] = COEFFICIENT_VARIABLE_R[b, t]
        return [
            TIME_DELAY_I,
            TIME_DELAY_II,
            COEFFICIENT_VARIABLE_R,
            COEFFICIENT_VARIABLE_S,
        ]
