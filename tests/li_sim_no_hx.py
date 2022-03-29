import numpy as np
import os
import pandas as pd
import pickle
from pathlib import Path
from ..src.simulator.CHP.CHP import CHP
from ..src.simulator.grid.models import Producer, Edge, GridObject, Node
from .sim_eval import MinlpEval

TIME_STEPS = 24
HIS_SUPPLY_T = 90
HIS_RETURN_T = 50
PIPE_DIAMETER = 0.5958
PIPE_LENGTH = 4000
ground_temp = 10
C = 4182

DATA_PATH = Path(__file__).parents[1] / "data"
# RESULT_PATH = Path(__file__).parents[1] / "results/li_2016/li_2016_day_ahead"
RESULT_PATH = Path(__file__).parents[1] / "results/li_2016/li_2016_day_ahead"


def load_data():
    heat_demand_dataset = np.array(pd.read_csv(DATA_PATH / "heat_demand_test.csv"))

    result_dict = {}
    for (dirpath, dirnames, filenames) in os.walk(RESULT_PATH):
        for filename in filenames:
            if filename.endswith(".pickle") & (filename[:7] == "episode"):
                # if filename.endswith(".pickle") & (filename[:16] == "ncl_episode_with"):
                ep_start_idx = filename.find("ep_") + len("ep_")
                ep_end_idx = filename.find("_", ep_start_idx)
                ep = int(filename[ep_start_idx:ep_end_idx])

                length_start_idx = filename.find("L_") + len("L_")
                length_end_idx = filename.find("_", length_start_idx)
                length = int(filename[length_start_idx:length_end_idx])

                with open(os.path.join(dirpath, filename), "rb") as f:
                    x = pickle.load(f)
                    result_dict[(length, ep)] = x

    return result_dict, heat_demand_dataset


class Pseudo_Consumer(Node):
    def __init__(self, demand):
        super().__init__(
            id=1,
            slots=("Supply", "Return"),
            blocks=len(demand),
        )
        self.demand = demand

        self.heat_capacity = C

    def solve(self, mass_flow):

        t_supply_p = self.edges[
            0
        ].get_outlet_temp()  # output temperature of dupply network
        t_return_p = t_supply_p - self.demand[self.current_step] / C / mass_flow

        self.mass_flow[0, self.current_step] = mass_flow
        self.mass_flow[1, self.current_step] = mass_flow
        self.temp[0, self.current_step] = t_supply_p
        self.temp[1, self.current_step] = t_return_p

        self.solvable_callback(self.edges[0], 1, mass_flow)  # consumer on slot 1
        self.solvable_callback(self.edges[1], 0, mass_flow)  # producer on slot 0

    def set_mass_flow(self, slot: int, mass_flow: float) -> None:
        raise Exception("set_mass_flow should not be called on a consumer")

    def clear(self):
        self._clear()
        self.pressure = np.full((2, TIME_STEPS), 0, dtype=float)
        self.pressure[0] = 100000

    def get_outlet_temp(self, slot: int) -> float:
        """
        Is called from downstream to get the average outlet temperature in the
        coming step.
        """
        assert slot == 1

        outlet_temp = self.temp[1, self.current_step]
        assert not np.isnan(outlet_temp)

        return outlet_temp


class Grid_without_HX:
    def __init__(self, demand, his_t_sup, his_t_ret, pipe_len):
        self.interval_length = 3600
        self.producer = Producer(
            blocks=TIME_STEPS,
            temp_upper_bound=110,
            control_with_temp=True,
        )

        self.producer.add_to_grid(
            self.solvable,
            self.interval_length,
        )

        chp_type = "keypts"
        self.CHP = CHP(
            chp_type=chp_type,
        )

        self.consumer = Pseudo_Consumer(demand * 10 ** 6)

        self.consumer.add_to_grid(
            self.solvable,
            self.interval_length,
        )
        self.edge_sup = Edge(
            blocks=TIME_STEPS,
            historical_t_in=his_t_sup,
            diameter=PIPE_DIAMETER,
            length=pipe_len,
            thermal_resistance=1.36,  # Valkema p25
            t_ground=ground_temp,
            max_flow_speed=3,
            min_flow_speed=0,
            friction_coefficient=8.054,
        )

        self.edge_ret = Edge(
            blocks=TIME_STEPS,
            historical_t_in=his_t_ret,
            diameter=PIPE_DIAMETER,
            length=pipe_len,
            thermal_resistance=1.36,  # Valkema p25
            t_ground=ground_temp,
            max_flow_speed=3,
            min_flow_speed=0,
            friction_coefficient=8.054,
        )

        self.edge_sup.link(
            nodes=(
                (self.producer, 0),
                (self.consumer, 0),
            )
        )
        self.edge_ret.link(
            nodes=(
                (self.consumer, 1),
                (self.producer, 1),
            )
        )
        self.edge_sup.add_to_grid(
            self.solvable,
            self.interval_length,
        )
        self.edge_ret.add_to_grid(
            self.solvable,
            self.interval_length,
        )
        self.link_nodes()

        self.producer.clear()
        self.consumer.clear()
        self.edge_sup.clear()
        self.edge_ret.clear()
        GridObject.reset_step()

        self._solvable_objects = []

    def solvable(self, object: GridObject, slot: int, mass_flow: float) -> None:
        self._solvable_objects.append((object, slot, mass_flow))

    def solve(self, temp, mass_flows):
        # cost, heat_prod, power_prod = self.CHP.solve_alpha(alpha)
        self.producer.temp[0] = temp
        for _ in range(TIME_STEPS):
            time_step = GridObject._current_step
            self.consumer.solve(mass_flows[time_step])

            while self._solvable_objects:
                (obj, slot, mass_flow) = self._solvable_objects.pop(0)
                flag = obj.set_mass_flow(slot, mass_flow)

            GridObject.increase_step()

        return (
            self.producer.q,
            self.edge_sup.temp[1],
            self.edge_ret.temp[0],
            self.edge_ret.temp[1],
        )

    def link_nodes(self) -> None:

        self.producer.link(tuple([self.edge_sup, self.edge_ret]))
        self.consumer.link(tuple([self.edge_sup, self.edge_ret]))


if __name__ == "__main__":
    results, demand_dataset = load_data()
    end_temp_diff = []
    for key, value in results.items():
        demand = demand_dataset[key[1] - 1]
        pipe_len = key[0]
        alpha = value["action"]
        if np.any(np.sum(alpha, axis=1) < 0.99):
            continue

        mass_flows = value["m_hs"]
        T_sup_in = value["tau_PS_in"]
        T_sup_out = value["tau_PS_out"]
        T_ret_in = value["tau_PR_in"]
        T_ret_out = value["tau_PR_out"]

        sim = Grid_without_HX(demand, T_sup_in[0], T_ret_in[0], pipe_len)
        sim_heat_prod, sim_T_sup_out, sim_T_ret_in, sim_T_ret_out = sim.solve(
            T_sup_in, mass_flows
        )

        print(
            key,
            np.mean(sim_T_sup_out - T_sup_out),
            np.mean(sim_T_ret_out - T_ret_out),
        )
        end_temp_diff.append(np.mean(np.abs(sim_T_ret_out - T_ret_out)))

    print(np.mean(end_temp_diff), np.std(end_temp_diff))
