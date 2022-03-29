import math
import numpy as np
from abc import ABC, abstractmethod
from itertools import groupby, product
from typing import Tuple
from .RL_utils import *
from util.util import *


class State(ABC):
    """
    State space of Q-learning algorithm.
    """

    def __init__(
        self,
        pipe_len,
        pipe_diameter,
        time_interval,
        max_flow_speed,
        max_supply_temp,
        min_supply_temp,
        historical_t_supply,
        historical_t_return,
        discretization_mass_step,
        discretization_T_step,
        result_p,
    ):
        self.pipe_len: float = pipe_len
        self.pipe_diameter: float = pipe_diameter
        self.time_interval: int = time_interval
        self.max_flow_speed: float = max_flow_speed
        self.max_supply_temp: int = max_supply_temp
        self.min_supply_temp: int = min_supply_temp
        self.historical_t_supply: int = historical_t_supply
        self.historical_t_return: int = historical_t_return
        self.discretization_mass_step: int = discretization_mass_step
        self.discretization_T_step: int = discretization_T_step
        # maximum capacity of the pipe
        self.pipe_volume: float = (
            WATER_DENSITY * pipe_len * pow(pipe_diameter / 2, 2) * PI
        )
        # maximum capacity of the pipe according to the mass flow
        self.mass_fun: float = (
            WATER_DENSITY
            * pow(pipe_diameter / 2, 2)
            * PI
            * max_flow_speed
            * time_interval
            * HOUR_TO_SEC
        )
        self.result_p: Path = result_p

    @abstractmethod
    def create_observation_space(self, *args, **kwargs):
        """
        Creation of all temperature, mass combinations of water chunks.
        """
        pass

    @abstractmethod
    def state_transition(self, *args, **kwargs):
        """
        State transition based on newly received observations.
        """
        pass

    @abstractmethod
    def reset(self, *args, **kwargs):
        """
        Reset the state on initial conditions.
        """
        pass

    @abstractmethod
    def convergence_pair(self, *args, **kwargs):
        """
        Get state, action pair for inspecting the convergence of Q-learning algorithm.
        """
        pass

    @property
    def get_mass(self):
        """
        Get water mass in the pipe.
        """
        return self.mass_fun

    @property
    def get_result_p(self):
        """
        Get Path to the result file.
        """
        return self.result_p

    def heat_demand_electricity_price_combinations(self) -> Tuple[list, list]:
        """
        Creation of heat demand, electricity price combinations.
        """
        heat_demand_combinations, electricity_price_combinations = [], []
        heat_demand_combinations.extend(
            [
                list(sublist)
                for sublist in (
                    Data.heat_demand_data[x : x + RLParams.future_heat_demand]
                    for x in range(
                        Data.heat_demand_data_len - RLParams.future_heat_demand + 1
                    )
                )
            ]
        )
        heat_demand_combinations.sort()
        heat_demand_combinations = list(k for k, _ in groupby(heat_demand_combinations))
        electricity_price_combinations.extend(
            [
                list(sublist)
                for sublist in (
                    Data.electricity_price_data[
                        x : x + RLParams.future_electricity_price
                    ]
                    for x in range(
                        Data.electricity_price_data_len
                        - RLParams.future_electricity_price
                        + 1
                    )
                )
            ]
        )
        electricity_price_combinations.sort()
        electricity_price_combinations = list(
            k for k, _ in groupby(electricity_price_combinations)
        )
        return heat_demand_combinations, electricity_price_combinations


class FullState(State):
    """
    Model of the full state space.
    """

    def __init__(
        self,
        pipe_len=4000,
        pipe_diameter=0.5958,
        time_interval=1,
        max_flow_speed=3,
        max_supply_temp=110,
        min_supply_temp=70,
        historical_t_supply=90,
        historical_t_return=50,
        discretization_mass_step=200000,
        discretization_T_step=10,
        future_season=0,
        future_time_of_the_day=0,
        future_demand=None,
        future_elec_price=None,
    ):
        super().__init__(
            pipe_len=pipe_len,
            pipe_diameter=pipe_diameter,
            time_interval=time_interval,
            max_flow_speed=max_flow_speed,
            max_supply_temp=max_supply_temp,
            min_supply_temp=min_supply_temp,
            historical_t_supply=historical_t_supply,
            historical_t_return=historical_t_return,
            discretization_mass_step=discretization_mass_step,
            discretization_T_step=discretization_T_step,
            result_p=Path(__file__).parents[3] / "results/rl_full_state/rl_full_state",
        )
        if future_demand is None:
            future_demand = []
        if future_elec_price is None:
            future_elec_price = []
        self.mass = self.pipe_volume
        self.number_of_elements = int(
            math.floor(self.pipe_volume / discretization_mass_step)
        )
        self.value = [[self.pipe_volume, self.historical_t_supply]]
        self.future_season = future_season
        self.future_time_of_the_day = future_time_of_the_day
        self.future_demand = future_demand
        self.future_elec_price = future_elec_price

    def create_observation_space(self) -> list:
        """
        Minimization of discretization error in temperature.
        """
        observation_space_pipe = []
        temp = list(
            range(
                self.min_supply_temp,
                self.max_supply_temp + self.discretization_T_step,
                self.discretization_T_step,
            )
        )
        observation_space_pipe.extend(
            [[*combo] for combo in product(temp, repeat=self.number_of_elements + 1)]
        )
        print(len(observation_space_pipe))
        (
            heat_demand_combinations,
            electricity_price_combinations,
        ) = self.heat_demand_electricity_price_combinations()
        observation_space = [
            [*item]
            for item in product(
                observation_space_pipe,
                RLParams.season_combinations,
                RLParams.time_of_the_day_combinations,
                heat_demand_combinations,
                electricity_price_combinations,
            )
        ]
        print(len(observation_space))
        return observation_space

    def create_observation_space_m(self) -> list:
        """
        Minimization of discretization error in mass flow.
        """
        observation_space_pipe = []
        max_mass_round = pipe_disc(self.mass_fun, self.discretization_mass_step)
        pipe_capacity_round = math.floor(
            self.discretization_mass_step
            * math.floor(self.pipe_volume / self.discretization_mass_step)
        )
        if max_mass_round > pipe_capacity_round:
            max_mass_round = pipe_capacity_round
        mass = [
            i * self.discretization_mass_step
            for i in list(
                range(
                    1,
                    int(max_mass_round / self.discretization_mass_step) + 1,
                )
            )
        ]
        mass_combinations = FullState.mass_combinations_fun(
            mass, pipe_capacity_round, self.pipe_volume
        )
        temp_len = max(len(elem) for elem in mass_combinations)
        temp = list(
            range(
                self.min_supply_temp,
                self.max_supply_temp + self.discretization_T_step,
                self.discretization_T_step,
            )
        )
        temp_combinations = FullState.temp_combinations_fun(temp, temp_len)
        for i in range(len(temp_combinations)):
            for j in range(len(mass_combinations)):
                if len(temp_combinations[i]) == len(mass_combinations[j]):
                    observation_space_pipe.append(
                        [
                            list(x)
                            for x in zip(mass_combinations[j], temp_combinations[i])
                        ]
                    )
        print(len(observation_space_pipe))
        # FullState.test_size(mass_combinations, temp_combinations, len(observation_space_pipe))
        (
            heat_demand_combinations,
            electricity_price_combinations,
        ) = self.heat_demand_electricity_price_combinations()
        observation_space = [
            [*item, EnvParams.base_return_T]
            for item in product(
                observation_space_pipe,
                heat_demand_combinations,
                electricity_price_combinations,
            )
        ]
        return observation_space

    def state_transition(
        self,
        supply_in_temp,
        supply_out_temp,
        return_in_temp,
        M,
        season_update,
        time_of_the_day_update,
        demand_update,
        elec_price_update,
    ) -> float:
        """
        self.state.value.insert(0, [M, supply_in_temp])
        """
        for j in range(int(M / self.discretization_mass_step)):
            self.value.insert(0, [self.discretization_mass_step, supply_in_temp])
        self.mass += M
        d = self.mass - self.pipe_volume
        R = 0
        while d >= self.value[-1][0]:
            d -= self.value[-1][0]
            R += (C * self.value[-1][0] * (self.value[-1][1] - return_in_temp)) / (
                self.time_interval * HOUR_TO_SEC
            )
            self.mass -= self.value[-1][0]
            self.value.pop()
        R += (
            C
            * d
            * (self.value[-1][1] - return_in_temp)
            / (self.time_interval * HOUR_TO_SEC)
        )
        self.mass -= d
        self.value[-1][0] -= d
        self.future_season = season_update
        self.future_time_of_the_day = time_of_the_day_update
        self.future_demand.pop(0)
        self.future_demand.append(demand_update)
        self.future_elec_price.pop(0)
        self.future_elec_price.append(elec_price_update)
        return R

    def reset(
        self,
        future_season=0,
        future_time_of_the_day=0,
        future_demand=None,
        future_elec_price=None,
    ) -> list:
        """
        Reset the state space on initial (time-step zero) state.
        """
        if future_demand is None:
            future_demand = []
        if future_elec_price is None:
            future_elec_price = []
        self.mass = self.pipe_volume
        self.value = [
            # number of blocks should be equal to the number of elements
            [self.discretization_mass_step, self.historical_t_supply]
            for _ in range(self.number_of_elements)
        ] + [
            [
                self.pipe_volume
                - self.number_of_elements * self.discretization_mass_step,
                self.historical_t_supply,
            ]
        ]
        self.future_season = future_season
        self.future_time_of_the_day = future_time_of_the_day
        self.future_demand = future_demand
        self.future_elec_price = future_elec_price

        return [
            [self.historical_t_supply] * (self.number_of_elements + 1),
            self.future_season,
            self.future_time_of_the_day,
            self.future_demand,
            self.future_elec_price,
        ]
        """
        return [
          [[self.pipe_volume, self.historical_t_supply]],
           self.future_demand,
           BASE_RETURN_T]
        """

    def convergence_pair(self) -> Tuple[tuple, tuple, tuple, tuple]:
        """
        Get state, action pair for inspecting the convergence of Q-learning algorithm.
        """
        heat_demand_discrete = [5, 20, 40]
        electricity_price_discrete = [20, 40, 60]
        s1 = [
            [self.max_supply_temp - self.discretization_T_step]
            * (self.number_of_elements + 1),
            RLParams.season_combinations[0],
            RLParams.time_of_the_day_combinations[0],
            [list(heat_demand_discrete)[2]],
            [list(electricity_price_discrete)[1]],
        ]
        a1 = RLParams.action_space[10]
        s2 = [
            [self.min_supply_temp + self.discretization_T_step]
            * (self.number_of_elements + 1),
            RLParams.season_combinations[2],
            RLParams.time_of_the_day_combinations[2],
            [list(heat_demand_discrete)[0]],
            [list(electricity_price_discrete)[1]],
        ]
        a2 = RLParams.action_space[0]
        s1, a1, s2, a2 = to_tuple(s1), to_tuple(a1), to_tuple(s2), to_tuple(a2)
        return s1, a1, s2, a2

    @property
    def get_num_elem(self) -> int:
        """
        Get number of water chunks.
        """
        return self.number_of_elements

    @property
    def get_next_state(self) -> list:
        """
        Get the external state space part of the next state.
        """
        return [
            list(list(zip(*self.value))[1]),
            self.future_season,
            self.future_time_of_the_day,
            self.future_demand,
            self.future_elec_price,
        ]

    @staticmethod
    def mass_combinations_fun(numbers, target, pipe_capacity) -> list:
        """
        Combinations of water chunks' masses.
        """
        results = []
        for x in range(len(numbers)):
            results.extend(
                [
                    [*combo, pipe_capacity - sum(combo)]
                    for combo in product(numbers, repeat=x)
                    if sum(combo) <= target
                ]
            )
        return results

    @staticmethod
    def temp_combinations_fun(numbers, n) -> list:
        """
        Combinations of water chunks' temperatures.
        """
        results = []
        for x in range(n):
            results.extend([[*combo] for combo in product(numbers, repeat=x + 1)])

        return results

    @staticmethod
    def test_size(mass, temp, n) -> None:
        elem_len = max(len(elem) for elem in mass)
        elem_count_mass = np.zeros(elem_len + 1)
        elem_count_temp = np.zeros(elem_len + 1)
        for i in range(elem_len + 1):
            for x in range(len(mass)):
                if len(mass[x]) == i:
                    elem_count_mass[i] += 1
            for y in range(len(temp)):
                if len(temp[y]) == i:
                    elem_count_temp[i] += 1
        final_sum = 0
        for i in range(len(elem_count_mass)):
            final_sum += elem_count_mass[i] * elem_count_temp[i]
        if int(final_sum) != n:
            raise Exception(
                "Dimension dismatch of temperature and mass flow (M) Cartesian product combination with their own dimensions"
            )


class AbstractState(State):
    """
    Model of the partial state space.
    """

    def __init__(
        self,
        pipe_len=4000,
        pipe_diameter=0.5958,
        time_interval=1,
        max_flow_speed=3,
        max_supply_temp=110,
        min_supply_temp=70,
        historical_t_supply=90,
        historical_t_return=50,
        discretization_mass_step=42000,
        discretization_T_step=3,
        future_season=0,
        future_time_of_the_day=0,
        future_demand=None,
        future_elec_price=None,
    ):
        super().__init__(
            pipe_len=pipe_len,
            pipe_diameter=pipe_diameter,
            time_interval=time_interval,
            max_flow_speed=max_flow_speed,
            max_supply_temp=max_supply_temp,
            min_supply_temp=min_supply_temp,
            historical_t_supply=historical_t_supply,
            historical_t_return=historical_t_return,
            discretization_mass_step=discretization_mass_step,
            discretization_T_step=discretization_T_step,
            result_p=Path(__file__).parents[3]
            / "results/rl_abstract_state/rl_abstract_state",
        )
        if future_demand is None:
            future_demand = []
        if future_elec_price is None:
            future_elec_price = []
        self.inlet_temp = historical_t_supply
        self.outlet_temp = historical_t_supply
        self.mass = 0
        self.future_season = future_season
        self.future_time_of_the_day = future_time_of_the_day
        self.future_demand = future_demand
        self.future_elec_price = future_elec_price

    def create_observation_space(self) -> list:
        """
        Creation of all temperature, mass combinations of water chunks.
        """
        temp = list(
            range(
                self.min_supply_temp,
                self.max_supply_temp + self.discretization_T_step,
                self.discretization_T_step,
            )
        )
        mass = list(
            range(
                0,
                pipe_disc(self.mass_fun, self.discretization_mass_step)
                + self.discretization_mass_step,
                self.discretization_mass_step,
            )
        )
        observation_space_pipe = [
            [*item]
            for item in product(
                temp,
                temp,
                mass,
            )
        ]
        print(len(observation_space_pipe))
        (
            heat_demand_combinations,
            electricity_price_combinations,
        ) = self.heat_demand_electricity_price_combinations()
        observation_space = [
            [*item]
            for item in product(
                observation_space_pipe,
                RLParams.season_combinations,
                RLParams.time_of_the_day_combinations,
                heat_demand_combinations,
                electricity_price_combinations,
            )
        ]
        print(len(observation_space))
        return observation_space

    def state_transition(
        self,
        supply_in_temp,
        supply_out_temp,
        return_in_temp,
        M,
        season_update,
        time_of_the_day_update,
        demand_update,
        elec_price_update,
    ) -> float:
        """
        State transition based on newly received observations.
        """
        self.inlet_temp = supply_in_temp
        self.outlet_temp = supply_out_temp
        self.mass = M
        R = (
            C
            * self.mass
            * (self.outlet_temp - return_in_temp)
            / (self.time_interval * HOUR_TO_SEC)
        )
        self.future_season = season_update
        self.future_time_of_the_day = time_of_the_day_update
        self.future_demand.pop(0)
        self.future_demand.append(demand_update)
        self.future_elec_price.pop(0)
        self.future_elec_price.append(elec_price_update)
        return R

    def reset(
        self,
        future_season=0,
        future_time_of_the_day=0,
        future_demand=None,
        future_elec_price=None,
    ) -> list:
        """
        Reset the state space on initial (time-step zero) state.
        """
        if future_demand is None:
            future_demand = []
        if future_elec_price is None:
            future_elec_price = []
        self.inlet_temp = self.historical_t_supply
        self.outlet_temp = self.historical_t_supply
        self.mass = 0
        self.future_season = future_season
        self.future_time_of_the_day = future_time_of_the_day
        self.future_demand = future_demand
        self.future_elec_price = future_elec_price

        return [
            [self.inlet_temp, self.outlet_temp, self.mass],
            self.future_season,
            self.future_time_of_the_day,
            self.future_demand,
            self.future_elec_price,
        ]

    def convergence_pair(self) -> Tuple[tuple, tuple, tuple, tuple]:
        """
        Get state, action pair for inspecting the convergence of Q-learning algorithm.
        """
        heat_demand_discrete = [5, 20, 40]
        electricity_price_discrete = [20, 40, 60]
        s1 = [
            [
                self.max_supply_temp - self.discretization_T_step,
                self.max_supply_temp - self.discretization_T_step,
                self.discretization_mass_step * 5,
            ],
            RLParams.season_combinations[0],
            RLParams.time_of_the_day_combinations[0],
            [list(heat_demand_discrete)[2]],
            [list(electricity_price_discrete)[1]],
        ]
        a1 = RLParams.action_space[10]
        s2 = [
            [
                self.min_supply_temp + self.discretization_T_step,
                self.min_supply_temp + self.discretization_T_step,
                self.discretization_mass_step * 5,
            ],
            RLParams.season_combinations[2],
            RLParams.time_of_the_day_combinations[2],
            [list(heat_demand_discrete)[0]],
            [list(electricity_price_discrete)[1]],
        ]
        a2 = RLParams.action_space[0]
        s1, a1, s2, a2 = to_tuple(s1), to_tuple(a1), to_tuple(s2), to_tuple(a2)
        return s1, a1, s2, a2

    @property
    def get_num_elem(self):
        """
        Get number of water chunks.
        """
        return 2

    @property
    def get_next_state(self):
        """
        Get the external state space part of the next state.
        """
        return [
            [self.inlet_temp, self.outlet_temp, self.mass],
            self.future_season,
            self.future_time_of_the_day,
            self.future_demand,
            self.future_elec_price,
        ]
