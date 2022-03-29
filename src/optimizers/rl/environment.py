import numpy as np
from ...simulator.grid.cases.one_consumer import build_grid
from .state import State, FullState, AbstractState
from .RL_utils import *
from typing import Tuple
from util.util import *


class Environment(object):
    """
    Environment for te Q-learning agent training.
    Specification of the state space and reward creation for a given action.
    """

    def __init__(
        self,
        pipe_len=4000,
        pipe_diameter=0.5958,
        time_interval=1,
        max_flow_speed=3,
        min_flow_speed=0,
        max_supply_temp=110,
        min_supply_temp=70,
        min_return_temp=45,
        discretization_mass_step=200000,
        discretization_T_step=10,
        action_space=None,
        season=None,
        time_of_the_day=None,
        heat_demand=None,
        electricity_price=None,
        historical_t_supply=90,
        historical_t_return=50,
        state_spec=True,
    ):
        # list attributes
        if action_space is None:
            action_space = RLParams.action_space
        if season is None:
            season = Data.season_train[0]
        if time_of_the_day is None:
            time_of_the_day = Data.time_of_the_day_train[0]
        if heat_demand is None:
            heat_demand = Data.heat_demand_train[0]
        if electricity_price is None:
            electricity_price = Data.electricity_price_train[0]
        # pipe parameters
        self.pipe_len: float = pipe_len
        self.pipe_diameter: float = pipe_diameter
        self.time_interval = time_interval
        self.max_flow_speed: float = max_flow_speed
        self.min_flow_speed: float = min_flow_speed
        self.max_supply_temp: float = max_supply_temp
        self.min_supply_temp: float = min_supply_temp
        self.min_return_temp: float = min_return_temp
        self.historical_t_supply: float = historical_t_supply
        self.historical_t_return: float = historical_t_return
        self.discretization_mass_step: int = discretization_mass_step
        self.discretization_T_step: int = discretization_T_step
        # external part
        self.season: list = season
        self.time_of_the_day: list = time_of_the_day
        self.heat_demand: list = heat_demand
        self.electricity_price: list = electricity_price
        # state
        self.state_spec = state_spec
        if state_spec:
            self.state: State = FullState(
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
                future_season=self.season[0],
                future_time_of_the_day=self.time_of_the_day[0],
                future_demand=self.heat_demand[: RLParams.future_heat_demand],
                future_elec_price=self.electricity_price[
                    : RLParams.future_electricity_price
                ],
            )
        else:
            self.state: State = AbstractState(
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
                future_season=self.season[0],
                future_time_of_the_day=self.time_of_the_day[0],
                future_demand=self.heat_demand[: RLParams.future_heat_demand],
                future_elec_price=self.electricity_price[
                    : RLParams.future_electricity_price
                ],
            )
        self.max_M = self.state.get_mass
        self._number_of_elements = self.state.get_num_elem
        self._action_space = action_space
        self._observation_space = self.state.create_observation_space()
        # reward
        self.operation_cost_up = operation_cost_upper_bound()
        self.operation_cost_down = operation_cost_lower_bound()
        self.min_heat_delivered = 0
        self.max_heat_delivered = heat_delivered_upper_bound(
            self.max_flow_speed, self.pipe_diameter, self.max_supply_temp
        )
        self.min_temperature_violation = self.discretization_T_step / 2
        self.max_temperature_violation = self.min_supply_temp - self.min_return_temp
        self.min_mass_violation = self.discretization_mass_step / 2
        self.max_mass_violation = 2 * self.max_M
        # grid
        self.grid = build_grid(
            np.array(self.heat_demand),
            self.pipe_len,
            self.pipe_diameter,
            self.historical_t_supply,
            self.historical_t_return,
            self.max_flow_speed,
            self.min_flow_speed,
            self.max_supply_temp,
            self.min_supply_temp,
            self.min_return_temp,
        )
        self.grid.clear()
        self.condition_flags = np.zeros(EnvParams.number_condition_flags, dtype=int)

    @property
    def get_observation_space(self):
        """
        Get all combinations of water chunks'
        temperatures and masses that state space consists of.
        """
        return self._observation_space

    @property
    def get_action_space(self):
        """
        Get action space.
        """
        return self._action_space

    @property
    def get_num_elem(self):
        """
        Get number of water chunks.
        """
        return self._number_of_elements

    @property
    def get_result_p(self):
        """
        Get Path of the result (for a specific state))
        """
        return self.state.get_result_p

    def operation_cost_reshape(self, cost, electricity_sale) -> float:
        """
        Reshape cost function sub-reward to fit upper and lower bound and gradient hyper-parameters.
        """
        return RewardFunParams.operation_cost_reshape_up - (
            RewardFunParams.operation_cost_reshape_up
            - RewardFunParams.operation_cost_reshape_down
        ) * pow(
            (
                abs(cost + electricity_sale - self.operation_cost_up)
                / abs(self.operation_cost_down - self.operation_cost_up)
            ),
            RewardFunParams.operation_cost_gradient,
        )

    def underdeliver_heat_reshape(self, heat_delivered, i) -> float:
        """
        Reshape underdelivered heat sub-reward to fit hyper-parameters.
        """
        return RewardFunParams.underdeliver_heat_reshape_up - (
            RewardFunParams.underdeliver_heat_reshape_up
            - RewardFunParams.underdeliver_heat_reshape_down
        ) * pow(
            abs(heat_delivered - self.heat_demand[i])
            / abs(self.min_heat_delivered - Data.max_heat_demand),
            RewardFunParams.underdeliver_heat_gradient,
        )

    def overdeliver_heat_reshape(self, heat_delivered, i) -> float:
        """
        Reshape overdelivered heat sub-reward to fit hyper-parameters.
        """
        return RewardFunParams.overdeliver_heat_reshape_up - (
            RewardFunParams.overdeliver_heat_reshape_up
            - RewardFunParams.overdeliver_heat_reshape_down
        ) * pow(
            abs(heat_delivered - self.heat_demand[i])
            / abs(self.max_heat_delivered - Data.min_heat_demand),
            RewardFunParams.overdeliver_heat_gradient,
        )

    def violation_temperature_reshape(self, temperature_violation) -> float:
        """
        Reshape maximum and minimum temperature sub-reward to fit hyper-parameters.
        """
        return RewardFunParams.constraint_violation_up - (
            RewardFunParams.constraint_violation_up
            - RewardFunParams.constraint_violation_down
        ) * pow(
            (
                (temperature_violation - self.min_temperature_violation)
                / (self.max_temperature_violation - self.min_temperature_violation)
            ),
            RewardFunParams.constraint_violation_gradient,
        )

    def violation_mass_flow_reshape(self, mass_flow_violation) -> float:
        """
        Reshape maximum mass flow sub-reward to fit hyper-parameters.
        """
        return RewardFunParams.constraint_violation_up - (
            RewardFunParams.constraint_violation_up
            - RewardFunParams.constraint_violation_down
        ) * pow(
            (
                (mass_flow_violation - self.min_mass_violation)
                / (self.max_mass_violation - self.min_mass_violation)
            ),
            RewardFunParams.constraint_violation_gradient,
        )

    def reward_fun(
        self,
        cost,
        electricity_production,
        heat_delivered,
        i,
        condition_flags,
        supply_in_temp,
        return_in_temp,
        M,
    ) -> float:
        """
        Create the reward function as the linear sum of sub-rewards.
        """
        operation_cost = self.operation_cost_reshape(
            cost, self.electricity_price[i] * electricity_production
        )
        underdeliver_heat = (
            self.underdeliver_heat_reshape(heat_delivered, i)
            if heat_delivered < self.heat_demand[i] - EnvParams.heat_demand_tolerance
            else 0
        )
        overdeliver_heat = (
            self.overdeliver_heat_reshape(heat_delivered, i)
            if heat_delivered > self.heat_demand[i] + EnvParams.heat_demand_tolerance
            else 0
        )
        supply_in_temp_up = (
            self.violation_temperature_reshape(
                abs(supply_in_temp - self.max_supply_temp)
            )
            if condition_flags[0]
            else 0
        )
        supply_in_temp_down = (
            self.violation_temperature_reshape(
                abs(supply_in_temp - self.min_supply_temp)
            )
            if condition_flags[1]
            else 0
        )
        return_in_temp_down = (
            self.violation_temperature_reshape(
                abs(return_in_temp - self.min_return_temp)
            )
            if condition_flags[2]
            else 0
        )
        mass_flow_up = (
            self.violation_mass_flow_reshape(abs(M - self.max_M))
            if condition_flags[3]
            else 0
        )
        return (
            operation_cost
            + underdeliver_heat
            + overdeliver_heat
            + supply_in_temp_up
            + supply_in_temp_down
            + return_in_temp_down
            + mass_flow_up
        )

    def update_condition_flags(self, condition_flags) -> None:
        """
        Update condition flags based on the simulator feedback.
        """
        for i in range(EnvParams.number_condition_flags):
            self.condition_flags[i] += int(condition_flags[i])

    def step(
        self, heat_electricity_production, i
    ) -> Tuple[list, float, float, float, float, list, bool, float, float]:
        """
        Based on the action taken by the Q-learning agent,
        simulator gives observations of temperatures and mass flows.
        They are used for the state construction, and reward signal creation.
        """
        heat_production = heat_electricity_production[0] / self.time_interval
        electricity_production = heat_electricity_production[1] / self.time_interval

        cost = -a[0] * heat_production - a[1] * electricity_production

        heat_production_in_W = heat_production * 10 ** 6
        (
            in_temp,
            out_temp,
            mass_flow,
            heat_delivered,
            pipe_condition,
            _,
        ) = self.grid.solve_one_step(heat=[heat_production_in_W])
        # print(pipe_condition)
        supply_in_temp = in_temp[0]
        supply_out_temp = out_temp[0]
        return_in_temp = in_temp[1]
        # return_out_temp = out_temp[1]
        mass_flow = mass_flow[0]
        M = mass_flow * self.time_interval * HOUR_TO_SEC
        heat_delivered = heat_delivered / (10 ** 6) * self.time_interval
        supply_in_temp_disc = pipe_disc(supply_in_temp, self.discretization_T_step)
        supply_out_temp_disc = pipe_disc(supply_out_temp, self.discretization_T_step)
        return_in_temp_disc = pipe_disc(return_in_temp, self.discretization_T_step)
        M_disc = pipe_disc(M, self.discretization_mass_step)

        condition_flag_max_in_supply_T = (
            supply_in_temp > self.max_supply_temp + self.min_temperature_violation
        )
        condition_flag_min_in_supply_T = (
            supply_in_temp < self.min_supply_temp - self.min_temperature_violation
        )
        condition_flag_min_in_return_T = (
            return_in_temp < self.min_return_temp - self.min_temperature_violation
        )
        condition_flag_max_mass_flow = M > self.max_M + self.min_mass_violation
        condition_flags = [
            condition_flag_max_in_supply_T,
            condition_flag_min_in_supply_T,
            condition_flag_min_in_return_T,
            condition_flag_max_mass_flow,
        ]
        condition_flag = any(condition_flags)
        self.update_condition_flags(condition_flags)

        discretization_error_temperature = (
            max(supply_in_temp_disc, supply_in_temp)
            - min(supply_in_temp_disc, supply_in_temp)
        ) / supply_in_temp

        discretization_error_mass = (max(M_disc, M) - min(M_disc, M)) / M
        season_update = self.season[
            i + 1 if i <= TIME_HORIZON - 2 else TIME_HORIZON - 1
        ]
        time_of_the_day_update = self.time_of_the_day[
            i + 1 if i <= TIME_HORIZON - 2 else TIME_HORIZON - 1
        ]
        demand_update = self.heat_demand[
            i + RLParams.future_heat_demand
            if i <= TIME_HORIZON - RLParams.future_heat_demand - 1
            else TIME_HORIZON - 1
        ]
        electricity_price_update = self.electricity_price[
            i + RLParams.future_electricity_price
            if i <= TIME_HORIZON - RLParams.future_electricity_price - 1
            else TIME_HORIZON - 1
        ]
        R = self.state.state_transition(
            supply_in_temp=supply_in_temp_disc,
            supply_out_temp=supply_out_temp_disc,
            return_in_temp=return_in_temp_disc,
            M=M_disc,
            season_update=season_update,
            time_of_the_day_update=time_of_the_day_update,
            demand_update=demand_update,
            elec_price_update=electricity_price_update,
        )
        O = (max(R, heat_delivered) - min(R, heat_delivered)) / heat_delivered

        reward = self.reward_fun(
            cost,
            electricity_production,
            float(heat_delivered),
            i,
            condition_flags,
            supply_in_temp,
            return_in_temp,
            M,
        )
        """
        return (
            [self.state.value, self.state.future_demand, self.state.return_T_in],
            O,
            reward,
            condition_flag,
            discretization_error_temperature,
            discretization_error_mass
        )  # in degree celsuis, kg/s, MWh
        """
        return (
            self.state.get_next_state,
            O,
            reward,
            cost + electricity_production * self.electricity_price[i],
            heat_delivered,
            condition_flags,
            condition_flag,
            discretization_error_temperature,
            discretization_error_mass,
        )

    def update_day(self, i, train) -> None:
        """
        Update season, time of the day, heat demand and electricity price for a new episode.
        """
        self.season = Data.season_train[i] if train else Data.season_test[i]
        self.time_of_the_day = (
            Data.time_of_the_day_train[i] if train else Data.time_of_the_day_test[i]
        )
        self.heat_demand = (
            Data.heat_demand_train[i] if train else Data.heat_demand_test[i]
        )
        self.electricity_price = (
            Data.electricity_price_train[i] if train else Data.electricity_price_test[i]
        )
        for consumer in self.grid.consumers:
            consumer.demand = np.array(self.heat_demand) * 10 ** 6

    def reset(self) -> list:
        """
        Reset the grid object on its beginner's state.
        (same as constructor, filled with one chunk of water of dimension of pipe's capacity
        with temperature equal to the historical supply temperature)
        """
        self.grid.clear()
        return self.state.reset(
            self.season[0],
            self.time_of_the_day[0],
            self.heat_demand[: RLParams.future_heat_demand],
            self.electricity_price[: RLParams.future_electricity_price],
        )

    def convergence_pair(self):
        """
        Get the state, action pair for which we inspect Q-function convergence.
        """
        return self.state.convergence_pair()
