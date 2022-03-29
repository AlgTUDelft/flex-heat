import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from tqdm import trange
from .RL_utils import *


now = datetime.now().strftime("%m_%d_%H_%M_%S")


class QAlgParam:
    """
    Q-learning algorithm parameters and common functions.
    """

    def __init__(
        self,
        number_of_episodes=1000,
        gamma=0.95,
        epsilon=0.8,
        alpha=0.8,
        epsilon_min=0.05,
        alpha_min=0.25,
    ):
        self.num_episodes: int = number_of_episodes
        self.gamma: float = gamma  # discounting factor of the future rewards
        self.epsilon: float = (
            epsilon  # exploration-explotation dillema (epsilon-greedy policy)
        )
        self._epsilon: float = epsilon
        self.alpha: float = (
            alpha  # 0.84  # learning rate (discretization causes stochasticity)
        )
        self._alpha: float = alpha
        self.epsilon_min: float = epsilon_min
        self.alpha_min: float = alpha_min
        self.decay_rate_e: float = (self.epsilon_min / self.epsilon) ** (
            1 / float(self.num_episodes)
        )
        self.decay_rate_a: float = (self.alpha_min / self.alpha) ** (
            1 / float(self.num_episodes)
        )

    def decay(self):
        """
        Decay of parameters epsilon (defines greedy policy)
        and alpha (defines learning rate).
        """
        self.epsilon *= self.decay_rate_e
        self.alpha *= self.decay_rate_a

    def reset(self):
        """
        Reset parameters epsilon and alpha on initial values.
        """
        self.epsilon = self._epsilon
        self.alpha = self._alpha


class QTable:
    """
    Initialization and creation of the Q-table.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        number_of_elements,
        min_supply_temp,
        q_low,
        q_high,
        q_init,
        state_spec,
    ):
        self.observation_space: list = observation_space
        self.action_space: list = action_space
        self.number_of_elements: int = number_of_elements
        self.min_supply_temp: float = min_supply_temp
        self.state_spec = state_spec
        self.q_low: float = q_low
        self.q_high: float = q_high
        self.q_init: float = q_init
        self._q: dict = self.initialize_q_table()

    def initialize_q_table(self) -> dict:
        """
        Initialization of the Q-table.
        """
        table = {}
        if self.state_spec:
            for i in trange(len(self.observation_space), desc="Building Q Table"):
                s = to_tuple(self.observation_space[i])
                for j in range(len(self.action_space)):
                    if (
                        self.observation_space[i][0]
                        == [self.min_supply_temp] * (self.number_of_elements + 1)
                        and self.observation_space[i][3] == [Data.max_heat_demand]
                        and self.action_space[j][0] <= Data.max_heat_demand / 2.5
                    ):
                        table[(s, self.action_space[j])] = self.q_low
                    elif (
                        self.observation_space[i][3] == [Data.min_heat_demand]
                        and self.observation_space[i][4] == [Data.max_electricity_price]
                        and (self.action_space[j] == self.action_space[0])
                    ):
                        table[(s, self.action_space[j])] = self.q_high
                    else:
                        table[(s, self.action_space[j])] = self.q_init
        else:
            for i in trange(len(self.observation_space), desc="Building Q Table"):
                s = to_tuple(self.observation_space[i])
                for j in range(len(self.action_space)):
                    table[(s, self.action_space[j])] = self.q_init
        return table

    @property
    def get_q_table(self):
        """
        Get Q-table.
        """
        return self._q


class QResult(ABC):
    """
    Define handling of results of Q-learning.
    """

    def __init__(
        self,
        average_factor,
        number_of_episodes,
        pipe_len,
        q_init,
        result_p,
    ):
        self.average_factor: int = average_factor
        self.number_of_episodes: int = number_of_episodes
        self.pipe_len: float = pipe_len
        self.q_init: float = q_init
        self.result_p: Path = result_p

    @abstractmethod
    def add_reward(self):
        pass

    @abstractmethod
    def add_episode(self):
        pass

    @abstractmethod
    def add_average(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def save_reward(self):
        pass

    def save_to_pickle(self, name, variable, i="all"):
        """
        Save variable to pickle file.
        """
        _name = (
            name
            + "_ep_"
            + i
            + "_L_"
            + str(self.pipe_len)
            + "_Q_"
            + str(self.q_init)
            + "_time_"
            + now
            + ".pickle"
        )
        with open(
            os.path.join(self.result_p, _name),
            "wb",
        ) as handle:
            pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)


class QResultTrain(QResult):
    """
    Define handling of results of Q-learning during training.
    """

    def __init__(
        self,
        average_factor=10,
        number_of_episodes=1000,
        pipe_len=4000,
        q_init=0,
        result_p=DataPaths.data,
    ):
        super().__init__(average_factor, number_of_episodes, pipe_len, q_init, result_p)
        self.r: float = 0
        self.r_hour_average: float = 0
        self.o_average = 0
        self.disc_error_T_average: float = 0
        self.disc_error_M_average: float = 0
        self.reward_collection: list = []
        self.reward_collection_hour: list = []
        self.observability_collection: list = []
        self.discretization_error_temperature: list = []
        self.discretization_error_mass: list = []
        self.Q_function_winter: list = []
        self.Q_function_summer: list = []

    def add_reward(self, reward):
        self.r += reward

    def add_episode(self, r_hour, o, disc_error_T, disc_error_M, q1, q2, j):
        self.r_hour_average += r_hour / (j + 1)
        self.o_average += (o / (j + 1)) * 100
        self.disc_error_T_average += (disc_error_T / (j + 1)) * 100
        self.disc_error_M_average += (disc_error_M / (j + 1)) * 100
        self.Q_function_winter.append(q1)
        self.Q_function_summer.append(q2)

    def add_average(self, i):
        print("Reward in episode {} is {}".format(i, self.r / self.average_factor))
        self.reward_collection.append(self.r / self.average_factor)
        self.reward_collection_hour.append(self.r_hour_average / self.average_factor)
        self.observability_collection.append(self.o_average / self.average_factor)
        self.discretization_error_temperature.append(
            self.disc_error_T_average / self.average_factor
        )
        self.discretization_error_mass.append(
            self.disc_error_M_average / self.average_factor
        )

    def reset(self):
        """
        Reset parameters on initial values.
        """
        self.r = 0
        self.r_hour_average = 0
        self.o_average = 0
        self.disc_error_T_average = 0
        self.disc_error_M_average = 0

    def save_reward(self):
        """
        Save the cumulative reward function during training.
        """
        print(
            "Average reward on training dataset is: {}".format(
                sum(self.reward_collection) / len(self.reward_collection)
            )
        )
        self.save_to_pickle(
            name="reward_function_training", variable=self.reward_collection
        )

    def save_q_function(self):
        """
        Save Q-function updates during training.
        """
        self.save_to_pickle(name="q_function_summer", variable=self.Q_function_summer)
        self.save_to_pickle(name="q_function_winter", variable=self.Q_function_winter)


class QResultTest(QResult):
    """
    Define handling of results of Q-learning during testing.
    """

    def __init__(
        self,
        average_factor=1,
        number_of_episodes=Data.test_len,
        pipe_len=4000,
        q_init=0,
        result_p=DataPaths.data,
    ):
        super().__init__(average_factor, number_of_episodes, pipe_len, q_init, result_p)
        # episode update
        self.o: float = 0
        self.operation_cost_episode: list = []
        self.heat_delivered_episode: list = []
        self.heat_demand_episode: list = []
        self.electricity_delivered_episode: list = []
        self.action_episode: list = []
        # average update
        self.r: float = 0
        self.r_hour: float = 0
        self.o_average: float = 0
        self.operation_cost_average: float = 0
        self.heat_delivered_average: float = 0
        self.heat_demand_average: float = 0
        self.electricity_delivered_average: float = 0
        self.condition_flag_max_in_supply_T_average: float = 0
        self.condition_flag_min_in_supply_T_average: float = 0
        self.condition_flag_min_out_return_T_average: float = 0
        self.condition_flag_max_mass_flow_average: float = 0
        # list update
        self.reward_collection: list = []
        self.reward_collection_hour: list = []
        self.observability_collection: list = []
        self.operation_cost: list = []
        self.heat_delivered: list = []
        self.heat_demand_plot: list = []
        self.electricity_delivered: list = []
        self.condition_flag_max_in_supply_T: list = []
        self.condition_flag_min_in_supply_T: list = []
        self.condition_flag_min_out_return_T: list = []
        self.condition_flag_max_mass_flow: list = []
        self.average_factor = average_factor
        self.number_of_episodes = number_of_episodes
        self.pipe_len = pipe_len
        self.q_init = q_init
        self.result_p = result_p

    def add_episode(
        self,
        observable,
        action,
        operation_cost,
        heat_delivered,
        electricity_delivered,
        heat_demand,
    ):
        self.o += observable
        self.action_episode.append(action)
        self.operation_cost_episode.append(operation_cost)
        self.heat_delivered_episode.append(heat_delivered)
        self.electricity_delivered_episode.append(electricity_delivered)
        self.heat_demand_episode.append(heat_demand)

    def add_reward(self, reward):
        self.r += reward

    def add_reward_hour(self, reward):
        self.r_hour += reward

    def add_condition_flags(self, max_in, min_in, min_out, max_mass):
        self.condition_flag_max_in_supply_T_average += max_in
        self.condition_flag_min_in_supply_T_average += min_in
        self.condition_flag_min_out_return_T_average += min_out
        self.condition_flag_max_mass_flow_average += max_mass

    def reset_episode(self):
        self.o = 0
        self.operation_cost_episode = []
        self.heat_delivered_episode = []
        self.heat_demand_episode = []
        self.electricity_delivered_episode = []
        self.action_episode = []

    def add_episode_sum(self, j):
        self.o_average += (self.o / (j + 1)) * 100
        self.operation_cost_average += sum(self.operation_cost_episode)
        self.heat_delivered_average += sum(self.heat_delivered_episode) / (j + 1)
        self.heat_demand_average += sum(self.heat_demand_episode) / (j + 1)
        self.electricity_delivered_average += sum(
            self.electricity_delivered_episode
        ) / (j + 1)

    def add_average(self, j):
        self.reward_collection.append(self.r / self.average_factor)
        self.reward_collection_hour.append(
            self.r_hour / ((j + 1) * self.average_factor)
        )
        self.observability_collection.append(self.o_average / self.average_factor)
        self.operation_cost.append(self.operation_cost_average / self.average_factor)
        self.heat_delivered.append(self.heat_delivered_average / self.average_factor)
        self.heat_demand_plot.append(self.heat_demand_average / self.average_factor)
        self.electricity_delivered.append(
            self.electricity_delivered_average / self.average_factor
        )
        self.condition_flag_max_in_supply_T.append(
            self.condition_flag_max_in_supply_T_average / self.average_factor
        )
        self.condition_flag_min_in_supply_T.append(
            self.condition_flag_min_in_supply_T_average / self.average_factor
        )
        self.condition_flag_min_out_return_T.append(
            self.condition_flag_min_out_return_T_average / self.average_factor
        )
        self.condition_flag_max_mass_flow.append(
            self.condition_flag_max_mass_flow_average / self.average_factor
        )

    def reset(self):
        """
        Reset parameters of episode on initial values.
        """
        self.r = 0
        self.r_hour = 0
        self.o_average = 0
        self.operation_cost_average = 0
        self.heat_delivered_average = 0
        self.heat_demand_average = 0
        self.electricity_delivered_average = 0
        self.condition_flag_max_in_supply_T_average = 0
        self.condition_flag_min_in_supply_T_average = 0
        self.condition_flag_min_out_return_T_average = 0
        self.condition_flag_max_mass_flow_average = 0

    def save_episode(self, i):
        """
        Saves action, objective function, delivered heat and delivered electricity
        for each step of each episode in testing dataset.
        """
        dict = {
            "action": self.action_episode,
            "objective_function": self.operation_cost_episode,
            "heat_delivered": self.heat_delivered_episode,
            "electricity_delivered": self.electricity_delivered_episode,
        }
        self.save_to_pickle(name="data_episode", variable=dict, i=str(i))

    def save_reward(self):
        """
        Saves reward signal during testing.
        """
        print(
            "Average reward on testing datest is: {}".format(
                sum(self.reward_collection) / len(self.reward_collection)
            )
        )
        self.save_to_pickle(
            name="reward_function_testing", variable=self.reward_collection
        )

    def save_operation_cost(self):
        """
        Saves sum of operation costs for each episode.
        """
        self.save_to_pickle(name="objective_function_sum", variable=self.operation_cost)

    def save_condition_flags(self):
        """
        Saves condition flags.
        """
        self.save_to_pickle(
            name="condition_flag_max_in_supply_T",
            variable=self.condition_flag_max_in_supply_T,
        )
        self.save_to_pickle(
            name="condition_flag_min_in_supply_T",
            variable=self.condition_flag_min_in_supply_T,
        )
        self.save_to_pickle(
            name="condition_flag_min_out_return_T",
            variable=self.condition_flag_min_out_return_T,
        )
        self.save_to_pickle(
            name="condition_flag_max_mass_flow",
            variable=self.condition_flag_max_mass_flow,
        )
