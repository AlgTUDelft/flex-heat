import math
import random
from .environment import Environment
from .Q_utils import *
from util.util import *


class QAgent:
    """
    Q-learning agent.
    """

    def __init__(
        self,
        pipe_len,
        pipe_diameter,
        time_interval,
        max_flow_speed,
        min_flow_speed,
        max_supply_temp,
        min_supply_temp,
        min_return_temp,
        discretization_mass_step,
        discretization_T_step,
        historical_t_supply,
        historical_t_return,
        state_spec,
        number_of_episodes,
        q_low,
        q_high,
        q_init,
        gamma,
        epsilon,
        alpha,
        epsilon_min,
        alpha_min,
    ):
        self.pipe_len = pipe_len
        self.environment = Environment(
            pipe_len=pipe_len,
            pipe_diameter=pipe_diameter,
            time_interval=time_interval,
            max_flow_speed=max_flow_speed,
            min_flow_speed=min_flow_speed,
            max_supply_temp=max_supply_temp,
            min_supply_temp=min_supply_temp,
            min_return_temp=min_return_temp,
            discretization_mass_step=discretization_mass_step,
            discretization_T_step=discretization_T_step,
            historical_t_supply=historical_t_supply,
            historical_t_return=historical_t_return,
            state_spec=state_spec,
        )
        self.number_of_episodes: int = number_of_episodes
        self.time_interval: int = time_interval
        self.observation_space: list = self.environment.get_observation_space
        self.action_space: list = self.environment.get_action_space
        self.number_of_elements = self.environment.get_num_elem
        self.q_param: QAlgParam = QAlgParam(
            number_of_episodes=number_of_episodes,
            gamma=gamma,
            epsilon=epsilon,
            alpha=alpha,
            epsilon_min=epsilon_min,
            alpha_min=alpha_min,
        )
        self.q_table: QTable = QTable(
            observation_space=self.observation_space,
            action_space=self.action_space,
            number_of_elements=self.number_of_elements,
            min_supply_temp=min_supply_temp,
            q_low=q_low,
            q_high=q_high,
            q_init=q_init,
            state_spec=state_spec,
        )
        self.epsilon = epsilon
        self.q = self.q_table.get_q_table
        self.q_init = q_init
        self.timer = Timer()

    def epsilon_greedy_policy(self, state) -> list:
        """
        Choose the next action for a given state based on the epsilon greedy strategy.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.action_space)
        else:
            return self.action_space[
                max(
                    list(range(len(self.action_space))),
                    key=lambda x: self.q[(state, self.action_space[x])],
                )
            ]

    def update_q_table(
        self, prev_state, action, reward, next_state, condition_flag
    ) -> None:
        """
        Update of the Q-table based on the taken action and the next state, and reward.
        """
        if condition_flag:
            qa = 0
        else:
            qa = max(
                [
                    self.q[next_state, self.action_space[a]]
                    for a in range(len(self.action_space))
                ]
            )
        self.q[prev_state, action] += self.q_param.alpha * (
            reward + self.q_param.gamma * qa - self.q[prev_state, action]
        )

    def decay(self):
        """
        Decay of epsilon (greedy parameter) and alpha (learning rate).
        Decay of these two parameters is increased with the training.
        """
        self.q_param.decay()

    def get_q_convergence(self, s1, a1, s2, a2):
        """
        Get values of the Q-table for two specified state, action pairs.
        """
        return self.q[s1, a1], self.q[s2, a2]

    def train(self, average_factor):
        """
        Training Q-learning agent.
        """
        s1, a1, s2, a2 = self.environment.convergence_pair()
        result_train = QResultTrain(
            average_factor=average_factor,
            number_of_episodes=self.number_of_episodes,
            pipe_len=self.pipe_len,
            q_init=self.q_init,
            result_p=self.environment.get_result_p,
        )

        for i in range(1, self.number_of_episodes + 1):
            self.environment.update_day(
                (i - 1) - Data.train_len * int(math.floor((i - 1) / Data.train_len)),
                train=True,
            )
            prev_state = to_tuple(self.environment.reset())
            r_hour, o, disc_error_T, disc_error_M = (0, 0, 0, 0)
            for j in range(TIME_HORIZON):
                action = self.epsilon_greedy_policy(prev_state)
                (
                    next_state,
                    observable,
                    reward,
                    _,
                    _,
                    _,
                    condition_flag,
                    error_T,
                    error_M,
                ) = self.environment.step(action, j)
                next_state = to_tuple(next_state)
                self.update_q_table(
                    prev_state, action, reward, next_state, condition_flag
                )
                prev_state = next_state
                result_train.add_reward(reward)
                o += observable
                disc_error_T += error_T
                disc_error_M += error_M
                if condition_flag:
                    break
                r_hour += reward
            self.decay()
            q1, q2 = self.get_q_convergence(s1, a1, s2, a2)
            result_train.add_episode(
                r_hour=r_hour,
                o=o,
                disc_error_T=disc_error_T,
                disc_error_M=disc_error_M,
                q1=q1,
                q2=q2,
                j=j,
            )
            if i % average_factor == 0:
                result_train.add_average(i=i)
                result_train.reset()

        # result_train.save_reward()
        # result_train.save_q_function()

    def test(self, average_factor):
        """
        Evaluating Q-learning agent after training.
        """
        result_test = QResultTest(
            average_factor=average_factor,
            number_of_episodes=self.number_of_episodes,
            pipe_len=self.pipe_len,
            q_init=self.q_init,
            result_p=self.environment.get_result_p,
        )
        for i in range(1, self.number_of_episodes + 1):
            self.environment.update_day(
                (i - 1) - Data.test_len * int(math.floor((i - 1) / Data.test_len)),
                train=False,
            )
            prev_state = to_tuple(self.environment.reset())
            result_test.reset_episode()
            for j in range(TIME_HORIZON):
                action = self.action_space[
                    max(
                        list(range(len(self.action_space))),
                        key=lambda x: self.q[(prev_state, self.action_space[x])],
                    )
                ]
                (
                    next_state,
                    observable,
                    reward,
                    operation_cost,
                    heat_delivered,
                    condition_flags,
                    condition_flag,
                    error_T,
                    error_M,
                ) = self.environment.step(action, j)
                next_state = to_tuple(next_state)
                # print(next_state, action, prev_state)
                prev_state = next_state
                result_test.add_reward(reward=reward)
                result_test.add_episode(
                    observable=observable,
                    action=action,
                    operation_cost=operation_cost,
                    heat_delivered=heat_delivered,
                    electricity_delivered=action[1] / self.time_interval,
                    heat_demand=Data.heat_demand_test[
                        (i - 1)
                        - Data.test_len * int(math.floor((i - 1) / Data.test_len))
                    ][j],
                )
                if condition_flag:
                    result_test.add_condition_flags(
                        max_in=round(
                            condition_flags[0] * ((TIME_HORIZON - j) / TIME_HORIZON), 2
                        ),
                        min_in=round(
                            condition_flags[1] * ((TIME_HORIZON - j) / TIME_HORIZON), 2
                        ),
                        min_out=round(
                            condition_flags[2] * ((TIME_HORIZON - j) / TIME_HORIZON), 2
                        ),
                        max_mass=round(
                            condition_flags[3] * ((TIME_HORIZON - j) / TIME_HORIZON), 2
                        ),
                    )
                    print("Agent entered unallowed state, and violated constraints")
                    break
                result_test.add_reward_hour(reward=reward)
            # result_test.save_episode(i=i)
            # result_test.add_episode_sum(j=j)
            if i % average_factor == 0:
                result_test.add_average(j=j)
                result_test.reset()
            # result_test.save_reward()
            # result_test.save_operation_cost()
            # result_test.save_condition_flags()
        print("Elapsed time (min): {}".format(self.timer.get_time_minutes(), 2))
