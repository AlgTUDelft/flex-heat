from .abdollahi_2015 import Abdollahi2015
from .basic_strategy import BasicStrategy
from .li_2016 import Li_2016_day_ahead, Li_2016_intraday
from .rl.Q_utils import *
from .rl.q_agent import QAgent
from .upper_bound import UpperBound
from util.util import *


if __name__ == "__main__":
    # Q-learning agent
    number_of_episodes = 25000000
    number_of_episodes_test = Data.test_len
    average_factor = int(number_of_episodes / 100)
    average_factor_test = 1
    # environment parameters
    full_state: bool = True
    discretization_mass_step = 200000
    discretization_T_step = 10
    # agent parameters
    q_low = -100
    q_high = +500
    q_init = 0
    gamma = 0.95
    epsilon = 0.8
    alpha = 0.8
    epsilon_min = 0.05
    alpha_min = 0.25
    agent = QAgent(
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
        state_spec=full_state,
        number_of_episodes=number_of_episodes,
        q_low=q_low,
        q_high=q_high,
        q_init=q_init,
        gamma=gamma,
        epsilon=epsilon,
        alpha=alpha,
        epsilon_min=epsilon_min,
        alpha_min=alpha_min,
    )
    agent.train(average_factor=average_factor)
    agent.test(
        average_factor=average_factor_test,
    )
    del agent

    # Basic strategy
    constant_temp = 90
    basic_strategy = BasicStrategy(
        constant_temp=constant_temp,
        pipe_len=pipe_len,
        pipe_diameter=pipe_diameter,
        time_interval=time_interval,
        max_flow_speed=max_flow_speed,
        min_flow_speed=min_flow_speed,
        max_mass_flow=max_mass_flow,
        min_mass_flow=min_mass_flow,
        max_supply_temp=max_supply_temp,
        min_supply_temp=min_supply_temp,
        min_return_temp=min_return_temp,
        historical_t_supply=historical_t_supply,
        historical_t_return=historical_t_return,
    )
    basic_strategy.calculate_basic_strategy_cost()

    # Upper bound
    upper_lower_bound = 24  # parameter depends on the length of the pipe
    upper_bound = UpperBound(upper_lower_bound=upper_lower_bound, pipe_len=pipe_len)
    upper_bound.calculate_upper_bound_cost()

    # Abdollahi 2015
    abdollahi2015 = Abdollahi2015(4000)
    abdollahi2015.run()
    # Li 2016 (day-ahead and intraday electricity market)
    max_iter = 10
    time_limit = 360
    li_2016 = Li_2016_day_ahead(
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
    )
    li_2016.run()
