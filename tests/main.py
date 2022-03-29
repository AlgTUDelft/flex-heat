from .sim_eval import MinlpEval
from util.util import *


if __name__ == "__main__":
    minlp_eval = MinlpEval(
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
    minlp_eval.run()
