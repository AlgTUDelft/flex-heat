# go to the environment you want to install the package in,
# and run python setup.py install
import os
import numpy as np
import pickle

# length of the episode
TIME_HORIZON = 24  # [h]

# math constants
PI = 3.14
MW_W = 1000000
HOUR_TO_SEC = 3600

# thermodynamics physics constants
CONVECTIVE_HEAT_TRANSFER_COEFF = 0.0005  # [MW/(m^2*C)]
WATER_DENSITY = 963  # [kg/m^3]
C = 0.004182  # [MJ/(kg*C)]
# source: https://www.engineeringtoolbox.com/overall-heat-transfer-coefficient-d_434.html
HEAT_TRANSFER_COEFF = np.array([0.000000735])  # [MW/(m*C)]
# source: https://en.wikipedia.org/wiki/Darcy-Weisbach_equation, https://www.pipeflow.com/pipe-pressure-drop-calculations/pipe-friction-factors
Re = 1000  # Reynolds number for laminar flow and circular pipe
FRICTION_COEFFICIENT = 64 / Re

# CHP parameters
a = [8.1817, 38.1805]  # [e/MWh]
EXTREME_POINTS = [[0, 10], [10, 5], [70, 35], [0, 50]]  # [MW]
# environment temperature
T_env = 10  # [C]

# condition flags
condition_flags = {
    "heat_underdelivered": "Underdelivered heat",
    "max_in_supply_T": "Maximum inlet temperature",
    "max_mass_flow": "Maximum mass flow",
    "min_in_supply_T": "Minimum inlet temperature",
    "min_out_return_T": "Minimum outlet temperature",
}

# district heating grid parameters (inputs)
pipe_len = 4000  # [m]
pipe_diameter = 0.5958  # [m]
time_interval = 1  # [h]
max_flow_speed = 3  # [m/s]
min_flow_speed = 0  # [m/s]
max_mass_flow = max_flow_speed * WATER_DENSITY * PI * (pipe_diameter / 2) ** 2  # [kg/s]
min_mass_flow = 0
max_supply_temp = 110  # [C]
min_supply_temp = 70  # [C]
max_return_temp = 80  # [C]
min_return_temp = 45  # [C]
historical_t_supply = 90  # [C]
historical_t_return = 50  # [C]
p_hes = 100000  # [Pa]
water_pump_efficiency = 0.8
max_power_consumption_water_pump = 20  # [MW]
min_power_consumption_water_pump = 0


def save_to_pickle(
    data_path_store, variable, variable_name, pipe_len, now, ep="all"
) -> None:
    """
    Save variable to pickle file under specified path.
    """
    name = (
        variable_name
        + "_ep_"
        + str(ep)
        + "_L_"
        + str(pipe_len)
        + "_time_"
        + now
        + ".pickle"
    )
    with open(
        os.path.join(data_path_store, name),
        "wb",
    ) as handle:
        pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)
