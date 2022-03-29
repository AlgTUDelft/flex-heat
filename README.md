# flex_heat

This repository contains implementations of five algorithms: upper bound, basic control strategy, linear program, mixed-integer nonlinear program and reinforcement learning for exploring pipeline energy storage for combined heat and power (CHP) economic dispatch. 

## Modules
- ./data: Data module contains the data used in experiments. These data include heat demand, electricity price, season and time of the day. Data is divided on training and testing dataset.
- ./src:
   - ../data_processing: Creation of training and testing datasets by parsing the data into four separate datasets: heat demand, electricity price, season, and time of the day, and data discretization.
   - ../optimizers: The implementation of five optimization algorithms:
       - .../upper_bound.py: Creation of the upper bound on profit for CHP economic dispatch.
       - .../basic_strategy.py: Calculation of profit by following the control based on constant inlet temperature of 90°C.
       - .../abdollahi_2015.py: Linear program of CHP economic dispatch without grid dynamics.
       - .../li_2016.py: Implementation of mixed-integer nonlinear program for CHP economic dispatch. This implementation constains operation on day-ahead and real-                time electricity market.
       - .../rl:
          - ..../q_agent.py: Implementation of Q-learning agent. Updates to Q-matrix based on environment information, and training and testing of Q-agent.
          - ..../environment.py: Environment of Q-learning agent. This environment generates the state, and reward based on observations from simulation                      environment.
          - ..../state.py: Two state spaces of Q-learning agent: full state space and partial state space.
   - ../simulator: The simulator is the property of collaborator company Flex-Technologies. The code will be made available with the technical report (manuscript in preparation).
- ./tests: Feasibility evaluation of outputs of optimizers by the simulator.
- ./results: This module stores experimental results of algorithms and simulator evaluation.
- ./plots: Replication of experimental plots. The code reads experimental results from ./results module, and generates plots related to algorithms' profit, feasibility, and optimization stability.
- ./util: util package contains mathematical constants, thermodynamics physics constants and district heating network parameters used in all experiments.

## Installation
Mathematical optimizers use [SCIP optimization suite](https://www.scipopt.org/). To install the util package:
```
python setup.py install
```

## Usage
To create and solve mathematical models, and to train and test Q-learning algorithm:
```
python -m src.optimizers.main
```
To evaluate outputs of the optimizers by the simulator:
```
python -m tests.main
```
To replicate experimental plots:
```
python -m plots.main
```
