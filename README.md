# Traffic Light Control using MADDPG

### Code Structure

`./brain.py`: Contains neural network which each agent uses.

`./DQNAgent.py`: Deeq q-learning agent which controls the traffic light at a intersection.

`./experience_replay.py`: Memory buffer which stores (state,action,reward,next_state,done) tuple of previous time steps.

`./main.py`: Root file.

`./env/TrafficSim.cpp`: Traffic Simulation code. Road network is built from input and C cars are generated with random paths.

`./tests/input.txt`: Input to build the road network.


