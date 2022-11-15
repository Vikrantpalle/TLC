from flame import TrafficSim
from DQNAgent import Agent
import numpy as np
import time

class Environment:
    def __init__(self,sim,agents_list) -> None:
        self.env = sim
        self.agents_list = agents_list
        self.episodes = 20
        self.n_cars = 10
        self.max_ts = 1000
        self.filling_steps = 10
        self.steps_b_update = 5

    def run(self):
        total_step = 0
        rewards_list = []
        timesteps_list = []
        max_score = -10000


        for episode_num in range(self.episodes):
            state = self.env.reset()

            state = np.array(state)
            state = state.reshape(1,-1)

            done = False
            reward_all = 0
            time_step = 0
            while not done and time_step < self.max_ts:
                actions = []
                for agent in self.agents_list:
                    actions.append(agent.policy(state))
                actions = np.array(actions)
                actions = actions.reshape(-1)   
                next_state, reward, done = self.env.step(actions,1)  
                
                next_state = np.array(next_state)
                next_state = next_state.reshape(1,-1)

                for idx, agent in enumerate(self.agents_list):
                    agent.observe((state,actions[idx],reward,next_state,done))  
                    if total_step >= self.filling_steps:
                        agent.decay_epsilon()
                        if(time_step % self.steps_b_update):
                            agent.train(10)
                        agent.update_target_model()

                total_step+=1
                time_step+=1
                state = next_state
                reward_all+=reward

            rewards_list.append(reward_all)
            timesteps_list.append(time_step)

            print(f'Episode {episode_num}, Score {reward_all}, Duration {time_step}, Goal {done}')











         





if __name__ == "__main__":
    with open('./tests/input.txt','r') as f:
            lines = [line.rstrip() for line in f.readlines()]
            # No of _,nodes,streets,cars
            I,S,C = list(map(int,lines[0].split(' ')))
            lines.pop(0)
            sim = TrafficSim(I,S,C)
            # streets: src, dest, name
            for i in range(S):
                src,dest,name = lines[0].split(' ')
                src  =int(src)
                dest = int(dest)
                sim.addStreet(src,dest,name)
                lines.pop(0)
    agents_list = []        
    for i in range(I):
        print(S, sim.action_space(i))
        agents_list.append(Agent(S, sim.action_space(i)))    
    env = Environment(sim,agents_list)    
    env.run()







    
    
