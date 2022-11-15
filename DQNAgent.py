from brain import Brain
from experience_replay import UER
import numpy as np
import random

class Agent:
    def __init__(self,state_size,action_space) -> None:
        self.state_size = state_size
        self.action_space = action_space 
        self.gamma = 0.95
        self.brain = Brain(self.state_size, self.action_space)
        self.epsilon = 0.35
        self.step = 0
        self.memory = UER(1000)
        self.update_target_frequency = 1


    def policy(self,state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        else:
            return np.argmax(self.brain.predict(state))     

    def find_target(self, batch):
        batch_len = len(batch)

        states = np.array([o[0] for o in batch])
        states_ = np.array([o[3] for o in batch])

        states = np.array(states).reshape(batch_len,-1)

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_,target=True)

        x = np.zeros((batch_len, self.state_size))
        y = np.zeros((batch_len, self.action_space))
        errors = np.zeros(batch_len)

        for i in range(batch_len):
            o = batch[i]
            s = o[0]
            a = o[1]
            r = o[2]
            s_ = o[3]
            done = o[4]

            t = p[i]
            old_value = t[a]

            if done:
                t[a] = r
            else:
                t[a] = r + self.gamma * np.amax(p_[i])    

            x[i] = s
            y[i] = t 
            errors[i] = np.abs(t[a] - old_value)

        return [x,y,errors]

    def decay_epsilon(self):
        self.step +=1
        self.epsilon = self.epsilon * np.exp(-self.step)  

    def observe(self, sample):
        self.memory.store(sample)

    def train(self,batch_size):
        batch = self.memory.sample(batch_size)
        x,y,_ = self.find_target(batch)
        self.brain.train(x,y)

    def update_target_model(self):
        if(self.step % self.update_target_frequency == 0):
            self.brain.update_target_model()         














