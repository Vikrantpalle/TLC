from collections import deque
import random

class UER:
    def __init__(self,capacity) -> None:
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def store(self,sample):
        self.memory.append(sample)

    def sample(self,n):
        n = min(n,len(self.memory))
        sample_batch = random.sample(self.memory, n)

        return sample_batch       