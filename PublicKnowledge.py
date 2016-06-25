"""
We store the public knowledge here
can be known by every one in the game
base of the game
"""
import numpy as np

N = 3

class UniformDisttribution:
    def __init__(self, l, r):
        self.l = l
        self.r = r
    
    def sample(self):
        return np.random.random() * (self.r - self.l) + self.l

"""
PublicKnowledge = [UniformDisttribution(0, 2)] + [UniformDisttribution(0, 1) for i in range(1, N)]

it's interesting for second price auction
"""
PublicKnowledge = [UniformDisttribution(0, 1) for i in range(N)]
