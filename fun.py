import theano
from theano import tensor as T
import numpy as np
from PublicKnowledge import PublicKnowledge
np.random.seed(0)

class Policy:
    def __init__(self):
        self.params = []
        self.inpdim = 1
        self.nonlinearity = lambda x:T.tanh(x)
        self.num_layers = 0
        self.build_network()

    def build_network(self):
        #inp of the network
        inp = T.matrix('inp', dtype = 'float32')
        reward = T.vector('reward', dtype = 'float32')
        var = T.vector('var', dtype = 'float32')
        fake_value = {
            inp: np.zeros(shape = (10,1)).astype(np.float32), 
            reward:np.zeros(shape =(10,)).astype(np.float32),
            var:np.zeros(shape =(10,)).astype(np.float32), 
        }

        def given(*inp):
            return {i: fake_value[i] for i in inp}

        def add_fc(x, inp, out, nonlinearity = lambda x:x):
            initW = np.random.normal(size = (inp,out))
            initb = np.zeros(shape = (out,))
            W = theano.shared(initW.astype('float32'))
            b = theano.shared(initb.astype('float32'))
            self.params += [W, b]
            x = T.dot(x, W) + b
            return nonlinearity(x)

        def calc_prob(gaussian, var):
            mean = gaussian[:, 0]
            sigma = T.exp( gaussian[:, 1] )

            prob = 1/(sigma * np.pi**0.5) * T.exp(-(var - mean)**2/(2 * sigma**2))
            return prob


        #forward used for inference
        x = inp
        for i in range(self.num_layers):
            x = add_fc(x, 1, 1, self.nonlinearity)
        gaussian = add_fc(x, 1, 2)

        self.forward = theano.function(inputs = [inp], outputs = gaussian)

        prob = calc_prob(gaussian, var)
        loss = T.mean( T.log(prob) * (reward- reward.mean()) )

        updates = []
        for i in self.params:
            grad_i = T.grad(loss, i)
            updates.append((i, i-grad_i*np.float32(0.1)))

        self.backward = theano.function(inputs = [inp, reward, var], outputs = loss, updates = updates)
        return

    def action(self, inp):
        inp = np.array(inp)
        inp = inp.reshape(inp.shape[0], -1).astype(np.float32)
        gaussian = self.forward(inp)
        mean = gaussian[:, 0]
        sigma = np.exp( gaussian[:, 0] )
        sampled_var = np.random.normal(size = mean.shape) * sigma + mean

        self.last_inp = inp
        self.last_var = sampled_var
        return sampled_var

    def learning(reward):
        assert self.last_var.shape == reward.shape
        return self.backward(self.last_inp, self.sampled_var)

class Player:
    def __init__(self, idx):
        self.idx = idx
        self.policy = Policy() 
        self.trainable = True

    def sample_valuation(self, num):
        valuation = [ [PublicKnowledge[self.idx].sample()] for i in range(num)]
        self.valuation = valuation
        return valuation

    def set_trainable(self, flag):
        self.trainable = flag

    def set_train(self):
        self.set_trainable(True)

    def set_freeze(self):
        self.set_trainable(False)

    def play(self, valuation):
        return self.policy.action(valuation)

    def utility(self, valuation, result):
        if result['bidder'] == self.idx:
            return valuation - result['price']
        else:
            return 0

    def inform(self, message):
        reward = map(self.utility, zip(self.valuation, message))
        if self.trainable:
            self.policy.learning(reward)
        return reward

class Mechanism:
    """
    Now only consider second price auction
    """
    def __init__(self):
        pass

    def decide(self, actions):
        for batches in range(len(actions)):
            if len(actions) == 1:
                result = {
                    'bider': 0,
                    'price': actions[0]
                }
            else:
                t = np.argsort(actions)
                result = {
                    'bider': t[0],
                    'price': actions[t[1]]
                }

class GameMaster:
    def __init__(self, players):
        self.mechanism = Mechanism()
        self.players = players

    def run(self, batchsize = 1):
        valuation = [i.sample_valuation(batchsize) for i in players]
        action = np.array( [i.play(j) for i, j in zip(players, valuation)] ).transpose(1, 0)
        result = self.mechanism.decide(action)
        reward = [i.inform(result) for i in players]
        return reward

if __name__ == '__main__':
    n = len(PublicKnowledge)
    players = [Player(i) for i in range(n)]
    gm = GameMaster(players)
    gm.run(10)
