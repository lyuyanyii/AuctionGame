import theano
from theano import tensor as T
import numpy as np
from PublicKnowledge import PublicKnowledge

class Policy:
    def __init__(self):
        self.params = []
        self.nonlinearity = lambda x: T.tanh(x)#lambda x:x*(x>0)
        self.num_layers = 3
        self.sigma = 0.1
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

        def add_fc(x, inp, out, nonlinearity = lambda x:x, initW = None, initb = None):
            if initW is None:
                initW = np.random.normal(size = (inp, out)) * 1./inp**0.5
            if initb is None:
                initb = np.zeros(shape = (out,))
            W = theano.shared(initW.astype('float32'))
            b = theano.shared(initb.astype('float32'))
            self.params += [W, b]
            x = T.dot(x, W) + b
            return nonlinearity(x)

        def calc_prob(gaussian, var):
            mean = gaussian[:, 0]
            if self.sigma is not None:
                sigma = self.sigma
            else:
                sigma = T.exp( gaussian[:, 1] )

            prob = 1/(sigma * np.pi**0.5) * T.exp(-(var - mean)**2/(2 * sigma**2))
            return prob


        #forward used for inference
        x = inp
        last = 1
        for i in range(self.num_layers):
            x = add_fc(x, last, 1, self.nonlinearity)
            last = 1
        gaussian = add_fc(x, last, 2)

        self.forward = theano.function(inputs = [inp], outputs = gaussian)

        prob = calc_prob(gaussian, var)
        loss = -T.mean( T.log(prob) * (reward- reward.mean()) )

        updates = []
        for i in self.params:
            grad_i = T.grad(loss, i)
            updates.append((i, i-grad_i*np.float32(0.001)))

        self.backward = theano.function(inputs = [inp, var, reward], outputs = loss, updates = updates)
        print('finish network building')
        return

    def action(self, inp):
        inp = np.array(inp)
        inp = inp.reshape(inp.shape[0], -1).astype(np.float32)
        gaussian = self.forward(inp)
        mean = gaussian[:, 0]
        if self.sigma is None: 
            sigma = np.exp( gaussian[:, 0] )
        else:
            sigma = self.sigma
        sampled_var = (np.random.normal(size = mean.shape) * sigma + mean).astype('float32')

        self.last_inp = inp
        self.last_var = sampled_var
        return sampled_var

    def learning(self, reward):
        reward = np.array(reward, dtype = 'float32')
        assert self.last_var.shape == reward.shape
        return self.backward(self.last_inp, self.last_var, reward)

    def dump(self):
        return [np.array(i.get_value()) for i in self.params]

    def loads(self, params):
        for a, b in zip(self.params, params):
            a.set_value(b) 

class Player:
    def __init__(self, idx):
        self.idx = idx
        self.policy = Policy() 
        self.trainable = True

    def sample_valuation(self, num):
        valuation = np.array([ [PublicKnowledge[self.idx].sample()] for i in range(num)])
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
        if result['bider'] == self.idx:
            return valuation - result['price']
        else:
            return 0

    def inform(self, message):
        reward = list( map(lambda a: self.utility(*a), zip(self.valuation.reshape(-1), message)) )
        if self.trainable:
            loss = self.policy.learning(reward)
            diff = np.abs(self.valuation.reshape(-1) - self.policy.last_var.reshape(-1)).mean()
            print(self.idx, diff)
        return reward

class FakePlayer(Player):
    def __init__(self, idx):
        self.idx = idx
        self.set_freeze() 
    def play(self, valuation):
        return valuation[:, 0]

class Mechanism:
    """
    Now only consider second price auction
    """
    def __init__(self):
        pass

    def decide(self, actions):
        results = []
        for batches, acts in enumerate(actions):
            assert len(acts)!=1
            t = np.argsort(-acts)
            results.append(  {
                'bider': t[0],
                'price': acts[t[1]]
            } )
        return results

    def inform(self, reward):
        return

class GameMaster:
    def __init__(self, players):
        self.mechanism = Mechanism()
        self.players = players

    def run(self, batchsize = 1):
        valuation = [i.sample_valuation(batchsize) for i in self.players]
        action = np.array( [i.play(j) for i, j in zip(self.players, valuation)] ).transpose(1, 0)
        result = self.mechanism.decide(action)
        reward = [i.inform(result) for i in self.players]
        return reward

def OneAgent():
    np.random.seed(0)
    n = len(PublicKnowledge)
    players = [Player(0)] + [FakePlayer(i) for i in range(1, n)]
    gm = GameMaster(players)
    for i in range(10000):
        gm.run(1000)
        print(players[0].policy.last_inp[0], players[0].policy.last_var[0])

def MultiAgent():
    np.random.seed(0)
    n = len(PublicKnowledge)
    players = [Player(i) for i in range(0, n)]
    gm = GameMaster(players)
    for i in range(100000):
        gm.run(1000)
#        print(players[0].policy.last_inp[0], players[0].policy.last_var[0])

if __name__ == '__main__':
    MultiAgent()
