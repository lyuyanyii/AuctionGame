import theano
import numpy as np
import tqdm
from theano import tensor as T
from fun import Player, GameMaster, FakePlayer
from PublicKnowledge import PublicKnowledge

class MechanismLeaner:
    def __init__(self, player_num):
        self.params = []
        self.momentum = []
        self.nonlinearity = lambda x: T.tanh(x)
        self.num_layers = 0
        self.sigma = 0.1
        self.player_num = player_num
        self.trainable = True
        self.build_network()

    def build_network(self):
        #inp of the network
        #price of each person
        inp = T.matrix('inp', dtype = 'float32')
        reward = T.vector('reward', dtype = 'float32')
        var = T.matrix('var', dtype = 'float32')
        fake_value = {
            inp: np.zeros(shape = (10,self.player_num)).astype(np.float32), 
            reward:np.zeros(shape =(10,)).astype(np.float32),
            var:np.zeros(shape =(10,2)).astype(np.float32), 
        }

        def given(*inp):
            return {i: fake_value[i] for i in inp}

        def add_fc(x, inp_num, out_num, nonlinearity = lambda x:x, initW = None, initb = None):
            if initW is None:
                initW = np.random.normal(size = (inp_num, out_num)) * 1./inp_num**0.5
            if initb is None:
                assert 0
                initb = np.zeros(shape = (out_num,))
            W = theano.shared(initW.astype('float32'))
            b = theano.shared(initb.astype('float32'))
            self.params += [W, b]
            x = T.dot(x, W) + b.dimshuffle('x', 0)
            return nonlinearity(x)

        #forward used for inference
        x = inp
        last = self.player_num
        for i in range(self.num_layers):
            x = add_fc(x, last, 10, self.nonlinearity)
            last = 10

#        predict = add_fc(x, last, self.player_num + 1, initb = np.array([0]*self.player_num + [0.5]))
#        predict = add_fc(x, last, self.player_num + 1, 
#                         initW=np.array([[0,0,0,0], [0,0,0,1],[0,0,0,0]]),
#                         initb = np.array([10,0,0,0]))
        predict = add_fc(x, last, self.player_num + 1, 
                         initW=np.array([[0,0,0,1], [0,0,0,0],[0,0,0,0]]),
                         initb = np.array([10,0,0,0]))
        bider = T.nnet.softmax( predict[:, :self.player_num] )
        price =predict[:, self.player_num:]
        self.forward = theano.function(inputs = [inp], outputs = T.concatenate([bider, price], axis = 1))

        def calc_prob(gaussian, var):
            mean = gaussian[:, 0]
            if self.sigma is not None:
                sigma = self.sigma
            else:
                sigma = T.exp( gaussian[:, 1] )

            prob = 1/(sigma * np.pi**0.5) * T.exp(-(var - mean)**2/(2 * sigma**2))
            return prob

        prob1 = bider[T.arange(var.shape[0]), var[:, 0].astype('int64')]
        prob2 = calc_prob(price, var[:, 1])
        loss = -T.mean( (T.log(prob1) + T.log(prob2)) * (reward-reward.mean()))

        updates = []
        self.momentum = [theano.shared(i.get_value()*0).astype('float32') for i in self.params]
        for i, j in zip(self.params, self.momentum):
            grad_i = T.grad(loss, i)
            a = np.float32(0.9)
            b = np.float32(0.1)
            updates.append((i, i-(j*a+grad_i*b)*np.float32(0.1)))
            updates.append((j, j*a + grad_i *b))

        self.backward = theano.function(inputs = [inp, var, reward], outputs = loss, updates = updates)
        return

    def handle(self, action):
        labels = []
        sorted_action = []
        for i in action:
            labels.append(np.argsort(-i))
            sorted_action.append(-np.sort(-i))
        return labels, np.array(sorted_action).astype('float32')

    def decide(self, action):
        self.last_action = action

        labels, sorted_action = self.handle(action)
        predict = self.forward(sorted_action)

        bider_ = predict[:, :-1]
        def sampling(p):
            b = np.random.random()
            for i, a in enumerate(p):
                b-=a
                if b<=0:
                    return i
            return i
        bider = [sampling(i) for i in bider_]

        mean = predict[:, -1]
        sigma = self.sigma
        price = (np.random.normal(size = mean.shape) * sigma + mean).astype('float32')
        result =  [{'bider': label[a], 'price': b} for a, b, label in zip(bider, price, labels)]
        return result

    def training(self, action, var, reward):
        labels, sorted_action = self.handle(action)
        player_num = len(labels[0])
        new_var = var.copy()
        new_var[:,0] = [list(a).index(b) for a, b in zip(labels, var[:,0])]
        return self.backward(sorted_action, new_var, reward)

class MechanismTrainer:
    def __init__(self):
        self.stored_params = None
        self.history = []

        n = len(PublicKnowledge)
#        self.players = [Player(0)] + [FakePlayer(i) for i in range(1, n)]
        self.players = [Player(i) for i in range(0, n)]
        self.batchsize = 30
        self.mechanism = MechanismLeaner(n)

    def store_params(self):
        self.stored_params = [ i.policy.dump() for i in self.players]

    def reset_params(self):
        for a, b in zip(self.players, self.stored_params):
            a.policy.loads(b)

    def run(self, batchsize = 100, record = True):
        valuation = np.array([i.sample_valuation(batchsize) for i in self.players])
        action = np.array( [i.play(j) for i, j in zip(self.players, valuation)] ).transpose(1, 0)
        result = self.mechanism.decide(action)
        if record:
            prices = [[i['price']] for i in result]
            biders = [[i['bider']] for i in result]
            var = np.concatenate([biders, prices], axis = 1)
            self.history.append([action, var])

        reward = [i.inform(result) for i in self.players] #train the policy of each player
        diff = np.abs(valuation.reshape(-1) - action.transpose(1, 0).reshape(-1)).mean()
        return diff


    def get_data(self):
        self.store_params()
        self.history = []
        for i in range(20):
            diff = self.run(10)
        self.reset_params()
        actions = np.concatenate( [i[0] for i in self.history], axis = 0 )
        var = np.concatenate( [i[1] for i in self.history], axis = 0 )
        return actions, var, np.zeros(shape = (actions.shape[0],)) - diff

    def get_batches(self):
        inps = [[], [], []]
        print('sample data')
        for i in tqdm.tqdm(range(self.batchsize)):
            data = self.get_data()
            for j in range(len(inps)):
                inps[j].append(data[j])

        ans = [np.concatenate(i, axis = 0).astype(np.float32) for i in inps]
        return ans

    def descent(self):
        inps = self.get_batches()
        for i in tqdm.tqdm(range(30)):
            loss = self.mechanism.training(*inps)
            if loss<-0.1:
                break

        self.store_params()
        for i in range(20):
            diff = self.run(100, record = False)
        self.reset_params()

        return diff

def main():
    np.random.seed(0)
    trainer = MechanismTrainer()
    for i in tqdm.tqdm(range(20)):
        diff = trainer.run(1000, record = False)
    for batch_num in range(10000):
        diff = trainer.descent()
        print('batch: {}, reward: {}'.format(batch_num, diff))

if __name__ == '__main__':
    main()
