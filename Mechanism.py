import theano
from theano import tensor as T
from fun import Player, GameMaster
from PublicKnowledge import PublicKnowledge

class MechanismLeaner:
    def __init__(self, player_num):
        self.params = []
        self.nonlinearity = lambda x: T.tanh(x)
        self.num_layers = 3
        self.sigma = 0.01
        self.player_num = player_num
        self.trainable = True
        self.build_network()

    def build_network(self):
        #inp of the network
        #price of each person
        inp = T.matrix('inp', dtype = 'float32')
        reward = T.vector('reward', dtype = 'float32')
        var = T.vector('var', dtype = 'float32')
        fake_value = {
            inp: np.zeros(shape = (10,self.player_num)).astype(np.float32), 
            reward:np.zeros(shape =(10,)).astype(np.float32),
            var:np.zeros(shape =(10,2)).astype(np.float32), 
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

        #forward used for inference
        x = inp
        last = self.num_layers
        for i in range(self.player_num):
            x = add_fc(x, last, 10, self.nonlinearity)
            last = 10

        predict = add_fc(x, last, self.player_num + 1)
        bider = T.nnet.softmax( predict[:, :self.player_num] )

        price = predict[:, self.player_num:]
        self.forward = theano.function(inputs = [inp], outputs = predict)

        def calc_prob(gaussian, var):
            mean = gaussian[:, 0]
            if self.sigma is not None:
                sigma = self.sigma
            else:
                sigma = T.exp( gaussian[:, 1] )

            prob = 1/(sigma * np.pi**0.5) * T.exp(-(var - mean)**2/(2 * sigma**2))
            return prob

        prob1 = bider[T.arange(var.shape[0]), var[:, 0]]
        prob2 = calc_prob(gaussian, var[:, 1])
        loss = -T.mean( (T.log(prob1) + T.log(prob2)) * (reward- reward.mean()) )

        updates = []
        for i in self.params:
            grad_i = T.grad(loss, i)
            updates.append((i, i-grad_i*np.float32(0.01)))

        self.backward = theano.function(inputs = [inp, var, reward], outputs = loss, updates = updates)
        return

    def decide(self, action):
        self.last_action = action
        predict = self.forward(action)

        bider_ = predict[:, 0]
        def sampling(p):
            b = np.random.random()
            for i, a in enumerate(p):
                b-=a
                if b>=1:
                    return i
            return i
        bider = [sampling(i) for i in bider_]

        mean = predict[:, 1]
        sigma = self.sima
        price = (np.random.normal(size = mean.shape) * sigma + mean).astype('float32')
        return [{'bider': a, 'price': b} for a, b in zip(bider, price)]

    def training(self, action, var, reward):
        return self.backward(action, var, reward)

class MechanismTrainer:
    def __init__(self):
        self.stored_params = None
        self.history = []

        n = len(PublicKnowledge)
        self.players = [Player(i) for i in range(0, n)]
        self.batchsize = 100
        self.mechanism = MechanismLeaner(n)

    def store_params(self):
        self.store_params = [ i.dump() for i in self.players]

    def reset_params(self):
        for a, b in zip(self.players, self.store_params):
            a.loads(b)

    def run(self, batchsize = 100, record = True):
        valuation = [i.sample_valuation(batchsize) for i in self.players]
        action = np.array( [i.play(j) for i, j in zip(self.players, valuation)] ).transpose(1, 0)
        result = self.mechanism.decide(action)
        if record:
            prices = [[i['price']] for i in result]
            biders = [[i['bider']] for i in result]
            var = np.concatenate([biders, prices], axis = 1)
            self.history.append([actions, var])

        reward = [i.inform(result) for i in self.players] #train the policy of each player
        diff = np.abs(valuation.reshape(-1) - action.reshape(-1)).mean()
        return diff


    def get_data(self):
        self.store_params()
        self.history = []
        for i in range(10):
            diff = self.run(100)
        self.reset_params()
        actions = np.concatenate( [i[0] for i in self.history], axis = 0 )
        var = np.concatenate( [i[1] for i in self.history], axis = 0 )
        return actions, var, np.zeros(shape = (actions.shape[0],)) + diff

    def get_batches(self):
        inps = [[], [], []]
        for i in range(self.batchsize):
            data = self.get_data()
            for j in range(len(inps)):
                inps[j].append(data[j])
        return inps

    def one_epoch(self):
        inps = get_batches()
        self.mechanism.training(*inps)

        for i in range(10):
            diff = self.run(1000, record = False)

        return diff

def main():
    trainer = MechanismTrainer()
    while True:
        diff = trainer.one_epoch()
        print(diff)

if __name__ == '__main__':
    main()
