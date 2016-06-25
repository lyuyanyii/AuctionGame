import theano
from theano import tensor as T

class MechanismLeaner:
    def __init__(self):
        self.params = []
        self.nonlinearity = lambda x: T.tanh(x)
        self.num_layers = 3
        self.sigma = 0.01
        self.build_network()

    def build_network(self):
        #inp of the network
        #price of each person
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
                initW = np.random.normal(size = (inp,out))
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
        for i in range(self.num_layers):
            x = add_fc(x, 1, 1, self.nonlinearity)
        gaussian = add_fc(x, 1, 2)

        self.forward = theano.function(inputs = [inp], outputs = gaussian)

        prob = calc_prob(gaussian, var)
        loss = -T.mean( T.log(prob) * (reward- reward.mean()) )

        updates = []
        for i in self.params:
            grad_i = T.grad(loss, i)
            updates.append((i, i-grad_i*np.float32(0.01)))

        self.backward = theano.function(inputs = [inp, var, reward], outputs = loss, updates = updates)
        return
