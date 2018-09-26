"""A multilayer perceptron class

Bronson Duhart
Creado: 2017-11-22
Última modificación: 2018-01-31
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = 'data/'
FIGS_PATH = 'figs/'
RESULTS_PATH = 'results/'

class MLP:
    """A multilayer perceptron"""

    def __init__(self, inputsize, architecture):
        """(MLP, int, list of tuples (int, str)) -> None

        Builds a multilayer perceptron for input of certain size, with as many layers as elements in architecture, where each tuple indicates the number of neurons in that layer and their activation function.
        """

        self.architecture = architecture
        self.nlayers = len(architecture)
        self.nins = inputsize
        self.activations = {
            'step': np.vectorize(lambda n: int(n >= 0)),
            'linear': np.vectorize(lambda n: n),
            'logsig': np.vectorize(lambda n:
                np.exp(n) / (np.exp(n) + 1)
                if n < 0
                else 1 / (1 + np.exp(-n))
            ),
            'tanh': np.vectorize(lambda n: np.tanh(n)),
        }

        self.weights = [None] * self.nlayers
        self.bias = [None] * self.nlayers
        self.delta_w = [None] * self.nlayers
        self.delta_b = [None] * self.nlayers
        self.potentials = [None] * self.nlayers
        self.outputs = [None] * self.nlayers
        self.sensitivities = [None] * self.nlayers

        self.clear()


    def clear(self):
        """Reset all parameters to random values."""

        # First layer takes the inputs of the network, while the rest take the outputs
        # of previous layers
        nouts = self.nins
        for L in range(self.nlayers):
            nins = nouts
            nouts = self.architecture[L][0]
            self.weights[L] = np.random.random((nouts, nins))
            self.bias[L] = np.random.random((nouts, 1))
            self.delta_w[L] = np.zeros((nouts, nins))
            self.delta_b[L] = np.zeros((nouts, 1))


    def activation(self, layer):
        f = self.architecture[layer][1]
        return self.activations[f](self.potentials[layer])


    def derivative(self, layer):
        if self.architecture[layer][1] == 'step':
            return self.potentials[layer] == 0

        elif self.architecture[layer][1] == 'linear':
            return np.ones(self.potentials[layer].shape)

        elif self.architecture[layer][1] == 'logsig':
            return self.outputs[layer] * (1 - self.outputs[layer])

        elif self.architecture[layer][1] == 'tanh':
            return 1 - self.outputs[layer] ** 2


    def get_error(self, target, output, mode='r'):
        """Returns the classification error when mode == 'c' and the mean square
         error when mode == 'r'.
         """

        if target.ndim == 1:
            ninst = 1
            target = np.reshape(target, (ninst, -1))
        elif target.ndim == 2:
            ninst = target.shape[0]

        output = output.T

        if mode == 'c':
            error = np.sum(np.all(target == output, axis=1)) / ninst
        elif mode == 'r':
            error = np.sum((target - output) ** 2) / (2 * ninst)

        return error


    def feedforward(self, inputs):
        """Propagate an input forwards through the network"""

        if inputs.ndim == 1:
            ninst = 1
            inner_input = np.reshape(inputs, (self.nins, ninst))
        elif inputs.ndim == 2:
            inner_input = inputs.T

        for L in range(self.nlayers):
            self.potentials[L] = self.weights[L].dot(inner_input) + self.bias[L]
            self.outputs[L] = self.activation(L)
            inner_input = self.outputs[L]

        return self.outputs[self.nlayers - 1]


    def backprop(self, error, inputs, eta=0.1, alpha=0):
        """Propagate an error backwards through the network"""

        if inputs.ndim == 1:
            ninst = 1
            inputs = np.reshape(inputs, (self.nins, ninst))
        elif inputs.ndim == 2:
            ninst = inputs.shape[0]
            inputs = inputs.T

        # Output layer
        L = self.nlayers - 1
        self.sensitivities[L] = error * self.derivative(L)

        # Hidden and input layers
        for L in range(self.nlayers - 2, -2, -1):
            if L >= 0:
                self.sensitivities[L] = \
                    self.weights[L + 1].T.dot(self.sensitivities[L + 1]) * self.derivative(L)

            # Weight update (delayed one step w.r.t. sensitivities calculation)
            inner_input = self.outputs[L] if L >= 0 else inputs

            self.delta_w[L + 1] = \
                eta * self.sensitivities[L + 1].dot(inner_input.T) / ninst \
                + alpha * self.delta_w[L + 1]

            self.delta_b[L + 1] = \
                eta * self.sensitivities[L + 1].dot(np.ones((ninst, 1))) / ninst \
                + alpha * self.delta_b[L + 1]

            self.weights[L + 1] += self.delta_w[L + 1]
            self.bias[L + 1] += self.delta_b[L + 1]


    def train(self, train_set, eta=0.1, alpha=0, epochs=250, bsize=1,
              test_set=None, prog=False, mode='r'):
        """Train the network over the dataset with learning rate eta, momentum alpha
        and the number of epochs given.

        The train and test sets are given as dicts {X, Y}. If a test set is given,
        the method will return the training and testing errors along all epochs.
        """

        bsize = int(bsize)
        # Random selection of indices
        indices = np.random.permutation(train_set['X'].shape[0])

        if test_set is not None:
            train_error = np.zeros(epochs)
            test_error = np.zeros(epochs)
        else:
            train_error = test_error = None

        for epoch in range(epochs):
            if prog: stime = time()

            for i in range(0, train_set['X'].shape[0], bsize):
                bidx = indices[i:i + bsize]
                inputs = train_set['X'][bidx, :]
                target = train_set['Y'][bidx, :]

                output = self.feedforward(inputs)
                error = target.T - output
                self.backprop(error, inputs, eta, alpha)

            if prog: etime = time()

            if test_set is not None:
                train_error[epoch] = self.test(train_set, mode)[0]
                test_error[epoch] = self.test(test_set, mode)[0]

            if prog:
                print('Epoch %d: %.2f seconds' % (epoch, etime - stime))

        if test_set is not None:
            return train_error, test_error
        else:
            return self.test(train_set, mode)


    def test(self, test_set, mode='r'):
        """Test the network with the given test_set.

        Returns the classification error when mode == 'c' and the mean square
        error when mode == 'r'.
        """

        output = self.feedforward(test_set['X'])
        return self.get_error(test_set['Y'], output, mode), output.T


    def __str__(self):
        s = \
        '''
        Number of inputs: {}
        Number of layers: {}
        Number of neurons: {}
        Activation functions: {}

        Weights:
        {}

        Bias:
        {}
        '''.strip().replace(' ' * 4, '')

        return s.format(
            self.nins,
            self.nlayers,
            ' '.join([str(layer[0]) for layer in self.architecture]),
            ' '.join([layer[1] for layer in self.architecture]),
            '\n\n'.join([str(w) for w in self.weights]),
            '\n\n'.join([str(b) for b in self.bias])
        )


def test_mlp_class():
    architecture = [(1, 'logsig'), (3, 'logsig')]
    data = np.reshape(np.arange(12), (4, 3))
    target = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]])
    ann = MLP(3, architecture)
    t1 = ann.train({'X': data, 'Y': target}, epochs=3)
    t2 = ann.train({'X': data, 'Y': target}, epochs=3, bsize=2)


if __name__ == '__main__':
    test_mlp_class()
