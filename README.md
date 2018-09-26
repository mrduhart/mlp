# Multilayer perceptron

This is a Python class that lets you create multilayer perceptrons with varying number of layers and units (neurons). I created it for my neural networks and deep learning courses as a demonstration tool.

*Note*: At the moment, it only works well with regression problems. I plan to improve support for classification problems in the future.

### Created With

* Python 3.5
* Numpy 1.14.1
* Matplotlib 2.1.2

### Usage

```
from mlp import MLP
import numpy as np

# Create a 3-1-3 logistic network
architecture = [(1, 'logsig'), (3, 'logsig')]
ann = MLP(3, architecture)

# Create some data (4x3 input with 4x3 output)
data = np.reshape(np.arange(12), (4, 3))
target = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]])

# Train with default parameters
t1 = ann.train({'X': data, 'Y': target})
print('Training error: {:.4f}\nNetwork outputs:\n{}'.format(*t1))

# Change training parameters
t2 = ann.train({'X': data, 'Y': target}, epochs=3, bsize=2)
print('Training error: {:.4f}\nNetwork outputs:\n{}'.format(*t2))
```

## Author

* **Bronson Duhart** - [mrduhart](https://github.com/mrduhart/)
