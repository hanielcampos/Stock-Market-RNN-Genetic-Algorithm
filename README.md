# Stock-Market-RNN-Genetic-Algorithm
A genetic algorithm that evolves generations of regression neural networks containing a combination of recurrent and dense layers. 

### Prerequisites
The genetic algorithm requires the following packages to work:

- Tensorflow (>=1.9)
- Numpy 
- Pandas 
- Tqdm
- ScikitLearn
- Random 
- Logging
- Matplotlib
- Time 
- Operator

## The Algorithm
The algorithm's approach to evolution is based on [this repository](https://github.com/harvitronix/neural-network-genetic-algorithm) by @harvitronix, with the modifications being that instead of evolving MLP classifiers this algorithm's able to evolve deep regression networks with a combination of dense and recurrent layers. I designed it to predict stock prices while reading the data from a CSV file in the standard Yahoo Finance format (Open, High, Low, Close, Adj Close, Volume), but I see no reason why it wouldn't be applicable in other scenarios where temporal regression is needed with only minor tweaks. I'll do my best to provide a good explanation of each aspect of the algorithm.

### The Network Class 
The class `Network` is used to define all attributes of a recurrent neural network that are of interest to the alogrithm. In it, the `network` method returns a dictionary item listing those attributes (which is then turned into a proper Tensorflow model by `compile_model` in `train.py`). It contains the following attributes: 

- `n_layers`  : The number of network hidden layers.
- `layer_info`: A list containing the following information for each one of the hidden layers:
  - Layer type (Dense, LSTM or GRU)
  - Number of layer neurons
  - Layer activation function
- `optimizer` : Optimizer to use when updating weight values.
- `final_act`  : The activation function of the output layer.

The algorithm uses this class as a source of information about the network, which is important when breeding and mutating populations.

### The Initial Population
The first population of networks is generated completely randomly via the `nn_param_choices` dictionary item. Each network is assigned a random number of hidden layers, which then are asigned random information (layer type, number of neurons and activation function), optimizer and output layer activation function. 

### Breeding and Evolving 
The individuals for all populations are ranked based on their performance with the testing dataset, which is given by the mean squared error (MSE) function (feel free to use other loss functions like MAE or RMSE). A certain top percentage (given by `retain` parameter in the `__init__` method of the `Optimizer` class) is kept and the others are discarded, with the exception of some random networks (done to prevent too much gravitation towards local maxima). The chance of this happening to a given network is given by the `random_select` parameter also in the `__init__` method. The retained population is then used to breed child networks and fill the remaining spots in the population.

When breeding two networks, the algorithm first randomly assigns the child the `n_layers` of one of its parents. It then loops through the parent's `layer_info` and randomly appends the child's `layer_info` at each iteration with the element from one of the parents. If the child's `n_layers` is greater than one of its parent's, it will run out of layer information before the loop ends, therefore, in that case, the algorithm assigns the remaining layers the same `layer_info` from the parent which the child got its `n_layers` from once. The child then has its `optimizer` and `final_act` randomly chosen from one of the parents.

Note: While the parents are always referred to as if there are only two, you could modify the algorithm so that each child has 3 or more parents.

In order to further avoid the algorithm getting stuck on a local maximum, each child can be randomly mutated in several ways. A child is either mutated or not based on the probability defined by the `mutate_chance` parameter in the `__init__` method of the `Optimizer` class. If it is chosen to be mutated, the algorithm will randomly choose one of the `network` attributes to mutate. If either `optimizer` or `final_act` are chosen, then the new one will randomly be assigned from `nn_param_choices`. Else, if `layer_info` is chosen the algorithm loops through the current `layer_info` and for each layer the new information will be drawn from `nn_param_choices` with probability defined by `mutate_chance`. Lastly, if `n_layers` is chosen, the algorithm assings a new, randomly chosen number of layers from `nn_param_choices` to the child. If the number is lesser than the original, the child's `layer_info` is cut at the new number of layers. However, if it is greater than the original, the empty layers are assigned random information from `nn_param_choices`.

The population is then trained and ranked, which leads to a repetition of this process.

### Final Steps
After evolving a specified amount of generations, the algorithm will print out the top 5 best performing networks and plot the average loss value and top network loss value across different generations. The model will also be saved to the HD.

# Acknowledgements
As previously mentioned, this algorithm is based on the one already developed by @harvotronix.

# Authors
- Haniel Campos 

# License 
This project is licensed under the MIT License - see the [LICENSE.md](docs/README.md) file for details
