"""Class that represents the network to be evolved."""
import random
import logging
from train import train_and_score, network_arch


class Network():
    """ Represent a network and let us operate on it. """

    def __init__(self, nn_param_choices=None):
        """Initialize our network.
        Args:
            nn_param_choices (dict): Parameters for the network, include:
                n_layers (list)             : [1, 2, 3, 4, ...]
                layer_info (list of tuples) : [(# of neurons , layer type , activation func)]
                optimizer (list)            : ['sgd', 'adam', 'rmsprop', ...]
                final_act (list)            : ['linear', 'tanh', 'selu', ...]
        """
        self.loss = 10.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents network parameters

    def create_random(self):
        """Create a random network."""
        number_of_layers = random.choice(self.nn_param_choices['n_layers'])
        self.network['n_layers'] = number_of_layers
        layer_info = []
        if number_of_layers == 0:
            pass
        else:
            for i in range(number_of_layers):
                layer_info += [random.choice(self.nn_param_choices['layer_info'])]
        self.network['layer_info'] = layer_info
        self.network['optimizer'] = random.choice(self.nn_param_choices['optimizer'])
        self.network['final_act'] = random.choice(self.nn_param_choices['final_act'])

    def create_set(self, network):
        """Set network properties.
        Args:
            network (dict): The network parameters
        """
        self.network = network

    def train(self, data):
        """Train the network and record the loss."""
        if self.loss == 10.:
            self.loss = train_and_score(self.network, data)

    def print_network(self):
        """Print out a network."""
        logging.info(network_arch(self.network))
        logging.info("Network loss: %.3f" % (self.loss))
