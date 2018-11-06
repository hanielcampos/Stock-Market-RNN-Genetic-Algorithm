"""
Class that holds a genetic algorithm for evolving a network.
Credit:
    Most of this code was originally inspired by:
    https://github.com/harvitronix/neural-network-genetic-algorithm/blob/master/optimizer.py
"""
import numpy as np
from operator import add
import random
from network import Network


class Optimizer():
    """Class that implements genetic algorithm for NN optimization."""

    def __init__(self, nn_param_choices, retain=0.3, random_select=0.1, mutate_chance=0.4):
        """Create an optimizer.
        Args:
            nn_param_choices (dict): Possible network paremters
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected network
                remaining in the population
            mutate_chance (float): Probability a network will be
                randomly mutated
        """

        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.nn_param_choices = nn_param_choices

    def create_population(self, count):
        """Create a population of random networks.
        Args:
            count (int): Number of networks to generate, aka the
                size of the population
        Returns:
            (list): Population of network objects
        """

        pop = []
        for _ in range(count):
            # Create a random network.
            network = Network(self.nn_param_choices)
            network.create_random()

            # Add the network to our population.
            pop.append(network)

        return pop

    @staticmethod
    def fitness(network):
        """Return the loss, which is our fitness function."""
        return network.loss

    def grade(self, pop):
        """Find average fitness for a population.
        Args:
            pop (list): The population of networks
        Returns:
            (float): The average loss of the population
        """
        scores = np.empty(len(pop))
        for i in range(len(pop)):
            scores[i] = self.fitness(pop[i])
        return np.mean(scores)

    def breed(self, mother, father):
        """Make two children as parts of their parents.
        Args:
            mother (dict): Network parameters
            father (dict): Network parameters
        Returns:
            (list): Two network objects
        """

        children = []
        for _ in range(2):
            child = {}

            # Pick number of hidden layers and optimizers for the kid.
            child['n_layers'] =random.choice(
                                             [mother.network['n_layers'], father.network['n_layers']]
                                             )
            child['optimizer'] =random.choice(
                                          [mother.network['optimizer'], father.network['optimizer']]
                                          )
            child['final_act'] = random.choice(
                                          [mother.network['final_act'], father.network['final_act']]
                                          )
            # Loop through the hidden layers and pick layer_info for for the kid.
            child_layer_info = []
            for i in range(child['n_layers']):
                try:
                    single_layer_info = []
                    # Try selecting a layer feature from the mother and father.
                    for k in range(3):
                        single_layer_info += [random.choice(
                            [mother.network['layer_info'][i][k], father.network['layer_info'][i][k]]
                        )]
                    child_layer_info += [tuple(single_layer_info)]
                except:
                    # In case of error, select the layer_info from whomever the kid inherited n_layers from.
                    if len(mother.network['layer_info']) == child['n_layers']:
                        child_layer_info += [mother.network['layer_info'][i]]
                    elif len(father.network['layer_info']) == child['n_layers']:
                        child_layer_info += [father.network['layer_info'][i]]
            child['layer_info'] = child_layer_info

            # Now create a network object.
            network = Network(self.nn_param_choices)
            network.create_set(child)

            # Randomly mutate some of the children.
            if self.mutate_chance > random.random():
                network = self.mutate(network)

            children.append(network)

        return children

    def mutate(self, network):
        """Randomly mutate one part of the network.
        Args:
            network (dict): The network parameters to mutate
        Returns:
            (Network): A randomly mutated network object
        """

        # Choose a random key.
        mutation = random.choice(list(self.nn_param_choices.keys()))

        # Mutate one of the params.
        if mutation == 'optimizer':
            # Self explanatory.
            network.network['optimizer'] = random.choice(self.nn_param_choices['optimizer'])
        elif mutation == 'final_act':
            # Self explanatory.
            network.network['final_act'] = random.choice(self.nn_param_choices['final_act'])
        elif mutation == 'layer_info':
            # Mutate the hidden layers.
            if network.network['n_layers'] == 0:
                # If there are no hidden layers, mutating them is equivalent to doing nothing.
                pass
            else:
                for i in range(network.network['n_layers']):
                    if self.mutate_chance > random.random():
                        network.network['layer_info'][i] = random.choice(self.nn_param_choices['layer_info'])
                    else:
                        pass
        elif mutation == 'n_layers':
            # Select new number of hidden layers.
            new_n = random.choice(self.nn_param_choices['n_layers'])
            if new_n == network.network['n_layers']:
                # If it's the same, do nothing.
                pass
            elif new_n < network.network['n_layers']:
                # If new_n is smaller than the original, select all layer_info from the original up to new_n.
                new_layer_info_smaller = []
                for i in range(new_n):
                    new_layer_info_smaller += [network.network['layer_info'][i]]
                network.network['layer_info'] = new_layer_info_smaller
                network.network['n_layers'] = new_n
            elif new_n > network.network['layer_info']:
                # If new_n is bigger than the original, select all layer_info from the original up to new_n and then complete with random layer_info.
                new_layer_info_bigger = []
                for i in range(new_n):
                    try:
                        new_layer_info_bigger += [network.network['layer_info'][i]]
                    except:
                        new_layer_info_bigger += [random.choice(self.nn_param_choices['layer_info'])]
                network.network['layer_info'] = new_layer_info_bigger
                network.network['n_layers'] = new_n

        return network

    def evolve(self, pop):
        """Evolve a population of networks.
        Args:
            pop (list): A list of network parameters
        Returns:
            (list): The evolved population of networks
        """

        # Get scores for each network.
        graded = [(self.fitness(network), network) for network in pop]

        # Sort on the scores.
        graded = [x[1]
                  for x in sorted(graded, key=lambda x: x[0], reverse=False)]

        for i in range(len(graded)):
            print "%d   :   %g" % (i + 1, graded[i].loss)

        # Check for nan or exploding gradients in losses and delete the network if true
        del_list = []
        for i in range(len(graded)):
            if (np.isnan(graded[i].loss)) or (np.isinf(graded[i].loss)):
                del_list += [i]
            else:
                pass

        graded = np.delete(graded, del_list).tolist()

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded) * self.retain)

        # The parents are every network we want to keep.
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Now find out how many spots we have left to fill.
        desired_length = len(pop) - len(parents)

        # In case the appropriate population isn't big enough, fill the remaining spots with random networks.
        if len(parents) >= 2:
            pass
        else:
            networks = self.create_population(desired_length)
            parents += networks


        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:
            # Get a random mom and dad.
            male = random.randint(0, len(parents) - 1)
            female = random.randint(0, len(parents) - 1)

            # Assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents += children

        return parents
