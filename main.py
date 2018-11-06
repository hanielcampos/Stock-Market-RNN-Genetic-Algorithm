'''
Author: Haniel Campos
Description: This genetic algorithm evolves neural networks in order to do
regression on stock prices. It assumes that data files come in as csv files in
the "Yahoo Finance" format. Change if needed.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import time
from optimizer import Optimizer
from tqdm import tqdm
from tensorflow.keras.models import save_model, model_from_json
from train import get_stock_data, compile_model, network_arch

start_time = time.time()

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='/Users/hanielcampos/PythonCode/ML/DL/stockGAProject/log.txt'
)

# Set random seed.
np.random.seed(16)

def train_networks(networks, data):
    """Train each network.
    Args:
        networks (list): Current population of networks
    """

    print "Training networks"
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(data)
        pbar.update(1)
    pbar.close()
    print "Training complete"


def get_average_loss(networks):
    """Get the average loss for a group of networks.
    Args:
        networks (list): List of networks
    Returns:
        float: The average loss of a population of networks.
    """

    total_loss = np.empty(len(networks),)
    for i in range(len(networks)):
        total_loss[i] = networks[i].loss

    return np.float(np.mean(total_loss))

def get_best_loss(networks):
    """ Get the loss value of the best performing network in a generation.
    Args:
        networks (list): List of networks
    Returns:
        float: The best loss out of a population of networks.
    """
    losses = np.empty(len(networks),)
    for i in range(len(networks)):
        losses[i] = networks[i].loss
    return np.amin(losses)


def generate(generations, population, nn_param_choices):
    """Generate a network with the genetic algorithm.
    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
    """

    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)

    # Prepare an array to record average and best losses.
    loss_t = np.empty((generations,))
    loss_bt = np.empty((generations,))

    data = get_stock_data()

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                    (i + 1, generations))

        # Train and get loss for networks.
        train_networks(networks, data)

        # Get and record the average loss for this generation.
        average_loss = get_average_loss(networks)
        loss_t[i] = average_loss

        # Get and record the best loss for this generation.
        best_loss = get_best_loss(networks)
        loss_bt[i] = best_loss

        # Print out the average and best loss of each generation.
        logging.info("Average Generation loss: %.3f" % (average_loss))
        logging.info('-'*80)
        logging.info("Best Generation loss: %.3f" % (best_loss))
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != (generations - 1):
            # Do the evolution.
            networks = optimizer.evolve(networks)
        else:
            pass

    # Record elapsed time
    end_time = time.time()
    time_elapsed = end_time - start_time
    minutes, seconds = divmod(time_elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    print(" Total running time was that of %d h : %d m : %d s" % (hours, minutes, seconds))

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.loss, reverse=False)

    # Print best network
    print "Best Performing Network:"
    print network_arch(networks[0].network)
    print "Network Loss:"
    print networks[0].loss

    # Save best network to hdf5 and JSON
    compile_model(networks[0].network).save("bestGeneticModel.hdf5")
    print("Saved best model to disk as HDF5")
    model_json = compile_model(networks[0].network).to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    print("Saved best model to disk as JSON")

    # Print out the top 5 networks.
    print "Top 5 Best Performing Networks:"
    print_networks(networks[:5])

    # Make and print plot with average loss history
    plt.figure()
    plt.plot(np.arange(1, generations + 1, 1), loss_t)
    plt.xlabel('Generation')
    plt.ylabel('Average loss')
    plt.grid(True)

    plt.figure()
    plt.plot(np.arange(1, generations + 1, 1), loss_bt)
    plt.xlabel('Generation')
    plt.ylabel('Best loss')
    plt.grid(True)

    plt.show()

def print_networks(networks):
    """Print a list of networks.
    Args:
        networks (list): The population of networks
    """

    logging.info('-'*80)
    for network in networks:
        network.print_network()


def main():
    """Evolve the populations."""

    generations = 5
    population = 20 # I suggest you set this as >= 10.

    # Define list with the layer types the NNs can choose from.
    layerType = []
    for i in ['Dense', 'GRU', 'LSTM']: # Layer type.
        for j in range(1, 11): # Number of neurons in layer.
            for k in ['selu', 'relu', 'tanh', 'linear']: # Layer activation function.
                layerType += [(i, j, k)]

    # Define the parameters the NNs can choose from.
    nn_param_choices = {
        'n_layers'  : range(0, 11), # testing
        'layer_info': layerType,
        'optimizer' : ['adam', 'adagrad', 'nadam', 'rmsprop'],
        'final_act' : ['selu', 'relu', 'tanh', 'sigmoid', 'linear']
    }

    logging.info("***Evolving %d generations with population %d***" %
                (generations, population))

    generate(generations, population, nn_param_choices)


if __name__ == '__main__':
    main()
