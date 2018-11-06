import numpy as np
import pandas as pd
from tqdm import tqdm
from normalize import Normalizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, TimeDistributed, Flatten,  Reshape, BatchNormalization, GRU, Activation
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Define an early stopper for training
early_stopper = EarlyStopping(patience=5)


def get_stock_data():
    print "Loading data"
    # Import the data from a CSV file coming from Yahoo Finance.
    data_csv = pd.read_csv('all_stocks_5yr.csv') # Change file path as needed

    # Define a function to remove any NaNs and Infs from an array
    def remove_nan_inf(a):
        s = a[~np.isnan(a).any(axis=1)]
        t = s[~np.isinf(s).any(axis=1)]
        return t

    # Define a Numpy array to hold the information for the NNs.
    n_features = 5
    data = np.empty((data_csv.shape[0], n_features))
    L = 1259  # Number of observations per subject (more info in README)

    print "Loading data"

    # Load the data and normalize it per subject (modify if necessary)
    for i in tqdm(range(int(float(data_csv.shape[0]) / L))):
        holder_array = np.empty((data.shape[1], L))  # Array to temporarily hold information
        for info, j in zip(['open', 'close', 'high', 'low', 'volume'], range(n_features)):
            for k in range(L):
                holder_array[j][k] = np.array(data_csv[info][i * L + k],
                                              dtype=np.float)

        holder_array = remove_nan_inf(holder_array.T)
        scaled_info = Normalizer().fit_transform(holder_array)
        for j in range(scaled_info.shape[0]):
            data[i * L + j] = scaled_info[j]

    # Set batch size, input shape and train/test prop.
    batch_size = 1
    train_test_prop = 0.8
    time_step = 1

    # Organize the data so that inputs are 3D and work with LSTMs.
    data_train = data[:int(data.shape[0] * train_test_prop)]
    data_test = data[int(data.shape[0] * train_test_prop):]
    x_train = data_train[:-1].reshape(data_train[:-1].shape[0], time_step, data_train[:-1].shape[1])
    y_train = data_train[1:][:, 0]
    x_test = data_test[:-1].reshape(data_test[:-1].shape[0], time_step, data_test[:-1].shape[1])
    y_test = data_test[1:][:, 0]

    print "Data successfully loaded"

    return (batch_size, x_train, x_test, y_train, y_test)


def network_arch(network):
    """     Returns an appropriate version of the network
        Args:
            network (dict): the parameters of the network
        Returns:
            A dictionary with the proper network information and architecture
    """
    networkLayers = []
    for i in range(len(network['layer_info'])):
        if network['layer_info'][i][0] == 'Dense':
            networkLayers += [network['layer_info'][i]]
        else:
            networkLayers += [(network['layer_info'][i][0], network['layer_info'][i][1], 'tanh')]
    networkInfo = {'Number of Layers': network['n_layers'], 'Layer Information': networkLayers,
                   'Optimizer': network['optimizer'], 'Output Activation': network['final_act']}
    return networkInfo


def compile_model(network):
    """     Compile a sequential model.
        Args:
            network (dict): the parameters of the network
        Returns:
            a compiled network.
        Note the input shape here is considered to be (1, 5) and loss function is set to MSE.
        Modify if necessary
    """
    # Get the network parameters.
    n_layers = network['n_layers']  # Note n_layers is the number of hidden layers.
    layer_info = network['layer_info']
    optimizer = network['optimizer']
    final_act = network['final_act']

    # Set the number of input and output features and time step.
    input_features = 5
    output_features = 1
    time_step = 1

    # Add input layer
    inputs = Input(shape=(time_step, input_features))

    # Add each layer

    if n_layers == 0:
        # If n_layers == 0, flatten and jump straight to the output layer.
        hidden_layer = Reshape((input_features,))(inputs)

    elif n_layers > 0:
        # If n_layers > 0, loop through layer_info.
        for i in range(n_layers):
            if i == 0:
                # For the first hidden layer, specify the layer input as 'inputs'
                if layer_info[i][0] == 'Dense':
                    hidden_layer = TimeDistributed(
                        Dense(layer_info[i][1], kernel_initializer='he_normal',
                              kernel_regularizer=l2(0.01), use_bias=False)
                    )(inputs)
                    hidden_layer = Activation(layer_info[i][2])(hidden_layer)
                    hidden_layer = BatchNormalization()(hidden_layer)
                    hidden_layer = Dropout(0.5)(hidden_layer)

                elif layer_info[i][0] == 'LSTM':
                    hidden_layer = LSTM(layer_info[i][1], return_sequences=True, kernel_initializer='he_normal', kernel_regularizer=l2(
                        0.01), use_bias=False)(inputs)
                    hidden_layer = Activation('tanh')(hidden_layer)
                    hidden_layer = BatchNormalization()(hidden_layer)
                    hidden_layer = Dropout(0.5)(hidden_layer)

                elif layer_info[i][0] == 'GRU':
                    hidden_layer = GRU(layer_info[i][1], return_sequences=True, kernel_initializer='he_normal', kernel_regularizer=l2(
                        0.01), use_bias=False)(inputs)
                    hidden_layer = Activation('tanh')(hidden_layer)
                    hidden_layer = BatchNormalization()(hidden_layer)
                    hidden_layer = Dropout(0.5)(hidden_layer)

            elif i > 0:
                # For the next hidden layers, simply add them along with the batch normalization and dropout.
                if layer_info[i][0] == 'Dense':
                    hidden_layer = TimeDistributed(
                        Dense(layer_info[i][1], use_bias=False,
                              kernel_initializer='he_normal', kernel_regularizer=l2(0.01))
                    )(hidden_layer)
                    hidden_layer = Activation(layer_info[i][2])(hidden_layer)
                    hidden_layer = BatchNormalization()(hidden_layer)
                    hidden_layer = Dropout(0.5)(hidden_layer)

                elif layer_info[i][0] == 'LSTM':
                    hidden_layer = LSTM(layer_info[i][1], return_sequences=True, use_bias=False,
                                        kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(hidden_layer)
                    hidden_layer = Activation('tanh')(hidden_layer)
                    hidden_layer = BatchNormalization()(hidden_layer)
                    hidden_layer = Dropout(0.5)(hidden_layer)

                elif layer_info[i][0] == 'GRU':
                    hidden_layer = GRU(layer_info[i][1], return_sequences=True, use_bias=False,
                                       kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(hidden_layer)
                    hidden_layer = Activation('tanh')(hidden_layer)
                    hidden_layer = BatchNormalization()(hidden_layer)
                    hidden_layer = Dropout(0.5)(hidden_layer)

        # Add the flattening layer
        hidden_layer = Flatten()(hidden_layer)

    hidden_layer = Dense(output_features, use_bias=True,
                         kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(hidden_layer)
    outputs = Activation(final_act)(hidden_layer)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    print(network_arch(network))

    return model


def train_and_score(network, data):
    """Train the model, return test loss.
    Args:
        network (dict): the parameters of the network
    """
    batch_size, x_train, x_test, y_train, y_test = data

    model = compile_model(network)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10000,
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

    return score
