import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, InputLayer, concatenate

from draw_neural_net import draw_neural_net

def extract_thought(model):
    '''
    extracts all weights from a keras model and returns them as a list
    '''
    thought = []
    for i,layer in enumerate(model.layers):
        weights = layer.get_weights()
        nearly_flat_list = [item for sublist in weights for item in sublist]
        flat_list = [item for sublist in nearly_flat_list for item in sublist]
        thought = thought + flat_list
    return(thought)

def self_auto_encoder(layers_list):
    '''
    sets up an auto-encoder keras model, which has the desired network (from layers_list) between in- and output layer
    '''
    #determine in- and output dimension
    dimension = 0
    layer_old = layers_list[0]
    for layer in layers_list[1:]:
        dimension += layer_old*layer
        layer_old = layer
    input = Input(shape=(dimension,))
    x = input
    for i,neurons in enumerate(layers_list):
        if i == 0:
            #input layer
            x = Dense(neurons, activation="relu",kernel_initializer=keras.initializers.Constant(value=1/(dimension*layers_list[0])),use_bias=False,trainable=False)(x)#kernel_initializer=keras.initializers.Ones()
        else:
            #hidden layers
            x = Dense(neurons, activation="relu",kernel_initializer=keras.initializers.RandomUniform(minval=-1.05, maxval=1.05, seed=12),use_bias=False)(x)#
    #output layer
    x = Dense(dimension, activation="softmax",kernel_initializer=keras.initializers.Constant(value=1/(dimension*layers_list[-1])),use_bias=False,trainable=True)(x)
    self_auto_encoder = Model(inputs=input, outputs=x)

    print(self_auto_encoder.summary())
    return(self_auto_encoder)

####

def NN_model(layers_list):
    '''
    return a keras neural network model with the desired layers and neurons from input list
    '''
    input = Input(shape=(layers_list[0],))
    x = input
    #x = Dense(layers_list[0], activation="relu", use_bias=True)(input)
    for neurons in layers_list[1:-1]:
        x = Dense(neurons, activation="relu", use_bias=False)(x)
    x = Dense(layers_list[-1], activation="softmax", use_bias=False)(x)
    model = Model(inputs=input, outputs=x)
    print(model.summary())
    return(model)

def NN_model_equal_weights(layers_list):
    '''
    return a keras neural network model with the desired layers and neurons from input list and same valued weights
    '''
    input = Input(shape=(layers_list[0],))
    x = input
    for neurons in layers_list[1:-1]:
        x = Dense(neurons, activation="relu", kernel_initializer = keras.initializers.Ones(),use_bias=False)(x)
    x = Dense(layers_list[-1], activation="softmax", kernel_initializer=keras.initializers.Ones(), use_bias=False)(x)
    model = Model(inputs=input, outputs=x)
    print(model.summary())
    return(model)

def observe_training(layers_list,train,target,epochs):
    '''
    Sets up a keras model with functions NN_model or NN_model_equal_weights and trains it.
    Creates an image of weights before and after training
    '''
    model = NN_model_equal_weights(layers_list)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    thought = extract_thought(model)
    print(thought)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')
    draw_neural_net(ax, .1, .9, .1, .9,layers_list, thought)
    fig.savefig('nn_initial.png')

    history = model.fit(train, target, validation_split=0.1, epochs=epochs, batch_size=100, verbose=1)

    thought = extract_thought(model)
    print(thought)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')
    draw_neural_net(ax, .1, .9, .1, .9, layers_list, thought)
    fig.savefig('nn_aftertraining.png')

    return(history)

def observe_all_training(layers_list,train,target,epochs):
    '''
    Sets up a keras model with functions NN_model or NN_model_equal_weights and trains it.
    Creates an image of weights for each epoch.
    '''
    m = len(str(epochs))

    model = NN_model(layers_list)
    optimizer = keras.optimizers.Adam(lr=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    #sparse_categorical_crossentropy
    thought = extract_thought(model)
    print(thought)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')
    draw_neural_net(ax, .1, .9, .1, .9,layers_list, thought, 0)
    fig.savefig('observe/nn_'+str(0)*m+'.png')

    for epoch in range(1,epochs):
        history = model.fit(train, target, validation_split=0.2, epochs=1, batch_size=100, verbose=1)
        thought = extract_thought(model)
        print(thought)
        fig = plt.figure(figsize=(12, 12))
        ax = fig.gca()
        ax.axis('off')
        draw_neural_net(ax, .1, .9, .1, .9, layers_list, thought, epoch)
        fig.savefig('observe/nn_' + str(0)*(m-len(str(epoch)))+str(epoch)+'.png')

    return(history)

####

def mindfulness_model(input_dim,layers_list,output_dim):
    """
    define a model for a classification task, which gets as additional inputs its own weights
    """
    # define two sets of inputs
    inputA = Input(shape=(input_dim,))

    # the first branch is our model
    x = Dense(layers_list[0], activation="relu", use_bias=False)(inputA)
    for neurons in layers_list[1:]:
        x = Dense(neurons, activation="relu", use_bias=False)(x)
    #output_layer = Dense(output_dim, activation="relu", use_bias=False)(x)
    x = Model(inputs=inputA, outputs=x)
    thought = extract_thought(x)
    m = len(thought)

    # the second branch is the weights of the first branch
    inputB = Input(shape=(m,))
    #y = Model(inputs=inputB, outputs=output_layer)

    # combine the output of the two branches
    combined = concatenate([x.output, inputB])
    z = Dense(output_dim, activation="relu")(combined)
    model = Model(inputs=[x.input, inputB], outputs=z)

    return(model)

def train_mindulness_model(model,samples,target,split,epochs):
    """
    train a mindfulness models
    """
    thought = []
    for i, layer in enumerate(model.layers):
        #fetch all weights but the ones from the last layer
        if i != len(model.layers) - 1:
            weights = layer.get_weights()
            nearly_flat_list = [item for sublist in weights for item in sublist]
            flat_list = [item for sublist in nearly_flat_list for item in sublist]
            thought = thought + flat_list

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #reshape thought input
    thought = np.reshape(thought,(1,len(thought)))
    thought = np.repeat(thought,samples.shape[0],axis=0)

    history = model.fit([samples,thought], target, validation_split=split, epochs=epochs, batch_size=20, verbose=1)

    # save graph with new weights/thought
    thought = []
    for i, layer in enumerate(model.layers):
        # fetch all weights but the ones from the last layer
        if i != len(model.layers) - 1:
            weights = layer.get_weights()
            nearly_flat_list = [item for sublist in weights for item in sublist]
            flat_list = [item for sublist in nearly_flat_list for item in sublist]
            thought = thought + flat_list
    print(thought)

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return()
