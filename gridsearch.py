import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import keras
from itertools import product

from draw_neural_net import draw_neural_net
from mindfulness import extract_thought, self_auto_encoder
from makevid import makevideo

'''performe a grid search with two hyper parameters on different combinations of layers and neurons'''

max_epochs = 112
threshold = 0.01

#open text file to save grid search results
file = open('gridsearch_with_max_' + str(max_epochs) + '_epochs_and_'+str(threshold)+'_threshold_.txt', 'w')

#fix the combination of layers
for combi in list(product([2,3,4,5,6,7,8,9,10], repeat= 3)):

    layers_list = list(combi)

    model = self_auto_encoder(layers_list)
    in_out_dimension = model.input.shape[1]
    thoughts = np.empty((2,in_out_dimension))
    m = len(str(max_epochs))

    optimizer = keras.optimizers.Adam(lr=0.1)
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    #extract weights
    weights = extract_thought(model)
    #isolate only the trainable weights as "thought"
    thought = weights[in_out_dimension*layers_list[0]:(len(weights)-in_out_dimension*layers_list[-1])]

    thoughts[0] = thought

    #save a picture of the neural net with its "thought"
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')
    draw_neural_net(ax, .1, .9, .1, .9,layers_list, thought,0)
    fig.savefig('observe/nn_'+str(0)*m+'.png')
    del ax, fig
    #train for one epoch on current own weight/thought
    thought = np.reshape(thought, (1, len(thought)))

    epoch = 1
    counter = 0

    while epoch != max_epochs:
        history = model.fit(thought, thought, validation_split=0, epochs=1, batch_size=100, verbose=0)
        # extract weights
        weights = extract_thought(model)
        # isolate only the trainable weights as "thought"
        thought = weights[in_out_dimension * layers_list[0]:(len(weights) - in_out_dimension * layers_list[-1])]
        #insert thought as row to numpy array
        thoughts[1] = thought
        #save a picture of the neural net with its "thought"
        fig = plt.figure(figsize=(12, 12))
        ax = fig.gca()
        ax.axis('off')
        draw_neural_net(ax, .1, .9, .1, .9, layers_list, thought,epoch)
        fig.savefig('observe/nn_' + str(0)*(m-len(str(epoch)))+str(epoch)+'.png')
        del ax,fig
        #check for change in weights
        thoughts_diff = np.diff(thoughts,axis=0)
        thoughts_diff_bool =abs(thoughts_diff) > threshold

        #count for changes
        if np.any(thoughts_diff_bool) == False:
            counter += 1
        else:
            counter = 0
        # if there is no change after a couple of epochs, break
        if counter == 10:
            file.write(str(layers_list) + ' no change after ' + str(epoch - 10) + ' epochs\n')
            print('wrote line for '+str(layers_list))
            # delete all frames
            filelist = [f for f in os.listdir('observe') if f.endswith(".png")]
            for f in filelist:
                os.remove(os.path.join('observe', f))
            print('broke out')
            break
        #case when there is still change after max. epochs
        if epoch == max_epochs-1:
            file.write(str(layers_list) + ' still change after ' + str(max_epochs) + ' epochs\n')
            print('wrote line for '+str(layers_list))
            # make a video out of the frames
            makevideo(layers_list)
            # delete all frames
            filelist = [f for f in os.listdir('observe') if f.endswith(".png")]
            for f in filelist:
                os.remove(os.path.join('observe', f))
        thoughts[0] = thoughts[1]
        thought = np.reshape(thought, (1, len(thought)))
        epoch += 1

    del model

file.close()

