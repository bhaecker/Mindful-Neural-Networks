# Mindful-Neural-Networks

## Mindfulness and thoughts

To watch your own thoughts without judging, is a popular exercice in mindfulness training. We want to see if we can transfer this exercice to artificial neural networs (ANN). Since thoughts are a rather abstract concept, we focus on something more tangible. Instead of thoughts we feed an ANN with its own weights, or to be more precise we train an ANN on its own weights. For that we initialise an ANN with random weights, train it for one epoch on said weights, take the adapted weights from the network, train the network on the new weights and so on. 

## Visuals
For better understandability we visualize the network and its weights. High weights are represented by thick lines, a redish colour represent positive values, and blueish colour represent negative values. Here we see two networks, one initialized with random weights, the other one with random uniformly chosen ones. 

GRAFIK

## Food for thought
We only consider the weights between neurons here, the bias (which is also part of the training) is set to zero at all time. 

For example: a fully connected network of three layers with ``4`` neurons each, has a set of weights of size ``4*4+4*4 = 32``.

GRAFIK

Since we can not feed a vector of size ``32`` into a layer of ``4`` neurons, we introduce an input layer of the size needed for each network, which is fully connected to the first layer. We freeze the weights between the input and the first layer, since we are not interested in adapting these weights. 

GRAFIK

In order to train a neural network with the backpropagation algorithm, we need a measure on how good it performs at the given moment. This measure or loss function, compares the prediction of the network and the target. Since we want the network learn in a non-judgemental way on the one hand, but need labels for the algorithm on the other hand, we run into a conceptual problem. 

## From non-judgemental to auto encoders

We solve this issue by using an auto encoder structure. For that we include an output layer after our last layer, which has exactly the same number of neurons as the input layer and train with a target, which is equal to the input.
Auto encoders are used to learn an encoding of data. Here we learn an encoding of data, which is the eocoding itself. 

GRAFIK

Like the input layer, we want to freeze the weights between the last layer and the output layer. It turns out that with a frozen set of weights between the last two layers, training is not working. This might be due to the way the backpropagation algorithm works. 

We unfreeze them, but do not include them in our input and in our analysis. 

## Training time

Since we can not train forever, we fix a maximal number of epochs for which we let the network learn. 

We can expect two behaviors of the network. Either it will run into a fix point, which means the weights do not change anymore. Or the weights continue to change and maybe oscillate or exhibit other patterns. 

Before we start the actual experiment, we need to define what it means to have a change in weights. We define it as follows:
If at least one of the weights changes more then a given threshold we consider it a change. If the network does not change for ten epochs, we stop the training. 




## Grid Search
We fix a threshold of ``0.01`` and a maximum number of epochs of ``~111``. 

<img src="https://github.com/bhaecker/Mindful-Neural-Networks/blob/master/graphics/plasticity.gif" width="500" height="790">

We then perform a grid search to test out different network structures and their behavior. 

## Results
It turns out that most networks do not change their weights after a couple of epochs. A typical training process looks like this for this case:


There are several networks, which do not get stuck in a fix point but in- and decrease their weights during the period of training.
Here are some examples.

