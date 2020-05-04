# Mindful-Neural-Networks

To watch your own thoughts without judging, is a popular exercice in mindfulness training. We want to see if we can transfer this exercice to artificial neural networs (ANN). Since thoughts are a rather abstract concept, we focus on something more tangible. Instead of thoughts we feed an ANN with its own weights, or to be more precise we train an ANN on its own weights. For that we initialise an ANN with random weights, tran it for one epoch on said weights, take the adapted weights from the network, train the network on the new weights and so on. 

We only consider the weights between neurons here, the bias (which is also part of the training) is set to zero at all time. 

For example: a fully connected network of three layers with ``4`` neurons each, has a set of weights of size ``4*4+4*4 = 32``.

GRAFIK

Since we can not feed a vector of size ``32`` into a layer of ``4`` neurons, we introduce an input layer of the size needed for each network, which is fully connected to the first layer. We freeze the weights between the input and the first layer, since we are not interested in adapting these weights. 

GRAFIK

In order to train a neural netwrok with the backpropagation algorithm, we need a measure on how good it performs at the given moment. This measure or loss function, compares the prediction of the network and the target. Since we want the network learn in a judgemental way on the one hand, but need labels for the algorithm on the other hand, we run into a conceptual problem. 

##From non-judgemental to auto encoders

We solve this issue by using an auto encoder structure. For that we include an output layer after our last layer, which has exactly the same number of neurons as the input layer and train with a target, which is equal to the input.
Auto encoders are used to learn an encoding of data. Here we learn an encoding of data, which is the eocoding itself. 

GRAFIK

##Training time

Since we can not train forever, we fix a maximal number of epochs for which we let the network learn. 

We can expect two behaviors of the network. Either it will run into a fix point, which means the weights do not change anymore. Or the weights continue to change and maybe oscillate or exhibit other patterns. 

Before we start the actuall experiment, we need to define what it means to have a change in weights. We define it as follows:
If at leat one of the weights changes more then a given threshold we consider it a change. If the network does not change for ten epochs, we stop the training. 

For a better undestanding, we 


<img src="https://github.com/bhaecker/Mindful-Neural-Networks/blob/master/graphics/plasticity.gif" width="500" height="790">
blubluibul

