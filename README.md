# Mindful-Neural-Networks

## Mindfulness and thoughts

To watch your own thoughts without judging, is a popular exercice in mindfulness training. We want to see if we can transfer this exercice to artificial neural networs (ANN). Since thoughts are a rather abstract concept, we focus on something more tangible. Instead of thoughts we feed an ANN with its own weights, or to be more precise we train an ANN on its own weights. For that we initialise an ANN with random weights, train it for one epoch on said weights, take the adapted weights from the network, train the network on the new weights and so on. 

## Visuals
For better understandability we visualize the network and its weights. High weights are represented by thick lines, a redish colour represents positive values, and blueish colours represent negative values. Here we see two networks, one initialized with same valued weights, the other one with random uniformly chosen ones. 

<p align="center">
  <img src="https://github.com/bhaecker/Mindful-Neural-Networks/blob/master/graphics/nn_constant.png" width="400" />
  <img src="https://github.com/bhaecker/Mindful-Neural-Networks/blob/master/graphics/nn_random.png" width="400" />
</p> 

We can do even more and observe a neural network learning. Lets take the famous Iris data set as an example. We can observe how a neural network is trained on the four features, which are classified into three classes.



<img align="left" src="https://github.com/bhaecker/Mindful-Neural-Networks/blob/master/graphics/Iris_4663.gif" width="500" >

<br><br><br><br><br><br><br>

For example on the left, we see a network with two hidden layers with six neurons each and initial weights all set to ``1`` trained for ``150`` epochs on the Iris data set. 


<img align="right" src="https://github.com/bhaecker/Mindful-Neural-Networks/blob/master/graphics/Iris_483.gif" width="500" >

<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>

We can do the same thing with a different network architecture. On the right we see a network with only one hidden layer with eight neurons, again trained on the Iris data set for ``500`` epochs.


<img align="left"  src="https://github.com/bhaecker/Mindful-Neural-Networks/blob/master/graphics/Iris_random_483.gif" width="500" >

<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>

Weights are usually initialized randomly before training. We take the network from before, but rather setting all weights to ``1``, we set them randomly. Again we train for ``150`` epochs.

<br><br><br><br><br><br><br><br><br>

## Food for thought
The discerning reader will have noticed that we only consider the weights between neurons here. The bias (which is also part of the training) is set to zero at all time. The equation for a neuron 

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;z^{[l]}_i&space;=&space;w^T_i&space;\cdot&space;g^{[l-1]}(z_{(i-1)}^{[l-1]})&space;&plus;&space;b_i," target="_blank"><img src="https://latex.codecogs.com/gif.latex?\fn_cm&space;z^{[l]}_i&space;=&space;w^T_i&space;\cdot&space;g^{[l-1]}(z_{(i-1)}^{[l-1]})&space;&plus;&space;b_i," title="z^{[l]}_i = w^T_i \cdot g^{[l-1]}(z_{(i-1)}^{[l-1]}) + b_i," /></a>
</p> 

where <a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;w^T_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\fn_cm&space;w^T_i" title="w^T_i" /></a> is the vector of weights, <a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;g^{[l-1]}(z_{(i-1)}^{[l-1]})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\fn_cm&space;g^{[l-1]}(z_{(i-1)}^{[l-1]})" title="g^{[l-1]}(z_{(i-1)}^{[l-1]})" /></a> the activation of the old neuron and <a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;b_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\fn_cm&space;b_i" title="b_i" /></a> the vector of biases, boils down to 

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;z^{[l]}_i&space;=&space;w^T_i&space;\cdot&space;g^{[l-1]}(z_{(i-1)}^{[l-1]})." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\fn_cm&space;z^{[l]}_i&space;=&space;w^T_i&space;\cdot&space;g^{[l-1]}(z_{(i-1)}^{[l-1]})." title="z^{[l]}_i = w^T_i \cdot g^{[l-1]}(z_{(i-1)}^{[l-1]})." /></a>
</p> 


<img align="right" img src="https://github.com/bhaecker/Mindful-Neural-Networks/blob/master/graphics/nn_534.png" width="400" />
 
<br><br><br><br><br><br><br>

For example: a fully connected network of three layers with ``5,3,4`` neurons respectively, has a set of weights (written as entries of vectors) of size ``5*3+3*4 = 27``.

<br><br><br><br><br><br>


 <img align="left" img src="https://github.com/bhaecker/Mindful-Neural-Networks/blob/master/graphics/nn_534_in.png" width="400" />
 <br><br><br><br><br>
 
Since we can not feed a vector of size ``32`` into a layer of ``4`` neurons, we introduce an input layer of the size needed for each network, which is fully connected to the first layer. We freeze the weights between the input and the first layer, since we are not interested in adapting these weights. 


<br><br><br><br><br><br><br>
In order to train a neural network with the backpropagation algorithm, we need a measure on how good it performs at the given moment. This measure or loss function, compares the prediction of the network and the target. Since we want the network learn in a non-judgemental way on the one hand, but need labels for the algorithm on the other hand, we run into a conceptual problem. 

## From non-judgemental to auto encoders

We solve this issue by using an auto encoder structure. For that we include an output layer after our last layer, which has exactly the same number of neurons as the input layer and train with a target, which is equal to the input.
Auto encoders are used to learn an encoding of data. Here we learn an encoding of data, which is the encoding itself. 

<p align="center">
  <img src="https://github.com/bhaecker/Mindful-Neural-Networks/blob/master/graphics/nn_534_aut.png" width="400" />
</p> 



Like the input layer, we want to freeze the weights between the last layer and the output layer. It turns out that with a frozen set of weights between the last two layers, training is not working. This might be due to the way the backpropagation algorithm works. 

We unfreeze them, but do not include them in our input and in our analysis. 

## Training time

Since we can not train forever, we fix a maximal number of epochs for which we let the network learn. 

We can expect two behaviors of the network. Either it will run into a fix point, which means the weights do not change anymore. Or the weights continue to change and maybe oscillate or exhibit other patterns. 

Before we start the actual experiment, we need to define what it means to have a change in weights. We define it as follows:
If at least one of the weights changes more then a given threshold we consider it a change. If the network does not change for ten epochs, we stop the training. 




## Grid Search
We fix a threshold of ``0.01`` and a maximum number of epochs of ``~111``. 

We then perform a grid search to test out different network structures and their behavior. 

## Results
It turns out that most networks do not change their weights after a couple of epochs. A typical training process looks like this for this case:


There are several networks, which do not get stuck in a fix point but in- and decrease their weights during the period of training.
Here are some examples:


<p align="center">
  <img src="https://github.com/bhaecker/Mindful-Neural-Networks/blob/master/graphics/plasticity_655.gif" width="400" />
  <img src="https://github.com/bhaecker/Mindful-Neural-Networks/blob/master/graphics/plasticity_3434.gif" width="400" />
</p> 
and 


<p align="center">
  <img src="https://github.com/bhaecker/Mindful-Neural-Networks/blob/master/graphics/plasticity_484.gif" width="400" />
  <img src="https://github.com/bhaecker/Mindful-Neural-Networks/blob/master/graphics/plasticity_11119.gif" width="400" />
</p> 

