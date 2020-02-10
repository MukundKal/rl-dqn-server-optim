# Minimizing Costs in Energy Consumption of a Data Center *(using Deep Q-Learning)*

Tech companies have tremendous cost coming from their data center. A large portion of
that is the cooling required to keep their servers operating with optimal performance. So, the
problem to solve is ​ _minimizing_ ​ the energy consumption of cooling the servers to keep them in
their optimal range which will as a result minimize the electricity cost and save tremendous
amounts of money and therefore, millions of dollars can be saved.


## Inspiration :

In 2016, DeepMind AI minimized a big
part of Google’s cost by reducing Google *Data
Centre Cooling Bill by 40%* using their DQN
AI model (Deep Q-Learning). In our project we
will try and work on this very problem. We will
set up our own server environment, and we will
build an AI that will be controlling the
cooling/heating of the server so that it stays in
an optimal range of temperatures while saving
the maximum energy, therefore minimizing the
costs.
After accounting for "electrical losses
and other non-cooling inefficiencies,
**"this 40 percent reduction translated into a 15 percent reduction in overall power saving"**, according to Google. Considering that the company used
some 4,402,836 MWh of electricity in 2014
(equivalent to the amount of energy
consumed by 366,903 US households), this
15 percent will translate into savings of
hundreds of millions of dollars over the
years.




## 1. Introduction :

We will set up our own server environment, and we will build an AI that will be
controlling the cooling/heating of the server so that it stays in an optimal range of temperatures
while saving the maximum energy, therefore minimizing the costs. We are going to optimise this
using ​ **Deep-Q Reinforcement Learning** ​ for the optimization problem. We’ll be comparing the
performance of the server’s internal cooling system with our RL agent’s performance in order to
conclude energy savings. The internal cooling system simply brings the current temperature of
the server into the optimal range bounds whereas our agent in its neural network architecture will
learn weights using which it will predict a suitable action that results into less energy use.

## Environment :

Before we define the states, actions and rewards, we need to explain how the server
operates. At a given timestep, the server model has as input its intrinsic temperature, the number
of users online and the data rate going through it. Given, the 3-tuple, the agent has to predict an
action as defined in the action space. The states and action spaces are described on the following
page. This RL setup is illustrated below :
The ​ **variables** ​of this environment (at any minute) are given as follows :

- The number of users online
- The temperature of the server
- The data transmission rate through the server


- The energy spent by the server’s internal cooling system that automatically brings the server’s
temperature back to the optimal range whenever the server’s temperature goes outside this
optimal range.
- The energy spent by the agent onto the server in order to heat it or cool it.
The ​ **state** ​of the server at a time t is given as the 3-tuple shown below :
    Thus, the input vector will consist of three elements.The agent will take this vector as
input, and will return the action to play at each timestep.
    The temperature of the server can be approximated as given and for simplicity purposes,
we just suppose that these correlations are linear.
    This is the case because the more the number of users online, the more processing power
will be used by the server which will lead to heat dissipation and raise the temperature. Also, a
high data network throughput will lead to heat dissipation which similarly raises the temperature.
Different months will have varying atmospheric conditions and temperatures for example :


The energy spent by a system (our agent or the server’s internal cooling system) that
changes the server’s temperature within 1 unit of time can be approximated as proportional to
change in the temperature caused i.e. for simplicity purposes; taking constant as

1. Thus,
    The actions are simply the temperature changes that the agent can cause inside the server,
in order to heat it up or cool it down. In order to make our actions discrete, we will consider 5
possible temperature changes from −3 ◦ C to +3 ◦ C, so that we end up with the 5 following
possible actions that the agent can play to regulate the temperature of the server :
    The ​ **reward** ​ in the RL framework which is used to train the agent is the energy difference
with our agent turned on vs the internal cooling system.


## 2. Mathematical Formulation :

The environment of the server and its interaction is modelled as a Markov Decision
Process (MDP) which follows the Markov Property : The value ​ **V(s)** ​is not dependent on past
actions or states and just depends on the current state. The MDP is used to modelling process
where the decision depends upon agent and environment can be stochastic or non-stochastic.
**Bellman Equation** ​is used to model in the MDP which consists includes recursion of future
rewards.
The Value function for non-stochastic environment is given below :
As an example, consider this grid world where the values of each of the states is given,
and the agent upon learning this policy can navigate this MDP maze in order to get reward at
**WIN** ​state.
Thus, the value function recursive equation is used to move towards the reward.
0.81 ​ **→** 0.90 → 1.00 → **WIN**
0.73 ↑ ╳ 0.90 ↑ ╳
**0.64** ​ → 0.73 → 0.81 ↑ 0.
The value function for stochastic environment where agent decisions are probabilistic is
given as follows:
In the MDP model, Q-learning is an extension used to determine the quality of each
action for a given state. The Q-value (s,a) is dependent on both the current state and the action
taken. Therefore, the maximum Q-value of a given state is its value.


Temporal Difference (TD) :
This is the difference between the predicted Q-value and the received Q-value from the
environment itself. TD has to be minimised in order for our agent to learn the MDP.
TD is like an intrinsic reward. The agent will learn the Q-values in such a way that:
we use the temporal difference to reinforce the (action, state) from time t − 1 to time t, according
to the following equation:
This is a form of Gradient Descent. If the action a(t) has a higher Q-value, the agent is
more likely to choose that action and move to the next state and if an action a(t) has a lower
Q-value, then the agent has a lower probability to select that action.
Deep Q-learning :
Deep Q-Learning consists of combining Q-Learning to an Artificial Neural Network that
is performing regression. Inputs are encoded vectors, each one defining a state of the
environment;
These inputs are propagated through the network, and agent chooses an action to play.
Then the action played is the one associated with the output neuron that has the highest
Q-value(argmax).
Experience Replay ​[5]
We only considered transitions from one state s(t) to the next state s(t+1). The problem
with this is that s(t) is mostly very correlated with s(t+1). Thus, our agent is not learning much.


This could be improved if, instead of considering only this one previous transition, we
considered the last ​ **M** ​ transitions where M is a large number. This pack of the last M transitions
is what is called the Experience Replay. Then from this Experience Replay we take some random
batches of transitions to make our updates. So that our agent learns better and does not overfit.

## 3. Solution Method:

Deep-Q Learning algorithm used :
Phase 1 :
In the start, in order to utilize the experience replay, we initialize a memory to store
transitions for later use.The memory of the Experience Replay is initialized to an empty list
M.We choose a max size of memory. For example an array of 100 transitions. We start in the
first state, corresponding to a month within the year which was defined in the state space.
Repeat for each Epoch :
Repeat for each time instant i.e every minute until end of epoch :


● Predict the Q-value of current state.
● Perform the action that corresponds to the max of all predicted Q-value:
● Agent receives a reward
● Perform action and proceed to next state
● Append the transition in memory M
Phase 2 :
Training the agent on random batches to utilise experience replay.
We take a random batch B ∈ M of transitions. For all transitions of
the random batch B :
● Get the predictions:
● Get the targets:


● Compute the loss b/w the predictions and the targets over the whole batch B :
Loss =
=
We ​ **backpropagate** ​this loss error back into the neural network, through stochastic
gradient descent, we update weights.
Backpropagation :
Here in, we implement the Reinforcement Learning using neural networks.
Backpropagation involves optimizing the weights such that we minimise the loss function.
This is an example of a neural network architecture with 2 hidden layers.
Now in order to minimize the cost function, and optimise the weights we would do the
following :
a) Randomly Initialize the weights


b) Implement forward propagation for all training examples
c) Implement the cost function
d) Implement Backpropagation on the Weights to compute partial derivatives
e) Using Adam( A type of Gradient Descent optimiser) to minimise the cost function with
the weights in theta.
Now in order to perform Backpropagation on the neural network, we will take the
example of a single neuron to show the Backpropagation algorithm.
Here, y is the vector of outputs and x is the vector of inputs. w is the vector of weights.
Now in order to minimise the cost function, we will perform Backpropagation, general
MSE
In our case, the loss function:
Further, we randomly initialize the weights to start with and forward propagate the neural
architecture.
Now to minimise the cost function and obtain optimum weights we will perform
Backpropagation. For such an architecture the Backpropagation will be done as follows. Back
propagation is used to calculate the gradient of the loss function with respect to the parameters.
Let us denote the input to the jth neuron in the hidden layer by Vj and the output of the
jth neuron in the hidden layer by Zj. Then, we have


The output from the sth neuron of the output layer is
Therefore the relation between the inputs and outputs is given by


Now again using the chain rule we get
Now using these equations we will Backpropagate on the neural network architecture and
obtain optimum weights which will minimize the cost function.


## Simulation:

Building the Environment:
Neural Network Architecture :


Implementing the DQN :
Simulation Setup :


Simulation - ​Training and Testing Code.

## 3. Results and Analysis :

After training the model for 100 epochs each being 5 months, the model was tested for 1
year and the results were a savings on around 40-50% depending on different trials.


## Important References :

_1. DeepMind, 2016;_ ​ **_DeepMind AI Reduces Google Data Centre Cooling Bill by 40%_**
2. _Richard Sutton et al., 1998,_ ​ **_Reinforcement Learning I: Introduction_**
3. _Arthur Juliani, 2016,_ ​ **_Simple Reinforcement Learning with Tensorflow (Part 4)_**
4. _D. J. White, 1993,_ ​ **_A Survey of Applications of Markov Decision Processes_**
5. _Richard Sutton, 1988,_ ​ **_Learning to Predict by the Methods of Temporal Differences_**
6. _Michel Tokic, 2010,_ ​ **_Adaptive ε-greedy Exploration in Reinforcement Learning Based_**
    **_on Value Differences_**
7. _Tom Schaul et al., Google DeepMind, 2016,_ ​ **_Prioritized Experience Replay_**


