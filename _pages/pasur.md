---
title: "Solving Pasur Using GPU-Accelerated CFR — Under Construction"
permalink: /pasur/
author_profile: true
tags:
  - Counterfactual Regret Minimization
  - Reinforcement Learning
  - PyTorch
  - Nash Equilibrium
  - Game Theory
  - Memory Management
  - GPU Optimization
  - Efficient Computing
  - Artificial Intelligence
---


This repository is dedicated to the paper [*Solving Pasur Using GPU-Accelerated Counterfactual Regret Minimization*](https://arxiv.org/abs/2508.06559). You can find the code for this project [**here**](https://github.com/sinabaghal/pasur). 


I begin by explaining CFR. Suppose we have two players, Alex and Bob, who are playing a game and the corresponding game tree is shown here. The goal of the CFR algorithm is to find an optimal strategy for both players. 

By strategy, we mean a probability distribution over the possible actions at each node of the tree. For example, at the node denoted by B in the figure below, Bob has two actions where based on his current strategy, one action may be chosen with probability 20% (as shown) and the other with probability 80%. 

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/CFR_Tree.png" width="90%" height="90%">
</p>

Utilities at terminal nodes are defined naturally via the game’s scoring system. Utilities at other nodes is calculated via a backup pass. Notice that since we are in a zero-sum setting, the utilities of Alex and Bob always sum to zero at each node of the game tree. In other words, $$u_a$$ plus $$u_b$$ equals zero.

Now, we are looking to find an optimal strategy for both players. Here by “optimal strategy,” we mean a Nash Equilibrium. This is a pair of strategies where no player can improve their payoff by changing their own strategy while the other keeps theirs fixed.

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/nash.png" width="80%" height="80%">
</p>

Next, we define something called instantaneous regret for each action. Instantaneous regret is defined as the counterfactual utility of that action minus the counterfactual utility of the current node. The definitions are written here. Counterfactual utility of a node is the probability of reaching that node—assuming the current player has purposefully reached that node—multiplied by the expected utility if play continues from that point. Counterfactual utility of an action is defined in the same way. 

CFR is a classic algorithm that provably converges to this Nash Equilibrium. The idea of CFR is straightforward. We start with a uniform strategy, meaning each action is taken with equal probability. For example, if there are two actions, each one is chosen with probability 50%. At each iteration, we compute these instantaneous regrets for all nodes and store them. We then update the strategy by assigning probabilities to actions in proportion to the sum of all accumulated regrets up to the current iteration.  Finally, the output of CFR is the weighted average of all strategies observed so far, where the weights are given by the reach probabilities.

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/cfr_algo.png" width="70%" height="70%">
</p>

The aim of this work is to run CFR on the Pasur game tree, which has a height of 48 and an average size of about 2 to the power of 30 nodes.

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/pasur_game_tree.png" width="70%" height="70%">
</p>

Let me explain the game itself next! 

Pasur is played in 6 rounds, and in each round each player is dealt 4 cards, which they play sequentially over 4 turns, taking turns one after another. At each turn, a player places a card face up and either lays it in the pool or collects it along with some other pool cards, according to the rule shown in this table. Figure below shows an example of the first turn by Alex and Bob.

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/ex.png" width="70%" height="70%">
</p>


<p align="center">
<img src="https://sinabaghal.github.io/images/GT.png" width="80%" height="100%">
</p>
