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
<img src="https://sinabaghal.github.io/files/pasur/CFR_Tree.png" width="110%" height="110%">
</p>

Utilities at terminal nodes are defined naturally via the game’s scoring system. Utilities at other nodes is calculated via a backup pass. Notice that since we are in a zero-sum setting, the utilities of Alex and Bob always sum to zero at each node of the game tree. In other words, $$u_a$$ plus $$u_b$$ equals zero.

Now, we are looking to find an optimal strategy for both players. Here by “optimal strategy,” we mean a Nash Equilibrium. This is a pair of strategies where no player can improve their payoff by changing their own strategy while the other keeps theirs fixed.

<p align="left">
<img src="https://sinabaghal.github.io/files/pasur/nash.png" width="90%" height="90%">
</p>

Next, we define something called instantaneous regret for each action. Instantaneous regret is defined as the counterfactual utility of that action minus the counterfactual utility of the current node. The definitions are written here. Counterfactual utility of a node is the probability of reaching that node—assuming the current player has purposefully reached that node—multiplied by the expected utility if play continues from that point. Counterfactual utility of an action is defined in the same way. 

CFR is a classic algorithm that provably converges to this Nash Equilibrium. The idea of CFR is straightforward. We start with a uniform strategy, meaning each action is taken with equal probability. For example, if there are two actions, each one is chosen with probability 50%. At each iteration, we compute these instantaneous regrets for all nodes and store them. We then update the strategy by assigning probabilities to actions in proportion to the sum of all accumulated regrets up to the current iteration.  Finally, the output of CFR is the weighted average of all strategies observed so far, where the weights are given by the reach probabilities.

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/cfr_algorithm.png" width="50%" height="50%">
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

Let me explain the rule and scoring system in more detail. If the played card is numeric, it can collect any subset of cards whose values add up to 11. If it is a Jack, it can collect all cards in the pool except for Kings and Queens. If it is a King, it can only collect a single King, and if it is a Queen, it can only collect a single Queen. 

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/rule.png" width="70%" height="70%">
</p>

At the end of the game, each player counts their score based on the cards they have collected, using the scoring system. The scoring system is as follows: any player with at least 7 clubs receives 7 points. Each Jack and each Ace is worth 1 point. There are also two key cards: the 10 of Diamonds, which is worth 3 points, and the 2 of Clubs, which is worth 2 points. 

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/score.png" width="30%" height="30%">
</p>

Here is an example of a full game being played. Notice the 6 rounds, and the fact that at the end of each round all the cards remaining in the pool carry over to the next round. Also, both players end each round with empty hands and are then dealt 4 new cards for the next round.

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/full_game.png" width="90%" height="90%">
</p>

And this is a basic but key observation that we will use to represent the full game tree, which has on average 2 to the power of 30 nodes, in a more compact way. In other words, if for two terminal nodes of the k-th round the pool carried over to the next round is the same, and the accumulated scores for Alex and Bob up to that point are also the same, then we may potentially consider the resulting root node of the next round to be identical. Notice that the score data we need to keep track of includes the number of club cards held by Alex and Bob, as well as the point difference from the point cards. We also add an extra value to the score to record whether Alex or Bob has accumulated at least 7 clubs by that terminal node of the round. In that case, we reset the number of clubs for both players to zero and set this new index to 1 or 2, depending on whether Alex or Bob was the one who collected the 7 clubs.

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/idea.png" width="80%" height="80%">
</p>

Based on this observation, we represent each node of the full game tree using two tensors: one is the score tensor, and the other is the game state tensor. The game state consists of the cards held by Alex and Bob, the cards in the pool, and the action history within the round. Note that at the beginning of each round, the game state resets to only reflect the card status.

This process is explained more clearly in the figure below. We represent the inherited scores from previous rounds using colored arrows going into a node. We call the left-hand side the Game Tree and the right-hand side the Full Game Tree. Notice that the branching factors are shown by the underlying lines. For example, here the branching factor is [2, 3]. 

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/unfolding.png" width="80%" height="80%">
</p>

We need to explain the unfolding process—how to convert the Game Tree into the Full Game Tree. For now, let me mention the 4 main tensors used throughout. The first is the game tensor _t_gme_, which encodes the card states and action history within each round. The second is the score tensor _t_scr_, which encodes all the unique scores inherited from previous rounds. And finally, the Full Game Tree tensor, _t_fgm_, where a _[g, s]_ entry means that the g-th row of the Game Tree has inherited the s-th score from _t_scr_. Connections between the layers of the Full Game Tree are also encoded in the _t_edg_ tensor, as shown in the figure above.

The next figure shows the game tree of height 48. Notice that in the first round, all the incoming arrows are colored red, because at the start of the game all scores are identically zero. In other words, nobody has collected any cards at that point. Throughout, we refer to this as the Game Tree (GT) and the Pasur Full Game Tree as shown above the Full Game Tree (FGT). Note that FGT is preserved through the full game tensor _t_fgm_ and also the score tensor _t_scr_. 

<p align="center">
<img src="https://sinabaghal.github.io/images/GT.png" width="80%" height="100%">
</p>
