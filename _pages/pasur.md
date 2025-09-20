---
title: "Solving Pasur Using GPU-Accelerated CFR — Under Construction 🚧🚧🚧🚧🚧🚧🚧"
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

## Table of Contents
- [Counterfactual Regret Minimization (CFR)](#counterfactual-regret-minimization-(cfr))
- [Pasur](#pasur)
- [Folding](#folding)
- [Pytorch Framework](#pytorch-framework)
  - [Game Tensor](#game-tensor)
  - [In-Hand Updates](#in-hand-updates)
  - [Between-Hand Updates](#between-hand-updates)
  - [Action Tensors](#action-tensor)
  - [Numeric Actions](#numeric-actions)


## Counterfactual Regret Minimization (CFR)


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

## Pasur

Let me explain the game itself next! 

Pasur is played in `6` rounds, and in each round each player is dealt `4` cards, which they play sequentially over `4` turns, taking turns one after another. At each turn, a player places a card face up and either lays it in the pool or collects it along with some other pool cards, according to the rule shown in this table. Figure below shows an example of the first turn by Alex and Bob.

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/ex.png" width="70%" height="70%">
</p>

Let me explain the rule and scoring system in more detail. If the played card is numeric, it can collect any subset of cards whose values add up to `11`. If it is a Jack, it can collect all cards in the pool except for Kings and Queens. If it is a King, it can only collect a single King, and if it is a Queen, it can only collect a single Queen. 

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/rule.png" width="70%" height="70%">
</p>

At the end of the game, each player counts their score based on the cards they have collected, using the scoring system. The scoring system is as follows: any player with at least `7` clubs receives `7` points. Each Jack and each Ace is worth `1` point. There are also two key cards: the `10` of Diamonds, which is worth `3` points, and the `2` of Clubs, which is worth `2` points. 

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/score.png" width="30%" height="30%">
</p>

Here is an example of a full game being played. Notice the `6` rounds, and the fact that at the end of each round all the cards remaining in the pool carry over to the next round. Also, both players end each round with empty hands and are then dealt `4` new cards for the next round.

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/full_game.png" width="90%" height="90%">
</p>

## Folding

And below is a basic but key observation that we will use to represent the full game tree, which has on average `2` to the power of `30` nodes, in a more compact way. In other words, if for two terminal nodes of the k-th round the pool carried over to the next round is the same, and the accumulated scores for Alex and Bob up to that point are also the same, then we may potentially consider the resulting root node of the next round to be identical. Notice that the score data we need to keep track of includes the number of club cards held by Alex and Bob, as well as the point difference from the point cards. We also add an extra value to the score to record whether Alex or Bob has accumulated at least `7` clubs by that terminal node of the round. In that case, we reset the number of clubs for both players to zero and set this new index to `1` or `2`, depending on whether Alex or Bob was the one who collected the `7` clubs.

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/idea.png" width="80%" height="80%">
</p>

Based on this observation, we represent each node of the full game tree using two tensors: one is the score tensor, and the other is the game state tensor. The game state consists of the cards held by Alex and Bob, the cards in the pool, and the action history within the round. Note that at the beginning of each round, the game state resets to only reflect the card status.

## Pytorch Framework

This process is explained more clearly in the figure below. We represent the inherited scores from previous rounds using colored arrows going into a node. We call the left-hand side the Game Tree and the right-hand side the Full Game Tree. Notice that the branching factors are shown by the underlying lines. For example, here the branching factor is `[2, 3]`. 

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/unfolding.png" width="110%" height="110%">
</p>

We need to explain the unfolding process—how to convert the Game Tree into the Full Game Tree. For now, let me mention the 4 main tensors used throughout. The first is the game tensor `t_gme`, which encodes the card states and action history within each round. The second is the score tensor `t_scr`, which encodes all the unique scores inherited from previous rounds. And finally, the Full Game Tree tensor, `t_fgm`, where a `[g, s]` entry means that the g-th row of the Game Tree has inherited the s-th score from `t_scr`. Connections between the layers of the Full Game Tree are also encoded in the `t_edg` tensor, as shown in the figure above.

The next figure shows the game tree of height 48. Notice that in the first round, all the incoming arrows are colored red, because at the start of the game all scores are identically zero. In other words, nobody has collected any cards at that point. Throughout, we refer to this as the Game Tree (GT) and the Pasur Full Game Tree as shown above the Full Game Tree (FGT). Note that FGT is preserved through the full game tensor `t_fgm` and also the score tensor `t_scr`. 

<p align="center">
<img src="https://sinabaghal.github.io/images/GT.png" width="100%" height="100%">
</p>

At this stage, I need to explain several things, including how PyTorch tensors are used to represent game states. I also need to explain how these tensors are updated throughout the process.

### Game Tensor

So let me explain the game state tensors. Each layer of the game tree is represented by a tensor of shape `M × 3 × m`. Here, `M` is the number of nodes in that layer, and `m` is the number of active cards. A card is inactive if it has not yet been played or if it has already been collected across all terminal nodes of the previous rounds.

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/map_idx.png" width="80%" height="100%">
</p>

So let’s look at a slice of a game tensor. Each slice corresponds to a node in the game tree and contains 3 rows. 

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/t_gme_0.png" width="80%" height="100%">
</p>

The first row encodes card ownership: 1 means Alex has the card, 2 means Bob has the card, and 3 means the pool contains the card. A value of 0 means the card has already been collected or is only active in another node of the game tree.

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/t_gme_01.png" width="80%" height="100%">
</p>

The second row encodes Alex’s action history. A value of 1 is added if a card is laid on his first turn, and 10 if it is picked on his first turn. Values of 2 and 20, 3 and 30, 4 and 40 are used similarly for the later turns.

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/t_gme_1.png" width="80%" height="80%">
</p>

Notice that a value of 41, for example, means the card was laid on the first turn and picked on the fourth turn. Similarly, a card that is laid and picked in the same turn can be identified by a value of 2 for that turn in the second row, while the corresponding value in the first row of the game tensor is 0.

Keeping track of inactive cards results in significant memory savings, as tensor sizes are kept optimally small. We also maintain padding tensors to help consolidate different tensor shapes whenever needed.

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/dyn_shape.png" width="80%" height="100%">
</p>

Next, I will explain how game tensors are updated throughout the process. There are two types of updates: in-hand updates and between-hand updates.

### In-Hand Updates

Let us begin with the in-hand updates. Consider the example above with two game nodes, where the inherited scores are `3` and `2`. In this case, the count score tensor _c_scr_ is given by `[3,2]`. Assume further that the branching factors are 2 and 3; `t_brf = [2,3]`. This setup corresponds to the figure on the left-hand side. Observe the count score tensor alongside the branch factor tensor. The complete game tensor for the first row of the game tree is also displayed. The right-hand side illustrates the unfolded Full Game Tree. The unfolding is constructed in the most natural way: each node of the game tree is repeated with the same label but shown in a different color, depending on the incoming colors within the tree. The next step is to identify the edge tensor, which maps each node in the second layer of the Full Game Tree to its corresponding node in the first layer. 

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/in_hand.png" width="110%" height="110%">
</p>

Here, the $$\otimes$$ notation refers to the PyTorch repeat_interleave operation. I also use an extension of repeat_interleave, which I call repeat blocks. This operation divides the source tensor into chunks and repeats each chunk a specified number of times. Notice that only nodes with matching colors are connected. 

A simple inspection shows that the edge tensor is constructed using the displayed formula via the repeat-blocks process. Block sizes are determined by the count score tensor, and the repeat tensor is given by the branching factor. The first block `[0, 1, 2]` is repeated twice, while the last block `[3, 4]` is repeated three times, since its branching factor is `3`. Similarly, column 1 of the FGT is updated. Recall that this column encodes the color of each node in the FGT. 

After these two updates, we proceed to update column 0 of the FGT tensor. Before doing so, the count score and branching factor are updated. Note that the actions will be applied to the game tensor at a later stage.

### Between-Hand Updates

Now we explain the between-hand updates. Remember that at the end of each round, the pool and the inherited scores are the only items that matter. We discard all other information in the game tensor except for the pool formation. We then find the unique pool formations using `torch.unique`. Notice that there is no need to sort the output, but we do need to record the inverse mapping from the unique operation. We apply the same procedure for the running score tensor `t_rus`.

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/bet_hand_0.png" width="110%" height="110%">
</p>

We find the new set of unique scores to be inherited by the nodes of the next round. At each node of the full game tree, we currently have two scores: one inherited from previous rounds, recorded in column 1 of the full game tensor, and the other being the running score accumulated during the current round. The resulting pair is encoded in the tensor shown here. We then apply a unique operation on this set of pairs, sum the corresponding scores, and apply unique again to determine the set of scores to be passed to the next round.

Finally, we establish the linkage between the two rounds for the full game tree. To do this, we first determine the pair index and then locate where that pair is mapped inside the score tensor. Similarly, the index of the game state can be found, as shown in the second-to-last line of the displayed code snippet.

The game tensor is then populated using the newly dealt cards at the root level of the next round, while the full game tensor already reflects the unfolding process for that root level.

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/bet_hand_1.png" width="120%" height="120%">
</p>


Notice that the linkage tensor is used to propagate back the utilities computed for each full game tree during the CFR algorithm. Since CFR is trained on the full game tree, we need to pass back the calculated utilities from the root level of each round to the terminal level of the preceding round.

### Action Tensor

Next, I explain how the action tensor is constructed. Note that the inherited score is not important when computing actions at each game tensor node. The action tensor has shape `M' × 2 × m`, where `M'` is the number of nodes in the next layer of the game tree. The first row encodes the hand cards used in each action, and the second row encodes the pool cards used.


<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/t_act.png" width="110%" height="110%">
</p>

Once the action tensor   `t_act` is obtained, the game tensor `t_gme` is constructed as follows. We first identify the pick actions, meaning those in which at least one card from the pool is selected. According to the rules of the game, whenever such a pick occurs, both the laid card and the selected pool cards are collected by the player. If no pick occurs, the laid card is simply added to the pool and removed from the player’s hand.  `t_act[*,1:2,:]` is updated via the procedure explained above, that is lay (weighted by `i_trn+1`) plus pick (weighted by `10 x (i_trn+1)`).

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/t_act_1.png" width="110%" height="110%">
</p>

Actions are divided into four categories: Numeric, Jack, King, and Queen actions. For example, since numeric actions do not involve face cards, when constructing them we filter the input game tensor to mask out all cards that are not numeric. We denote the resulting tensor as `t_2x40`. Note that, due to the dynamic shaping discussed earlier, `t_2x40` does not necessarily have shape 40; the notation is used for clarity.  

Once `t_2x40` is obtained, we apply `torch.unique` without sorting to obtain `tu_2x40`, while also recording the reverse indices. For numeric actions in particular, we separate pick and lay actions and construct the resulting action tensors individually. These action tensors are denoted by `tu_pck` and `tu_lay`, along with the corresponding count tensors `cu_pck` and `cu_lay`. Similarly, we obtain `tu_jck`, `tu_kng`, `tu_qun`, `cu_jck`, `cu_kng`, and `cu_qun`.  

Afterward, we apply padding to restore all tensors to the original shape of `t_gme`. In other words, `tu_pck.shape[2] = t_gme.shape[2]`, and so on. Next, we reverse the unique operation. The details of this step are omitted here, but we refer you to the full paper for further explanation.  

At this stage, we have the tensors `t_pck`, `t_lay`, `t_jck`, `t_kng`, `t_qun`, `c_pck`, `c_lay`, `c_jck`, `c_kng`, and `c_qun`. Finally, we concatenate the action tensors in such a way that all actions corresponding to each node of the game tree are grouped together. To achieve this, we concatenate these action tensors and then use the sorting indices obtained from the count tensors to shuffle the concatenated result, yielding `t_act`. The figure below summarizes this operation. The branching factor `t_brf` is constructed as shown.

<p align="center">
<img src="https://sinabaghal.github.io/files/pasur/t_act_scheme.png" width="110%" height="110%">
</p>

### Numeric Actions




