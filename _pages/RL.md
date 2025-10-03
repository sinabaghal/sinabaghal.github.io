---
title: "Basics of Reinforcement Learning"
permalink: /rl/
author_profile: true
tags:
  - Reinforcement Learning
  - Machine Learning
---

This tutorial provides an introduction to the fundamentals of reinforcement learning.  The main reference is the [video lecture series](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps) by Sergey Levine.

## Table of Contents
- [Goal of RL](#goal_of_rl)
- [Imitation Learning](#imitation_learning)

## Goal of RL

In reinforcement learning, there is an _agent_ and an _environment_.  At time step $$t$$, the state is denoted by $$s_t$$.  Given state $$s_t$$, the agent takes an action $$a_t$$ resulting in a reward value $$r_t := r(s_t, a_t)$$.  

<br>
<p align="center">
<img src="https://sinabaghal.github.io/files/RL/00.png" width="70%" height="70%">
</p>
<br>

Agent’s _policy_ is parameterized by $$\pi_\theta$$,  where $$\pi_\theta(\cdot \mid s_t)$$ defines a probability distribution over possible actions at time $$t$$,  given the state $$s_t$$.  

The goal of an RL algorithm is to maximize the _expected cumulative reward_:  

$$
\text{argmax}_\theta \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^T \gamma^t r(s_t, a_t) \right],
$$

where $$0 \leq \gamma < 1$$ is a discount factor. Notice that:  

- More weight is placed on earlier steps. 
- Objective $$\mathbb{E}_{\pi_\theta}$$ is a smooth function of $$\theta$$ where the reward function $$r$$ itself may be non-smooth (e.g., $$r \in \{\pm 1\}$$).
- $$s_t$$ is independent of $$s_{t-1}$$ (_Markov Property_)

Graphical model below describes the relationships between states and actions:

<br>
<p align="center">
<img src="https://sinabaghal.github.io/files/RL/01.png" width="70%" height="70%">
</p>
<br>

## Imitation Learning

The analogous concept in reinforcement learning, compared to supervised learning, is called _imitation learning_, where the agent learns by mimicking expert actions.  However, imitation learning often does not work well in practice due to the _distributional shift problem_.  This arises because, in supervised learning, samples are assumed to be _i.i.d._, while in reinforcement learning the agent’s past actions affect future states.  

To formalize this, assume that $$\pi^*$$ is the expert policy and the learned policy $$\pi_\theta$$ makes an error with probability at most $$\epsilon$$ under the training distribution:  

$$
\Pr_{s_t \sim p_{\text{train}}}\big[\pi_\theta(s_t) \neq \pi^*(s_t)\big] \leq \epsilon.
$$

Based on this, write 

$$
p_{\theta}(s_t) = (1-\epsilon)^t p_{\text{train}}(s_t)+(1-(1-\epsilon)^t)p_{\text{mistake}}(s_t)
$$

Denote $$c_t(s_t, a_t) = 1_{\{a_t \neq \pi^*(s_t)\}} \in \{0, 1\}$$. Then, the total number of times the policy $$\pi_\theta$$ deviates from the optimal policy grows quadratically with $$T$$.

$$
\begin{aligned}
\mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^T c(s_t,a_t) \right] &= \sum_{t=0}^T\int p_\theta(s_t)c(s_t,a_t) ds_t \\
&=  \sum_{t=0}^T (1-\epsilon)^t\cdot\int p_{\text{train}}(s_t)c(s_t,a_t) ds_t + \sum_{t=0}^T (1-(1-\epsilon)^t)\cdot\int p_{\text{mistake}}(s_t)c(s_t,a_t) ds_t  \\ 
&\leq  \sum_{t=0}^T (1-\epsilon)^t\cdot\epsilon + \sum_{t=0}^T 1-(1-\epsilon)^t \\ 
&\leq  \sum_{t=0}^T (1-\epsilon)^t\cdot\epsilon + 2\epsilon\cdot\sum_{t=0}^T t \\ 
&= \epsilon\cdot\mathcal{O}(T^2)
\end{aligned}
$$

This bound is achieved in the _tightrope walking_ problem, where the agent must learn to go straight; otherwise, it will enter unknown territory. Imitation learning can still be useful with some modifications, such as including bad actions along with corrective steps to be taken afterward.



