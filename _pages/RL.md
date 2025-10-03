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
\text{argmax}_\theta \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \right],
$$

where $$0 \leq \gamma < 1$$ is a discount factor. Notice that:  

- More weight is placed on earlier steps. 
- Objective $$\mathbb{E}_{\pi_\theta}$$ is a smooth function of $$\theta$$ where the reward function $$r$$ itself may be non-smooth (e.g., $$r \in \{\pm 1\}$$).
- $s_t$ is independent of $$s_{t-1}$$ (_Markov Property_)




