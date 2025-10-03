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

In reinforcement learning, there is an **agent** and an **environment**.  At time step $$t$$, the state is denoted by $$s_t$$.  Given state $$s_t$$, the agent takes an action $$a_t$$.  This results in a reward value $$r_t := r(s_t, a_t)$$.  

The agent’s decision-making is parameterized by a policy $$\pi_\theta$$,  where $$\pi_\theta(\cdot \mid s_t)$$ defines a probability distribution over possible actions at time $$t$$,  given the state $$s_t$$.  

The goal of an RL algorithm is not just to maximize the immediate reward,  but to maximize the **expected cumulative reward**:  

$$
\max_\theta \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \right],
$$

where $$0 \leq \gamma < 1$$ is a discount factor. 
