---
title: "Feynman-Kac Theorem"
permalink: /pages/feyman_kac/
author_profile: true
---

Feynman-Kac Theorem states that the price of a derivative can be represented as the solution to a certain PDE. Specifically, if $f(t, S)$ is the price of the derivative
at time $t$ when the underlying asset price is S which follows 

$$
dS(t) = rS(t)dt + σS(t)d \tilde{W}(t)
$$

,then $f(t, S)$ solves the following PDE:

$$
 \frac{\partial f}{\partial t} + rS \frac{\partial f}{\partial S} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 f}{\partial S^2} = r f
 $$
 
with the terminal condition $f(T, S) = h(S)$, where $h(S)$ is the payoff at maturity $T$. Document below provides some concrete examples on how Feynman-Kac Theorem can be used for pricing derivatives!

[Document (Partial Differential Equations for Pricing)](https://sinabaghal.github.io/files/notes/Feynman_Kac.pdf)
