---
title: "Deep Smoothing of IVS"
permalink: /deepsmoothing/
author_profile: true
---

This project implements the methodologies presented in the paper *Deep Smoothing of the Implied Volatility Surface* by Ackerer et al (2020). The main idea is to use feedforward neural networks as a corrective tool to modify the prior models considered for volatility surfaces. By letting

$$
\begin{equation}
\omega(k,\tau; \theta):= \omega_{\text{prior}}(k,\tau;\theta_{prior})\omega_{\text{nn}}(k,\tau;\theta_{nn}),
\end{equation}
$$

we enrich the space of parameters used for fitting the volatility surface. Here $$\theta_{\text{prior}}$$ and $\theta_{\text{nn}}$ are two disjoint set of parameters. Furthermore, to ensure our volatility surface is free of arbitrage, we use the ideas by Roper (2010) which argues that if the following are satisfied then the call price surface is free of Calendar \& Butterfly arbitrage resp.

$$
\begin{align*}
\ell_{cal}&=\partial_\tau \omega(k,\tau) \geq 0 \\
    \ell_{but}&=\left(1-\frac{k\partial_k \omega(k,\tau)}{2\omega(k,\tau)} \right)^2-\frac{\partial_k \omega(k,\tau)}{4}\left(\frac{1}{\omega(k,\tau)}+0.25 \right)+\frac{\partial^2_{kk}\omega(k,\tau)}{2}\geq 0  
\end{align*}
$$

Another condition required to guarantee a free-arbitrage VS is the large moneyness behaviour which states that $\sigma^2(k,\tau)$ is linear for $k\to \pm \infty$ for every $\tau>0$. Roper achieves this by imposing $\frac{\sigma^2(k,\tau)}{\vert k \vert}<2$ which in turn is achieved by minimizing the following 

$$
\begin{equation}
    \frac{\partial^2 \omega(k,\tau)}{\partial k \partial k}  
\end{equation}
$$

We use the above three constraints to shape the loss function utilized in training the implied variance $\omega$. As for learning rate scheduling, we use a slightly different approach than Ackerer et al;  we employ model weights perturbation along with a divergence handling scheme. 


Numerical results show that our enhanced model, incorporating a neural network with the loss function $\omega(k,\tau; \theta)$, fits the Bated model data perfectly and produces an arbitrage-free volatility surface. 


The figure on the left shows how well our model fits the training data, while the figure on the right displays the log of the implied surface generated by our model.

<p align="center">
    <img src="http://sinabaghal.github.io/images/ref_V.png" width="45%" height="45%" style="display:inline-block;" />
    <img src="http://sinabaghal.github.io/images/ref_VOl.png" width="45%" height="45%" style="display:inline-block;" />
</p>

Python code can be found at the following link:

For convenience, we include the main body of the code below. Please ensure you have utils.py from above link available to run it successfully.