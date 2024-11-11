---
title: "Deep Smoothing of IVS"
permalink: /deepsmoothing/
author_profile: true
tags:
  - Deep Learning
  - Feed Forward Networks
  - Volatility Surfaces
  - Vectorization
  - SSVI
  - CVX
---
---

This project ([Github repository](https://github.com/sinabaghal/deepsmoothigIVS)) is a python-based implementation of the methodologies presented in the paper *Deep Smoothing of the Implied Volatility Surface* by Ackerer et al (2020). The main idea is to use feedforward neural networks as a corrective tool to modify the prior models considered for volatility surfaces. By letting

$$
\begin{equation}
\omega(k,\tau; \theta):= \omega_{\text{prior}}(k,\tau;\theta_{prior})\cdot\omega_{\text{nn}}(k,\tau;\theta_{nn}),
\end{equation}
$$

we enrich the space of parameters used for fitting the volatility surface. Here $$\omega$$ is the implied variance and $$\theta_{\text{prior}}$$ and $$\theta_{\text{nn}}$$ are two disjoint set of parameters. Furthermore, to ensure our volatility surface is free of arbitrage, we use the ideas from _Arbitrage-free SVI Volatility Surfaces_ (2013) where they argue that if the following are satisfied then the call price surface is free of Calendar & Butterfly arbitrage resp.

$$
\begin{align*}
\ell_{cal}&:=\partial_\tau \omega(k,\tau) \geq 0 \\
    \ell_{but}&:=\left(1-\frac{k\partial_k \omega(k,\tau)}{2\omega(k,\tau)} \right)^2-\frac{\partial_k \omega(k,\tau)}{4}\left(\frac{1}{\omega(k,\tau)}+0.25 \right)+\frac{\partial^2_{kk}\omega(k,\tau)}{2}\geq 0  
\end{align*}
$$

Here, we will walk through the main steps of the deep smoothing framework.

## Notation and Initial Values

European call options with the following table of notation and values are used:

<div align="center">

| **Parameter**             | **Value/Definition**                                 |
|---------------------------|------------------------------------------------------|
| Spot Price                | spot = $1$                                           |
| Strike                    | $K$                                                  |
| Interest Rate             | rate $= 0.0$                                         |
| Dividend Rate             | $q = 0.0$                                            |
| Forward Price             | $F_t$                                                |
| Forward Log Moneyness     | $k = \log \frac{K}{F_t}$                             |
| Implied Volatility        | $\sigma(k, \tau)$                                    |
| Implied Variance          | $\omega(k, \tau) := \sigma(k, \tau)^2 \tau$          |

</div>

**Table:** Parameters and Definitions

## Bates model

Next, let’s review the Bates model.

$$
\begin{align*}
\frac{dS_t}{S_t} &= (r - \delta) \, dt + \sqrt{V_t} \, dW_{1t} + dN_t \\
dV_t &= \kappa (\theta - V_t) \, dt + \sigma \sqrt{V_t} \, dW_{2t} 
\end{align*}
$$

$$dW_{1t}dW_{2t} = \rho dt$$ and $$N_t$$ is a compound Poisson process with intensity $$\lambda$$ and independent jumps $$J$$ with

$$
\ln(1 + J) \sim \mathcal{N} \left( \ln(1 + \beta) - \frac{1}{2} \alpha^2, \alpha^2 \right)
$$

### Characteristic Function

The characteristic function of the log-strike in the Bates model is given by:

$$
\begin{align*}
\phi(\tau,u) &= \exp\left(\tau \lambda\cdot \left(e^{-0.5\alpha^2u^2+iu\left(\ln(1+\beta)-0.5\alpha^2\right)}-1\right)\right) \\
&\cdot \exp\left(\frac{\kappa\theta \tau(\kappa-i\rho\sigma u)}{\sigma^2}+iu\tau(rate-\lambda\cdot \beta)+iu\cdot\log spot\right) \\
&\cdot \left(\cosh \frac{\gamma \tau}{2}+\frac{\kappa-i\rho\sigma u}{\gamma}\cdot\sinh \frac{\gamma \tau}{2}\right)^{-\frac{2\kappa \theta}{\sigma^2}} \\
&\cdot \exp\left(-\frac{(u^2+iu)v_0}{\gamma \coth \frac{\gamma \tau}{2}+\kappa-i\rho\sigma u}\right).
\end{align*}
$$

### Fast Fourier Transform

The option price calculation for the Bates model can be efficiently computed using an analytical formula that leverages FFT. Following [Carr and Madan, 1999], we apply a smoothing technique to be able to compute the FFT integral. Recall that the option value is given by 

$$
C_\tau(k) = spot \cdot \Pi_1 - K e^{-rate \cdot \tau} \cdot \Pi_2
$$

where 

$$
\begin{align*}
\Pi_1 &= \frac{1}{2} + \frac{1}{\pi} \int_0^\infty \text{Re} \left[ \frac{e^{-iu \ln(K)} \phi_\tau(u - i)}{iu \phi_\tau(-i)} \right] du \\
\Pi_2 &= \frac{1}{2} + \frac{1}{\pi} \int_0^\infty \text{Re} \left[ \frac{e^{-iu \ln(K)} \phi_\tau(u)}{iu} \right] du
\end{align*}
$$

Since the integrand is singular at the required evaluation point $$u = 0$$, FFT cannot be used to evaluate call price $$C_\tau(k)$$. To offset this issue, we consider the modified call price $$c_\tau(k) := \exp(\alpha k) C_\tau(k)$$ for $$\alpha > 0$$. Denote the Fourier transform of $$c_\tau(k)$$ by

$$
\Psi_\tau(v) = \int_{-\infty}^{\infty} e^{ivk} c_\tau(k) \, dk \Rightarrow  C_\tau(k) = \frac{\exp(-\alpha k)}{\pi} \int_0^\infty e^{-ivk} \Psi_{\tau}(v) \, dv
$$

It can be shown that 

$$
\Psi_\tau(v) = \frac{e^{-r \tau} \phi_\tau(v - (\alpha + 1)i)}{\alpha^2 + \alpha - v^2 + i(2\alpha + 1)v}
$$

#### FFT Setup

We set up the FFT calculation as follows:

- Log strike levels range from $$-b$$ to $$b$$ where 

$$
b = \frac{Ndk}{2}
$$

- $$\Psi_\tau(u)$$ is computed at the following $$v$$ values:

$$
v_j = (j-1)du \text{ for } j = 1, \cdots, N
$$

- Option prices are computed at the following $$k$$ values:

$$
k_u = -b + dk(u-1) \text{ for } u = 1, \cdots, N
$$

- To apply FFT, we need to set

$$
dk \cdot du = \frac{2\pi}{N}
$$

- Simpson weights are used:

$$
3 + (-1)^j - \delta_{j-1} \text{ for } j = 1, \cdots, N
$$

Having this setup ready, call prices are obtained as follows:

$$
C(k_u) = \frac{\exp(-\alpha k_u)}{\pi} \sum_{j=1}^{N} e^{-i \frac{2\pi}{N} (j-1)(u-1)} e^{ibv_j} \Psi(v_j) \frac{\eta}{3} \left(3 + (-1)^j - \delta_{j-1}\right)
$$

## Loss Function

We define 4 different loss functions and construct the total loss function as a linear combination of these with coefficients being the penalty parameters. The first loss function is the prediction error, which is defined as below:

$$
\begin{align*}
\mathcal{L}_0(\theta) &= \sqrt{\frac{1}{\vert \mathcal{I}_0 \vert} \sum_{(\sigma,k,\tau) \in \mathcal{I}_0} \left(\sigma - \sigma_\theta(k,\tau)\right)^2}
\end{align*}
$$

Another condition required to guarantee a free-arbitrage VS is the large moneyness behaviour which states that $$\sigma^2(k,\tau)$$ is linear for $$k\to \pm \infty$$ for every $$\tau>0$$. Roper achieves this by imposing $$\frac{\sigma^2(k,\tau)}{\vert k \vert}<2$$ which in turn is achieved by minimizing the following 

$$
\begin{equation}
    \frac{\partial^2 \omega(k,\tau)}{\partial k \partial k}  
\end{equation}
$$

The above three constraints along with the prediction error are used to shape the loss function utilized in training the implied variance $$\omega$$. 



For learning rate scheduling, a slightly different approach is taken compared to Ackerer et al. The following table summerizes the convergence techniques used for training:


| Checkpoint Interval                          | A checkpoint is set every 500 epochs.                                                                                                                                                                                   |
|----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Bad Initialization                           | After the first 4 checkpoints, if the best loss is not below 1, the model is reinitialized.                                                                                                                               |
| Learning Rate Adjustment                     | Every 4 checkpoints, if the best loss is not below 0.05, the learning rate is reset to its initial value.                                                                                                                 |
| Weights Perturbation                         | After each checkpoint, regardless of other conditions, the weights of the model are perturbed. This is to help escape local minima.                                                                                        |
| Divergence Handling (Bad Perturbation)       | If the current loss is at least 10% worse than the best loss so far and > 0.1, and this occurs after the first checkpoint, the models are reloaded from the last saved state, and training continues from the last checkpoint with the best loss value. |



Numerical results show that the enhanced model, incorporating a neural network with the loss function $$\omega(k,\tau; \theta)$$ with SSVI as prior, fits the Bates model data perfectly and produces an arbitrage-free volatility surface. The figure below shows how well the model fits the training data.

<p align="center">
<img src="http://sinabaghal.github.io/images/ref_V.png" width="100%" height="100%">
</p>

The figure below illustrates 29 trained implied volatility surfaces obtained using the deep smoothing algorithm for long term maturies obly _i.e.,_ $$\tau>$$ 20 days. Different values for the Bates model parameters are used in each case. Displayed parameters are $$\alpha, \beta, \kappa, v0, \theta, \rho, \sigma, \lambda$$ respectively. If you cannot guess the definition of these parameters see the technical report inside the github repository. 

<p align="center">
<img src="http://sinabaghal.github.io/images/all_models_long.png" width="80%" height="100%">
</p>

The figure below displays the same volatility surfaces for short term maturities  _i.e.,_ $$\tau<=$$ 20 days.

<p align="center">
<img src="http://sinabaghal.github.io/images/all_models_short.png" width="80%" height="100%">
</p>

The following also is an example of the training trajectory where a feedforward neural network with 4 hidden layers, with 40 units in each layer. 

<p align="center">
<img src="http://sinabaghal.github.io/images/train_metrics.png" width="100%" height="100%">
</p>

See the technical report inside the Github repo for more details. 


