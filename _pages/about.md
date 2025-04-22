---
permalink: /
title: "About me"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---
I am a data scientist, quantitative analyst, and former mathematician with a passion for applying my expertise in data science and quantitative finance. My professional experience includes roles as a data scientist, machine learning researcher, and quantitative analyst.

I earned my PhD in Mathematics from the University of Waterloo in 2021, and I hold both a master's and a bachelor's degree in Mathematics from Sharif University of Technology. My [PhD research](https://uwspace.uwaterloo.ca/handle/10012/16872) focused on Stochastic Optimization, while my earlier studies delved into Fundamental Mathematics.

# Projects 

### Generative Modeling of Heston Volatility Surfaces Using Variational Autoencoders

This project focuses on training a Variational Autoencoder (VAE) to produce Heston volatility surfaces. The Heston model is used in stochastic volatility option pricing models. Once trained, this VAE can generate new volatility surfaces, which could be useful for various financial applications such pricing exotic derivatives. To see project's page, click [**here**](https://sinabaghal.github.io/vae4heston/). You can find the code for this project [**here**](https://github.com/sinabaghal/VariationalAutoEncoderforHeston). 

![](http://sinabaghal.github.io/images/part1.gif)


### Implementing Deep Smoothing for Implied Volatility Surfaces (IVS)

This project is a python-based implementation of the methodologies presented in the paper [*Deep Smoothing of the Implied Volatility Surface*](https://arxiv.org/pdf/1906.05065) with different aspects related to the neural network training, convergence behavior, and associated implementation details developed independently.  

A volatility surface is a representation of implied volatility across different strike prices and maturities, crucial for pricing and hedging options accurately. Ensuring the surface is arbitrage-free is essential to prevent inconsistencies that could lead to riskless profit opportunities, which would undermine the model’s reliability in real markets. Deep smoothing focuses on applying deep learning methods to generate smooth, arbitrage-free implied volatility surfaces which at a high-level can be summarized as follows: Imagine you are given a set of points $$(k, \tau, iv)$$ representing market data. These points form part of a 3D surface that we aim to construct with two key objectives:

- We need to fit the market data.  
- We must ensure that the surface meets certain curvature properties. In quantitative finance, this means creating an arbitrage-free surface. Mathematically, this translates to minimizing a specified loss function across the entire surface.

The convergence To see project's page, click [**here**](https://sinabaghal.github.io/deepsmoothing/). You can find the code for this project [**here**](https://github.com/sinabaghal/deepsmoothigIVS). 

<p align="center">
<img src="http://sinabaghal.github.io/images/header.png" width="150%" height="150%">
</p>

### Quantitative Finance and Option Pricing 

Once upon a time, I became interested in quantitaive finance, in particular Option Pricing. Stochastic models are used for option pricing where the underlying uncertainly is driven by a class of continou time martingales known as [**Wiener processes**](https://en.wikipedia.org/wiki/Wiener_process). The classic source for learning about options and how they are priced is Stochastic Calculus for Finance II written by Shreve. I compiled my own solutions to all of this book's exercises and you could find them [**here**](https://sinabaghal.github.io/shreve/).


