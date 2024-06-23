---
title: "Implied Volatility "
permalink: /pages/implied_volatility/
author_profile: true
---

Implied volatility is a crucial concept in financial markets, reflecting the market's forecast of a security's price fluctuation over a specific period. It is derived from the market price of an option and represents the volatility that, when input into the Black-Scholes model, yields the observed market price. Implied volatility is a key metric for traders and investors in the assessment of market expectations and the pricing of options. Recall the Black-Scholes formula:

$$
c(\sigma) = SN(d_+)-Ke^{-r\tau}N(d_-) \text{ where } d_{\pm} = \tfrac{1}{\sigma\sqrt{\tau}}[\log \tfrac{S}{K}+(r\pm\tfrac{\sigma^2}{2})\tau]
$$

In reality, stock price movements do not conform to geometric Brownian motion. Empirical evidence shows that stock log returns exhibit features like stochastic volatility and the leverage effect, which are not accounted for by this equation. Despite these limitations, the Black-Scholes model remains widely used in practice due to its straightforward pricing formula. The complexity of modeling is instead focused on the input volatility parameter, 
$$\sigma$$. Consequently, when its limitations are acknowledged, the BS formula serves as a useful tool for interpreting market prices.