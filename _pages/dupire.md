---
title: "The Dupire Formula"
permalink: /pages/dupire/
author_profile: true
---


Dupire's formula is a fundamental result in quantitative finance that provides a framework for modeling the local volatility of an asset. Introduced by Bruno Dupire in 1994, this formula relates the local volatility of an underlying asset to the prices of European call and put options across different strikes and maturities. The key insight of Dupire's approach is that the market-implied volatility, which is often observed to vary with strike price and time to maturity, can be translated into a local volatility function. This local volatility function is deterministic and can be used to construct a risk-neutral process for the underlying asset that matches observed market prices of European options. Dupire's formula is expressed as a partial differential equation, and solving this equation allows for the calibration of the local volatility surface, providing a more accurate and dynamic way to price and manage derivatives compared to models assuming constant volatility. This approach has significant implications for risk management and the development of more sophisticated trading strategies. Assuming no dividend payments, Dupire's formula is given by:

$$
\sigma_{\text{loc}}^2(K, T) = \frac{\frac{\partial C}{\partial T} + r K \frac{\partial C}{\partial K}}{\frac{1}{2} K^2 \frac{\partial^2 C}{\partial K^2}}
$$

The document below provides a proof of this formula. 

[Document (The Dupire Formula)](https://sinabaghal.github.io/files/notes/dupire.pdf)


