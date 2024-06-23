### Implied Volatility 

Implied volatility is a crucial concept in financial markets, reflecting the market's forecast of a security's price fluctuation over a specific period. It is derived from the market price of an option and represents the volatility that, when input into the Black-Scholes model, yields the observed market price. Implied volatility is a key metric for traders and investors in the assessment of market expectations and the pricing of options.

$$
c(\sigma) = SN(d_+)-Ke^{-r\tau}N(d_-) \text{ where } d_{\pm} = \tfrac{1}{\sigma\sqrt{\tau}}[\log \tfrac{S}{K}+(r\pm\tfrac{\sigma^2}{2})\tau]
$$

$$\frac{\partial c}{\partial \sigma}$$ is denoted by 
$$\mathcal{V}$$ and it holds that 

$$
\mathcal{V} =  S\sqrt{\tau}N'(d_+) > 0.
$$

