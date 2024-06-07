### Exotic Options

Exotic Options' payoffs depend on the path of the underlying asset. In other words, their payoffs are path-dependent.

**Example:** 
In this note, we consider three types of exotic options on geometric Brownian motions assets:

- Barrier options (_e.g.,_ up \& out)
- Lookback options 
- Asian options

## Definition

A _Barrier options_ are a type of financial derivative whose payoff depends on whether the underlying asset's price reaches a specific predetermined level, called the barrier, within a certain timeframe. If the asset's price hits this barrier, the option is either activated (knock-in) or deactivated (knock-out). These options can be either calls or puts and usually have lower premiums than standard options due to their conditional nature.

A _Lookback options_ are a type of exotic financial derivative that allows the holder to "look back" over the life of the option and choose the optimal underlying asset's price to determine the payoff. There are two main types of lookback options:

- Fixed Lookback Options: The strike price is determined at the inception of the option, but the payoff is based on the maximum or minimum underlying asset price during the option's life.
- Floating Lookback Options: The strike price is determined based on the maximum or minimum underlying asset price during the option's life, providing the holder with the most favorable price movement.

As an example, a lookback call option expiring at time $T$ will have the following payoff formula:

- Fixed-Strike: $\max(S_{\max}-K,0)$
- Floating-Strike: $\max(S(T)-S_{\min},0)$

These options are more expensive than standard options because they offer a significant advantage in eliminating the uncertainty of timing market highs and lows.

An _Asian options_ is a type of financial derivative where the payoff is determined by the average price of the underlying asset over a specific period, rather than its price at a single point in time. This averaging feature can reduce the option's volatility and, typically, its premium compared to standard options. Asian options can be either calls or puts.

In this note, we provide some insights into how these exotic options are priced. Pricing barrier and lookback options requires an understanding of the joint density between the maximum of a Brownian motion $W(t)$ over an interval $[0, T]$ and $W(T)$. This distribution is derived using _Reflection Principle_ which is defined next.

## Reflection Principle

Consider a Brownian motion $W(t)$ and define the stopping time $\tau_m$ as below:

$$
\tau_m = \min \{ t : W(t) = m \}
$$

Refelection principle asserts that 

$$
\mathbb{P}\left(\tau_m \leq t, W(t) \leq \omega \right) = \mathbb{P}\left(W(t) \leq 2m-\omega \right) \quad \omega \leq m, m>0.
$$

<p align="center">
<img src="http://sinabaghal.github.io/images/reflection_principle.png" width="40%" height="40%">
</p>

Using reflection principle, we can compute the joint density of $M(t)$ and $W(t)$

$$
\mathbb{P}\left(M(t)\geq m, W(t)\leq \omega \right) = \mathbb{P}\left(W(t)\geq 2m-\omega \right), \quad w\leq m, m\geq 0.
$$

From $S(t)=rS(t)\mathrm{d} t+ \sigma S(t)\mathrm{d} \widetilde{W}(t)$, obtain the following

$$
S(t) = S(0)e^{\sigma \widehat{w}(t)}, \quad \alpha = \frac{1}{\sigma}\left(r-\frac{\sigma^2}{2}\right)
$$

Applying a change of measure argument, we could compute joint density of $\widehat{M}(t)$ and $\widehat{W}(t)$.

$$
\widehat{W}(t) = \alpha t+W(t), \quad \widehat{M}(t) = \max_{0\leq s\leq t} \widehat{W}(s).
$$

Payoff functions for barriers and lookbacks are computed as follows:

```math
\begin{align}
    V_b(T) &= S(0)e^{\sigma \widehat{W}(t)}-K){1}_{\{\widehat{W}(T)\geq k, \widehat{M}(T)\leq b\}} \\ 
       V_l(T) &= S(0)(e^{\sigma \widehat{M}(T)}-e^{\sigma \widehat{W}(T)} )
\end{align}
```

## Boundary conditions

Boundary conditions play a key role in pricing exotic options! Often, we obtain the same PDE as vanilla call or put options when analyzing these options, but intrinsically different boundary conditions play an instrumental role in their analytics!

```math
\begin{align*}
R_b &= \{(t,x): 0\leq t<T, 0\leq x \leq B \} \\ v^B(t,0) &=0  ,0\leq t \leq T \\ v^B(t,B) &= 0 ,0\leq t <T \\v^B(T,x) &= (x-K)^+  ,0\leq x\leq B
\end{align*}
```


```math
\begin{align*}
    \mathcal{R}_l &= \{(t,x,y): 0\leq t<T, 0\leq x \leq y \}\\
    v^L(t,0,y) &=e^{-r(T-t)}y ,0\leq t \leq T, y \geq 0\\ 
    v^L_y(t,y,y) &= 0 ,0\leq t <T, y>0\\ 
    v^L(T,x,y) &= y-x ,0\leq x\leq y
\end{align*}
```

## Partial Differential Equations

The call has not been knocked out by $t$ and $S(t)=x$
\begin{align*}
    v^B_t(t,x) + rxv^B_x(t,x)+\frac{1}{2}\sigma^2x^2v^B_{xx}(t,x) &= rv^B(t,x) &\forall (t,x) \in   \mathcal{R}_{\text{Barrier}}.
\end{align*}
\begin{align*}
    v^L_t(t,x,y) + rxv^L_x(t,x,y)+\frac{1}{2}\sigma^2x^2v^L_{xx}(t,x,y) &= rv^L(t,x,y) &\forall (t,x,y) \in   \mathcal{R}_{\text{Lookback}}.
\end{align*}

**Delta-hedging for Barriers** 

$v(t,x)$ is discontinuous at the corner of $R_b$ at which delta 

$$\Delta = v\_x (t,S(t)), \gamma = v_{xx}(t,S(t))$$

are large negative values. Normal delta-hedging becomes impractical as the large volume of trades renders significant the presumably negligible bid-ask spread.The common industry practice is to price and hedge the up-and-out call as if the barrier were at a level slightly higher than $B$.

$dY(t)$

$\mathrm{d} Y(t)$ is different from $\mathrm{d} S(t)$ and $\mathrm{d} t$. This follows from the fact that $Y(t)$ is monotonic and thus has zero quadratic variation. Moreover, $Y(t)$'s flat regions has Lebesgue measure 1 and hence $\mathrm{d} Y(t) \neq \Theta(t)\mathrm{d} t$ for any procecss $\Theta(t)$. The following holds
$$
\begin{align*}
    \mathrm{d} Y(t) \mathrm{d} Y(t) &= 0 \\ 
    \mathrm{d} Y(t) \mathrm{d} S(t) &= 0 \\ 
    \mathrm{d} e^{-rt}v(t,S(t),Y(t)) &= e^{-rt}[\cdots]\mathrm{d} t+e^{-rt}\sigma S(t)v_x(t,S(t),Y(t))\mathrm{d} \widetilde{W}(t)+e^{-rt}v_y(t,S(t),Y(t))\mathrm{d} Y(t)\\
    \mathrm{d} Y(t) &\neq 0 \text{ iff } S(t) = Y(t) \Rightarrow \text{ 2nd boundary condition}
\end{align*}
$$
