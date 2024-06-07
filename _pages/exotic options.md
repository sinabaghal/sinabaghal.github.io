### Exotic Options

Exotic Options' payoffs depend on the path of the underlying asset. In other words, their payoffs are path-dependent.

**Example:** 
In this note, we consider three types of exotic options on geometric Brownian motions assets:

- Barrier options (_e.g.,_ up \& out)
- Lookback options 
- Asian options

Pricing of exotic options for GBM assets depend on a the following principle:

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
    V_b(T) &= (S(0)e^{\sigma \widehat{W}(t)}-K){1}_{\{\widehat{W}(T)\geq k, \widehat{M}(T)\leq b\}} \\ 
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
