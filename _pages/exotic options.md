### Exotic Options

Exotic Options' payoffs depend on path of the underlying asset are called path-dependent 

**Example:** 
In this note, we consider three types of exotic options on geometric Brownian motions assets:

- Barrier options (_e.g.,_ up \& out)
- Lookback options 
- Asian options

Pricing of exotic options for GBM assets depend on a the following principle:

**Reflection Principle** 

$$
\mathbb{P}\left(\tau_m \leq t, W(t) \leq \omega \right) = \mathbb{P}\left(W(t) \leq 2m-\omega \right) \quad \omega \leq m, m>0.
$$


Using reflection principle, we can compute the joint density of $M(t)$ and $W(t)$
\[
\bP\left(M(t)\geq m, W(t)\leq \omega \right) = \bP\left(W(t)\geq 2m-\omega \right), \quad w\leq m, m\geq 0.
\]
From $S(t)=rS(t)\dd t+ \sigma S(t)\dd \widetilde{W}(t)$, obtain the following
\[
S(t) = S(0)e^{\sigma \widehat{w}(t)}, \quad \alpha = \frac{1}{\sigma}\left(r-\frac{\sigma^2}{2}\right)
\]
Applying a change of measure argument, we could compute joint density of $\widehat{M}(t)$ and $\widehat{W}(t)$.
\[
\widehat{W}(t) = \alpha t+W(t), \quad \widehat{M}(t) = \max_{0\leq s\leq t} \widehat{W}(s).
\]
\paragraph{Payoffs} Payoff functions for barriers and lookbacks are computed as follows:
\begin{align*}
    V_{\text{barrier}}(T) &= \left(S(0)e^{\sigma \widehat{W}(t)}-K\right)\bm{1}_{\{\widehat{W}(T)\geq k, \widehat{M}(T)\leq b\}} \\ 
       V_{\text{lookback}}(T) &= S(0)\left(e^{\sigma \widehat{M}(T)}-e^{\sigma \widehat{W}(T)} \right)
\end{align*}
\paragraph{Boundary conditions} 
\begin{align*}
    \mathcal{R}_{\text{Barrier}} &= \{(t,x): 0\leq t<T, 0\leq x \leq B \} \\
    v^B(t,0) &=0  ,0\leq t \leq T \\ 
    v^B(t,B) &= 0 ,0\leq t <T \\ 
    v^B(T,x) &= (x-K)^+  ,0\leq x\leq B
\end{align*}
\begin{align*}
    \mathcal{R}_{\text{Lookback}} &= \{(t,x,y): 0\leq t<T, 0\leq x \leq y \} \\
    v^L(t,0,y) &=e^{-r(T-t)}y ,0\leq t \leq T, y \geq 0\\ 
    v^L_y(t,y,y) &= 0 ,0\leq t <T, y>0 \\ 
    v^L(T,x,y) &= y-x ,0\leq x\leq y
\end{align*}
\paragraph{Partial Differential Equations}
The call has not been knocked out by $t$ and $S(t)=x$
\begin{align*}
    v^B_t(t,x) + rxv^B_x(t,x)+\frac{1}{2}\sigma^2x^2v^B_{xx}(t,x) &= rv^B(t,x) &\forall (t,x) \in   \mathcal{R}_{\text{Barrier}}.
\end{align*}
\begin{align*}
    v^L_t(t,x,y) + rxv^L_x(t,x,y)+\frac{1}{2}\sigma^2x^2v^L_{xx}(t,x,y) &= rv^L(t,x,y) &\forall (t,x,y) \in   \mathcal{R}_{\text{Lookback}}.
\end{align*}
\paragraph{Delta-hedging for Barriers} $v(t,x)$ is discontinuous at the corner of $\mathcal{R}_{\text{Barrier}}$ at which delta (\textit{i.e.,} $v_x(t,S(t))$) and gamma (\textit{i.e.,} $v_{xx}(t,S(t))$) are large negative values. Normal delta-hedging becomes impractical as the large volume of trades renders significant the presumably negligible bid-ask spread.The common industry practice is to price and hedge the up-and-out call as if the barrier were at a level slightly higher than $B$.
\paragraph{dY(t)}  $\dd Y(t)$ is different from $\dd S(t)$ and $\dd t$. This follows from the fact that $Y(t)$ is monotonic and thus has zero quadratic variation. Moreover, $Y(t)$'s flat regions has Lebesgue measure 1 and hence $\dd Y(t) \neq \Theta(t)\dd t$ for any procecss $\Theta(t)$. The following holds
\begin{align*}
    \dd Y(t) \dd Y(t) &= 0 \\ 
    \dd Y(t) \dd S(t) &= 0 \\ 
    \dd e^{-rt}v(t,S(t),Y(t)) &= e^{-rt}[\cdots]\dd t+e^{-rt}\sigma S(t)v_x(t,S(t),Y(t))\dd \widetilde{W}(t)+e^{-rt}v_y(t,S(t),Y(t))\dd Y(t)\\
    \dd Y(t) &\neq 0 \text{ iff } S(t) = Y(t) \Rightarrow \text{ 2nd boundary condition}
\end{align*}
