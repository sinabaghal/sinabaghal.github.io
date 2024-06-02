---
title: 'Exercise 1.1'
date: 2024-06-02
permalink: /posts/shreve_1_1/
tags:
  - Probability Theory
---

<h2 id="exercise-1.1">Exercise 1.1</h2>
<p>Consider a probability space <span
class="math inline">(<em>Ω</em>, ℱ, ℙ)</span>. Show that</p>
<ul>
<li><p>If <span class="math inline"><em>A</em>, <em>B</em> ∈ ℱ</span>
and <span class="math inline"><em>A</em> ⊆ <em>B</em></span>, then <span
class="math inline">ℙ(<em>A</em>) ≤ ℙ(<em>B</em>)</span></p></li>
<li><p>If <span
class="math inline"><em>A</em> ⊆ <em>A</em><sub><em>n</em></sub></span>
and <span
class="math inline">lim ℙ(<em>A</em><sub><em>n</em></sub>) = 0</span>,
then <span class="math inline">ℙ(<em>A</em>) = 0</span>.</p></li>
</ul>
<h3 class="unnumbered" id="proof">Proof</h3>
<p>Consider the following set of countable many disjoint sets in <span
class="math inline">ℱ</span>: <span
class="math display"><em>A</em>, <em>B</em> ∖ <em>A</em>, ∅, ∅, ∅, ⋯.</span>
Therefore, <span
class="math inline">ℙ(<em>B</em>) = ℙ(<em>A</em>) + ℙ(<em>B</em> ∖ <em>A</em>) ≥ ℙ(<em>A</em>)</span>.
To see the second part, note that <span
class="math display">ℙ(<em>A</em>) ≤ ℙ(<em>A</em><sub><em>n</em></sub>)  ∀<em>n</em>.</span>
Taking <span class="math inline"><em>n</em> → +∞</span> yields that
<span
class="math inline">ℙ(<em>A</em>) ≤ lim ℙ(<em>A</em><sub><em>n</em></sub>) = 0</span>.
Thus, <span class="math inline">ℙ(<em>A</em>) = 0</span>.</p>
