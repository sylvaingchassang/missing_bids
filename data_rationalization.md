# Missing Bids: Data Rationalization
------------------------------------

## Requirements from competitive equilibrium
--------------------------------------------

For every history $h$ in data set $E$, there must exist
$z_h = (D_h, P_h^-, P_h^+)$ and $c_h\in [0,b_h]$ such that
$D_h \in [0,1]$, $P_h^- \in [0, 1- D_h]$ and $P_h^+ \in [0, D_h]$ such that

** IC **

$[(1-\rho^-) b - c_h](D_h+P_h^-) \leq (b_h - c_h) D_h$

$[(1+\rho^+) b - c_h](D_h-P_h^+) \leq (b_h - c_h) D_h$


** Costs **

$\frac{c_h}{b_h} \geq \frac{1}{1+m}$

Let $Z(\hat D, \hat P^+, \hat P^-)$ denote the set of such triplets $z$.
Note that I'm using realized event frequencies rather than true ones --
a priori the true frequencies would be more natural, but it's less tractable.

### Information constraints

For $X \in \{\hat D, \hat P^+, \hat P^-\}$,
$\left| \log\frac{X_h}{1 - X_h}  - \log\frac{X}{1 - X}\right| \leq k$


### Consistency with data

In addition, with high probability, aggregate demand at competitive auctions
must be consistent with observed data.

Let
$$
\hat z_h = (1_{\land b_{-i, h} > b_h}, 1_{\land b_{-i, h} \in
((1-\rho^+)b_h, b_h)}, 1_{\land b_{-i, h} \in (b_h, (1+\rho^+)b_h )})
$$
and $Y = \{(0,0,0), (1,0,0), (0,1,0), (1,0,1)\}$

For any mapping $p^C: Y \to [0,1]$ we define

\begin{align*}
(\hat D(p^C), \hat P^-(p^C), \hat P^+(p^C)) =
\frac{1}{|E|} \sum_{y \in Y} p^C(y) \times |E_y| \times y
\end{align*}

where $E_y = \{h \in E \;s.t.\; y_h = y\}$

Let $A_T(p^C) = [\hat D(p^C) \pm T] \times [\hat P^-(p^C) \pm T]
\times [\hat P^+(p^C)\pm T] \cap [0,1]^3$

Let $\tilde s_{comp}(p^C) = \frac{1}{|E|} \sum_{y \in Y} p^C(y) \times |E_y|$

The maximum share of competitive auctions consistent with data is

$$\max_{p^C} \{\tilde s_{comp}(p^C) \;|\; [\tilde s_{comp} \times
\textsf{convex } Z(p^C)]\cap A_T(p^C) \neq \emptyset\}$$


## Simplifying the computation of $\textsf{convex } Z(p^C)$
-----------------------------------------------------------

First observe that the IC constraint and the cost constraint can be
rewritten as

$$
\left[1-\rho^- - \rho^-\frac{D_h}{P_h^-}\right] \lor \frac{1}{1+m}
\leq 1+\rho^+ - \rho^+\frac{D_h}{P_h^+} \qquad\qquad(IC)
$$

The information constraints can be rewritten as
$$X_h \in [\underline B(X, k), \overline B(X,k)]$$
for $X \in \{D, P^-, P^+\}$ and

$$
\underline B(X,k) =
\frac{\frac{X}{1-X}\exp(-k)}{1 + \frac{X}{1-X}\exp(-k)}
\quad \text{and} \quad
\overline B(X,k) =  \frac{\frac{X}{1-X}\exp(k)}{1 + \frac{X}{1-X}\exp(k)}
$$

Let $B(p^C) \equiv
\times_{X \in \{D, P^-, P^+\}}[\underline B(X,k), \overline B(X,k)]$.


Note that there exists $P_h^-$ such that $(IC)$ holds iff
$$1+\rho^+ - \rho^+\frac{D_h}{P_h^+} > \frac{1}{1+m}
\iff \frac{D_h}{P_h^+} < \left(1+\rho^+ - \frac{1}{1+m}\right)\Big/\rho^+$$

Hence, the set of acceptable $P_h^+$ given $D_h$ is

$$P_h^+ \in \left[\frac{\rho^+}{1+ \rho^+ -1/(1+m)} D_h , D_h\right]$$

Given $D_h$ and $P_h^+$ such that a solution $P_h^-$ to $(IC)$ exists, the
range of values $P_h^-$ solving $(IC)$ is the set of values
$P_h^- \in [0, 1- D_h]$ such that

\begin{align}
&1-\rho^- - \rho^-\frac{D_h}{P_h^-} \leq 1+\rho^+ - \rho^+\frac{D_h}{P_h^+}\\
\iff & -\rho^- -\rho^+ + \rho^+\frac{D_h}{P_h^+} \leq \rho^- \frac{D_h}{P_h^-} \\
\iff P_h^- \leq \overline P_h^-.
\end{align}

where

$$
\overline P_h^- = \min\left\{1- D_h,
\frac{\rho^- D_h}{\left[-\rho^- - \rho^+ +  \rho^+\frac{D_h}{P_h^+}\right]^+}
\right\}.
$$


** The Problem **
-----------------

We define

$Z(p^C)$ the set of points satisfying (IC), and cost constraint.

We are interested in solving

$\max_{p^C} \sum_{y \in Y} p^C$ such that

$$\textsf{convex}(Z(p^C))\cap B(p^C) \cap A(p^C) \neq \emptyset$$


** claim 1**
------------

The set $Z(p^C)$ admits as extreme points
   - $D_h = 0$, $P_h^+ = 0$, $P_h^- = 0$
   - $D_h = 0$, $P_h^+ = 0$, $P_h^- = 1$
   - $D_h = 1$, $P_h^+ = 1$, $P_h^- = 0$
   - $D_h = 1$, $P_h^+ = \frac{\rho^+}{1+ \rho^+ -1/(1+m)}$, $P_h^- = 0$

It's convex hull is therefore the corresponding pyramid

** claim 2**
------------

Sets $A$ and $\textsf{convex}(Z)$ are both convex. If they do not have an
intersection, they must be separated by a hyperplane.

If $A$ and $\textsf{convex}(Z)$ are separated, then they are separated by one
of the faces of the pyramid that is not the plan $(D_h, P_h^+, P_h^-=0)$.