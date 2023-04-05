$$
\newcommand{\v}[1]{\boldsymbol{#1}} 
\newcommand{\m}[1]{\mathbf{#1}} 
\newcommand{\d}{\mathrm{d}} 
\newcommand{\cl}[1]{\mathcal{#1}} 
\newcommand{\tr}{\mathrm{Tr}} 
$$


# Scalable Approaches to Self-Supervised Learning using Spectral Analysis

**Reading Group -- Wednesday April 5th 2023**

*By Ross Viljoen and Vincent Dutordoir*


### Abstract
Learning the principal eigenfunctions of an operator is a fundamental problem in various machine learning tasks, from representation learning to Gaussian processes. However, traditional non-parametric solutions suffer from scalability issues --- rendering them impractical on large datasets.

This reading group will discuss parametric approaches to approximating eigendecompositions using neural networks. In particular, Spectral Inference Networks (SpIN) offer a scalable method for approximating eigenfunctions of symmetric operators on high-dimensional function spaces using bi-level optimization methods and gradient masking (Pfau et al., 2019).

A recent improvement on SpIN, called NeuralEF, focuses on approximating eigenfunction expansions of kernels (Deng et al., 2022a). The method is applied to modern neural-network based kernels (GP-NN and NTK) as well as scaling up the linearised Laplace approximation for deep networks (Deng et al., 2022a). Finally, self-supervised learning can be expressed in terms of approximating a contrastive kernel, which allows NeuralEF to learn structured representations (Deng et al., 2022b).

#### References

David Pfau, Stig Petersen, Ashish Agarwal, David G. T. Barrett,and Kimberly L. Stachenfeld. "Spectral inference networks: Unifying deep and spectral learning." ICLR (2019).

Zhijie Deng, Jiaxin Shi, and Jun Zhu. "NeuralEF: Deconstructing kernels by deep neural networks." ICML  (2022a).

Zhijie Deng, Jiaxin Shi, Hao Zhang, Peng Cui, Cewu Lu, Jun Zhu. "Neural Eigenfunctions Are Structured Representation Learners." arXiv preprint arXiv:2210.12637 (2022b).


---

### Introduction

Spectral methods are ubiquitous:
- mathematics: spectral theorem (diagonalisation of compact, self-adjoint operators)
- physics: Solving PDEs (e.g., heat, wave, Schrödinger equations) in frequency domain (i.e. using spectral metin frequency domain (i.e. using spectral methods).
- Machine learning:
    - Kernels on manifold use the eigenbasis w.r.t. to the Laplace-Beltrami operator $\Delta_{\cl{M}}$.
    - Spectral Mixture Kernels
    - Fourier Neural Operators


| Machine Learning | Physics/Engineering
| -------- | -------- |
|  ![](https://i.imgur.com/2flz0gC.gif) | ![](https://i.imgur.com/UBVjvVJ.png)


### Topic of today's reading group

![](https://i.imgur.com/VxmM3Vf.png)

:::success
**Find non-linear embedding of data on low-dimensional manifold.**
:::

A well known algorithm is Principle Component Analysis.
1. Construct Gram for all pair of points
$$
k(\m{X}, \m{X}) \in \mathbb{R}^{N \times N}
$$
2. Each eigenvector $\v{u}$ contains a one-dimensional embedding for each point $x_n$.


Scales cubically and quadratically with $N$ for compute and memory. What to do with new data?

### Overview of reading group

This reading group will cover three papers in chronological order that build on each other.

1. Pfau et al. (2019). "Spectral inference networks: Unifying deep and spectral learning."
    - Introduces the problem and formulates the problem of finding embeddings (i.e. eigenfunctions) as an optimization problem.
    - Focus is not necessarily on kernel operators.
2. Deng et al. (2022). "NeuralEF: Deconstructing kernels by deep neural networks."
    - Improve upon SpIN algorithm (more computationally efficient and simpler).
    - Apply the method to popular DNN-inspired kernels.
3. Deng  et al. (2022). Neural Eigenfunctions Are Structured Representation Learners
    - A final final improvement on the core algorithm with applications in self-supervised learning.


### Spectral decomposition as Optimization

Eigenvectors of a matrix are defined as $\m{A} \v{u} = \lambda \v{u}$ for some scalar $\lambda$, called the eigenvalue.

Let $\m{A}$ be symmetric then, the eigenvector corresponding to the largest eigenvalue is the solution of

$$
\max_\v{u} \v{u}^\top \m{A} \v{u}\quad\text{s.t.}\quad \v{u}^\top \v{u} = 1
$$

:::info
**Proof (sketch)**

- $\m{A}$ is symmetric and real, therfore $\m{A} = \m{V} \m{\Lambda} \m{V}^\top$, where $\v{V}$ is a orthonormal basis: $\m{V}^\top \m{V} = \m{I}$ and $\m{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_n)$ with $\lambda_1 > \lambda_2 > \ldots > \lambda_n$.
-  Let $\v{u} = \m{V}\v{\alpha} = \sum \alpha_i \v{v}_i$

Rewriting the objective:
\begin{align}
\v{u}^\top \m{A} \v{u} &= \v{\alpha}^\top \m{V}^\top \m{V} \m{\Lambda} \m{V}^\top \m{V} \v{\alpha} \\
&= \tr(\v{\alpha}\v{\alpha}^\top \m{\Lambda})\\
&= \sum\nolimits_i \alpha_i^2 \lambda_i
\end{align}

Now adding a Lagrange multiplier $\ell$ for the constaint (which can be rewritten as $\| \v{\alpha} \|^2_2 = 1$) gives
$$
\sum\nolimits_i \alpha_i^2 \lambda_i + \ell (\| \v{\alpha} \|^2_2 - 1)
$$
which can be solved by setting the derivative w.r.t. $\v{\alpha}$ and $\ell$ to zero. This gives
$$
\alpha_1 = 1 \quad\text{and}\quad\alpha_{\neq1}=0.
$$
:::

The constrained optimisation objective from above, is equivalent (up to a scaling factor) to
\begin{equation}
\max_\v{u} \frac{\v{u}^\top \m{A} \v{u}}{\v{u}^\top \v{u}}
\end{equation}
because we normalize the vector to lie on the unit hypersphere: $\frac{\v{u}}{\sqrt{\v{u}^\top\v{u}}} \in \cl{S}^d$. This expression is known as the **Rayleigh quotient** and equals the eigenvalue corresponding to the eigenvector at its maximum.

To compute the top N eigenvectors $\m{U} = (\v{u}_1, \ldots , \v{u}_N )$, we can solve a sequence of maximization problems greedily:
$$
\v{u}_i = \text{arg}\max_\v{u} \frac{\v{u}^\top \m{A} \v{u}}{\v{u}^\top \v{u}}\quad\text{s.t.}\quad \v{u}_1^\top\v{u}=0,\ldots,\v{u}_{i-1}^\top\v{u}=0.
$$

If we drop the orthogonality constraint we can formulate this as a single unconstrained objective
$$
\text{arg}\max_{\m{U}} \tr\big[ (\m{U}^\top\m{U})^{-1} \m{U}^\top \m{A} \m{U} \big]
$$
Following a similar derivation as above, the maximum is given by $\sum_{i=1}^N \lambda_i$ (the largest $N$ eigenvalues). However, without the orthogonality constaint $\v{u}_i^\top\v{u}_j = 0$ for $i\neq j$, $\m{U}$ will simply span the same subspace as the true eigenfunctions.

:::info
**Remark.** The objective is invariant to right multiplications (i.e. linear combination of the basis).


*Proof.* Set $\tilde{\m{U}} = \m{U} \m{W}$ and show that the objective has the same optimum for $\tilde{\m{U}}$ as for $\m{U}$.

:::


The above derivation can be used to optimize for the eigenvectors in (kernel) PCA where $\m{A} = \m{X} \m{X}^\top$ or $\m{A} = k(\m{X}, \m{X})$ for $\m{X} \in \mathbb{R}^{N\times D}$ a feature matrix.

An embedding for a new datapoint can be found by interpolating the eigenfunctions:
$$
\v{u} = \sum_{n=1}^N k(x_n, x) \v{u}_n
$$
which scales $\cl{O}(N)$.

:::success
Spectral Inference Networks (SpIN) overcomes these limitations by using a **parametric** form for the embeddings.
:::

### From eigenvectors to eigenfunctions


Suppose that instead of a matrix $\m{A}$ we have a symmetric (not necessarily positive definite) kernel $k(x, x)$ where $x$ and $x'$ are in some measurable space $\Omega$, which can be either continuous or discrete.

Define inner product on $\Omega$ w.r.t. to input density $p(x)$ as
$$
\langle f, g \rangle = \int_\Omega f(x) g(x) p(x) \d x = \mathbb{E}_{x\sim p} [f(x)g(x)]
$$
and a kernel operator $\cl{K}$ as
$$
\cl{K} f = \langle k(x,\cdot), f\rangle = \int k(x, x') f(x') p(x') \d x' = \mathbb{E}_{x'\sim p}[k(x, x') f(x')]
$$

we are interested in finding the top $N$ eigenfunctions $u:\Omega \rightarrow \mathbb{R}$ of the operator $\cl{K}$, for which hold 
$$
(\cl{K}u)(x) = \int k(x',x)u(x')p(x')\d x' = \lambda\,u(x)\quad\text{and}\quad \langle u_i, u_j\rangle = \delta_{ij}.
$$


#### In vector space
In vector space we could cast the problem of spectral decomposition to an optimization problem using the following objective:

$$
\max_{\m{U}} \tr\big[ (\m{U}^\top\m{U})^{-1} \m{U}^\top \m{A} \m{U} \big]
\iff \max_{\ \ \ \ \ \m{U} \\ \m{U}^\top \m{U} = \m{I}} \tr\big[  \m{U}^\top \m{A} \m{U} \big]
$$

#### In function space
The analogue of this in function space looks like this:

Let $\v{u}:\cl{X}\rightarrow\mathbb{R}^N$ represent the N eigenfunctions (or a linear combination thereof, which means that they share the same span).


\begin{align}
\max_{\v{u}} &\  \tr\Big[\mathbb{E}_{x}\big[\v{u}(x) \v{u}^\top(x)\big]^{-1}  \mathbb{E}_{x,x'}\big[k(x,x')\v{u}(x)\v{u}(x')\big] \Big]
\\[2mm]& \iff
\max_{\v{u}} \tr\Big[\mathbb{E}_{x,x'}\big[k(x,x')\v{u}(x)\v{u}(x')\big] \Big]
\text{ s.t. } \mathbb{E}_{x}[\v{u}(x) \v{u}^\top(x)]  = \m{I}
\end{align}

To simplify notation

$$
\m{\Sigma} = \mathbb{E}_{x}\big[\v{u}(x) \v{u}^\top(x)\big]\quad\text{and}\quad\m{\Pi} = \mathbb{E}_{x,x'}\big[k(x,x')\v{u}(x)\v{u}(x')\big]
$$
which gives, parameterizing the eigenfunctions using a neural net with parameters $\theta$, the following objective:
$$
\begin{equation}
\cl{L}(\theta) = \max_{\v{u}_\theta} \tr\left[ \m{\Sigma}^{-1} \m{\Pi}\right]
\end{equation}
$$

:::danger
**Problems**
- The objective is invariant to linear transformations of the features $\v{u}(x)$, optimizing it will only give a function that spans the top eigenfunctions of $\cl{K}$. There is also no ordering (i.e. first eigenfunction corresponds to largest eigenvalue).
- Minibatch estimations of $\cl{L}$ do **not** lead to unbiased estimates of the gradients. We can not use traditional SGD.

:::

### Ordering through gradient masking

The objective is invariant to linear transformation of the output. The solution only spans the top eigenfunctions...

We can use the invariance of trace to cyclic permutation to rewrite the objective. 
Let $\m{\Sigma} = \m{L}\m{L}^\top$, then the objective can be rewritten
$$
\tr \left(\m{\Sigma}^{-1} \m{\Pi}\right) = \tr\left( \m{L}^{-1} \m{\Pi} \m{L}^{-\top}\right) = \tr\,\m{\Lambda} = \sum \lambda_i
$$

where we used a shorthand $\m{\Lambda} \triangleq \m{L}^{-1} \m{\Pi} \m{L}^{-\top}$. This matrix has the convenient property that the upper left $n \times n$ block only depends on the first $n$ eigenfunctions.


By zero-ing out the gradient from higher to lower eigenvalues we can impose an ordering. Summarized in the following claim (in the appendix):

![](https://i.imgur.com/MPVJ5BA.png)
proof by induction.


| Forward  | Backward |
| -------- | -------- |
| ![](https://i.imgur.com/LoMExMr.png)     | ![](https://i.imgur.com/uOq28OS.png)     |

The 'masked' gradient can be written down in closed-form:
![](https://i.imgur.com/yAB5acw.png)
where $\m{\Sigma} = \m{L}^\top \m{L}$.


### Bi-level optimization


In SGD we use stochastic gradient *estimates* $\tilde{\v{g}}$ which typically are unbiased:
$$
\mathbb{E}[\tilde{\v{g}}] = \nabla_\theta \cl{L}
$$

The objective expression is a nonlinear function of multiple expectations, so naively replacing $\m{\Sigma}, \m{\Pi}$ and their gradients with empirical estimates will be biased.

Reframe objective as bi-level optimization, in which $\m{\Sigma}$ and $\m{\Pi}$ are estimated as well as $\theta$.

Bilevel stochastic optimization is the problem of simultaneously solving two coupled minimization.

Traditional optimization 
$$
\max_x f(x)\quad\text{s.t.}\quad g(x) \le 0.
$$

Formally, a bilevel optimization problem reads:
$$
 \min_{x\in\cl{X},y}\ F(x, y)\quad\text{s.t.}\quad
\ G(x, y) \le 0,\text{ and }
 y \in S(x)
$$
where $S(x)$ is the set of optimal solutions of the x-parameterized problem
$$
S(x) = \text{arg}\min_{y\in\cl{Y}} f(x, y)
$$

Bi-level optimization the objective depends on the 'decision' of another actor whose decision depends on our action...

:::success
**Example: Toll Setting**
 Consider a network of highways that is operated by the government. The government wants to maximize its revenues by choosing the optimal toll setting for the highways. However, the government can maximize its revenues only by taking the highway users' problem into account. For any given tax structure the highway users solve their own optimization problem, where they minimize their traveling costs by deciding between utilizing the highways or an alternative route
:::

Bilevel stochastic problems are common in machine learning and include actor-critic methods, generative adversarial networks and imitation learning. SpIN relies on a classic result from 1997 "Vivek S Borkar. Stochastic approximation with two time scales". It boils down to having a second objective which is solved using a different (slower) learning rate
$$
\min_{\m{\Sigma}, \nabla_\theta\m{\Sigma}} \|\m{\Sigma} - \bar{\m{\Sigma}}_t \|^2 + \| \nabla_\theta\m{\Sigma} -\bar{\nabla_\theta{\m{\Sigma}_t}} \|^2
$$
where $\bar{\m{\Sigma}}_t$ and $\bar{\nabla_\theta{\m{\Sigma}}}_t$ are moving averages.


### SpIN Algorithm

1. Minimizes the objective end-to-end by stochastic optimization
$$
\tr\Big[\mathbb{E}_{x}\big[\v{u}(x) \v{u}^\top(x)\big]^{-1}  \mathbb{E}_{x,x'}\big[k(x,x')\v{u}(x)\v{u}(x')\big] \Big]
$$
2. Performs the optimization over a parametric function class such as deep neural networks.
3. Uses the modified gradient (masking) to impose an ordering on the learned features
4. Uses bilevel optimization to overcome the bias introduced by finite batch sizes

Problems:
- Need to cholesky decompose $\m{\Sigma}$ in every iteration
- Need to track Jacobian $\nabla_\theta \m{\Sigma}$


-------

![](https://i.imgur.com/TbDHRyh.png)


## Kernel methods

Kernel methods project data into a high-dimensional feature space $\cl{H}$ to enable linear manipulation of nonlinear data.  In other words, let $\phi: \cl{X} \rightarrow \cl{H}$ be the feature map, then $k:\cl{X} \times \cl{X} \rightarrow \mathbb{R}$ linearly depends on the feature representations of the inputs
$$
k(x, x') = \langle \phi(x), \phi(x') \rangle
$$
Often we don't know $\cl{H}$ (it is typically infinite dimensional). The subtlety is that we can leverage the "kernel trick" to bypass the need of specifying $\cl{H}$ explicitly.

There is a rich family of kernels. 

Classic kernels: Squared Exponential, Matérn Family, Linear, Polynomial, etc. They may easily fail when processing real-world data like images and texts due to inadequate expressive power and the curse of dimensionality.

Modern kernel have been developed on the inductive biases of NN architectures, such as the NTK and NN-GP (discussed briefly by Ross and the topic of next week's reading group).

## Kernel approximations

An idea to scale up kernel methods is to approximate the kernel with the inner product of some explicit vector representations of the data
$$
k(x,x') \approx \v{\nu}(x)^\top \v{\nu}(x'),
$$
where $\v{\nu}: \cl{X} \rightarrow \mathbb{R}^R$. There are two broad approaches...

### Bochner's theorem

Bochner's theorem: a stationary kernels is psd if it is the Fourier transform of a positive measure $S(\omega)$ (power spectrum):
\begin{align}
k(x-y) &= \int S(\omega) \exp(i \omega r) \d \omega \\
&\approx \frac{1}{R} \sum  \exp(i \omega_r (x-y))\quad \omega_r \sim S(\omega)\\
&= 
\begin{bmatrix}
\frac{1}{\sqrt{R}} \exp(i \omega_1 x) \\
\frac{1}{\sqrt{R}} \exp(i \omega_2 x) \\
\vdots \\
\frac{1}{\sqrt{R}} \exp(i \omega_R x) \\
\end{bmatrix}^\top
\begin{bmatrix}
\frac{1}{\sqrt{R}} \exp(i \omega_1 y) \\
\frac{1}{\sqrt{R}} \exp(i \omega_2 y) \\
\vdots \\
\frac{1}{\sqrt{R}} \exp(i \omega_R y) \\
\end{bmatrix}
\end{align}

- Can only be used for stationary kernel
- Scales exponentially bad with dimension

### Mercer decompsition

A kernel $k$ has a kernel operator $\cl{K}$ and we are interested in finding it's eigenfunctions.

Eigenfunctions $u:\Omega \rightarrow \mathbb{R}$ of the operator $\cl{K}$ obey 
$$
(\cl{K}u)(x) = \int k(x',x)u(x')p(x')\d x' = \lambda\,u(x)\quad\text{and}\quad \langle u_i, u_j\rangle = \delta_{ij}.
$$


Using the eigenfunctions corresponding to non-zero eigenvalues $k$ has the representation
$$
k(x,x') = \sum\nolimits_i \lambda_i u_i(x) u_i(x').
$$

#### Nyström approximation

We can use the data $\m{X} = \{x_n\}_{n=1}^N$ to obtain an MCMC estimate of the top R eigenfunctions by taking the eigenvectors of $k(\m{X},\m{X})$.
$$
\{\hat{\lambda}_i,\ (\hat{u}_i(x_1), \ldots, \hat{u}_i(x_N))\}_{i=1}^R
$$

Nystrom methods approximate the eigenfunctions for out-of-sample points $x$ as:
$$
\hat{u}_i(x) = \frac{1}{N} \sum\nolimits_n k(x,x_n) \hat{u}_i(x_n)
$$

which approximates the kernel
$$
k(x,x') \approx \sum_{i=1}^R \hat{\lambda}_i \hat{u}_i(x) \hat{u}_i(x')
$$

Unlike RFFs, Nyström method can be applied to approximate any positive-definite kernel. Yet, it is non-trivial to scale it up. 
- the matrix eigendecomposition is costly for even medium sized training data.
- evaluating the eigenfuction at a new location entails evaluating $k$ for N times, which is unaffordable when coping with the modern kernels specified by deep architectures.


## NeuralEF

Back to the SpIN objective:

\begin{align}
\max_{\v{u}} &\  \tr\Big[\mathbb{E}_{x}\big[\v{u}(x) \v{u}^\top(x)\big]^{-1}  \mathbb{E}_{x,x'}\big[k(x,x')\v{u}(x)\v{u}(x')\big] \Big]
\\[2mm]
\end{align}

Can we instead enforce an ordering directly in the optimisation objective such that the bi-level procedure becomes unnecessary?

Characterise the spectral decomposition as an asymmetric maximisation problem:

\begin{align}
\max_{\hat{u}_j} \ \Pi_{jj} \quad\text{s.t.}\quad \Sigma_{jj} = 1,\ \Pi_{ij} = 0, \ \forall i < j \\
\end{align}

where:

\begin{align}
\Sigma_{jj} = \mathbb{E}_{x}\big[\hat{u}_j(x) \hat{u}_j(x)\big]\quad\text{and}\quad\Pi_{ij} = \mathbb{E}_{x,x'}\big[k(x,x')\hat{u}_i(x)\hat{u}_j(x')\big]
\end{align}

The claim is that the pairs $(\Pi_{jj}, \hat{u}_j)$ will converge to the true eigenpairs $(\lambda_j, u_j)$, ordered such that $\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_k$.

:::info
**Proof**

For $j=1$, there is no orthogonality constraint, so the problem is the eigenfunction version of finding the largest eigenvector by maximisation:

\begin{align}
\Pi_{11} &= \mathbb{E}_{x,x'}\big[k(x,x')\hat{u}_1(x)\hat{u}_1(x')\big] \\
&= \mathbb{E}_{x,x'}\big[\left(\sum_{j\geq1} \lambda_j u_j(x)u_j(x') \right)\hat{u}_1(x)\hat{u}_1(x')\big] \quad (\text{Mercer's th}^\mathrm{m})\\
&= \sum_{j\geq1} \lambda_j \mathbb{E}_{x,x'}\big[u_j(x)u_j(x')\hat{u}_1(x)\hat{u}_1(x')\big] \\ 
&= \sum_{j\geq1} \lambda_j \mathbb{E}_{x}\big[u_j(x)\hat{u}_1(x)\big]\mathbb{E}_{x'}\big[u_j(x')\hat{u}_1(x') \big] \\ 
&= \sum_{j\geq1} \lambda_j \langle u_j, \hat{u}_1 \rangle ^2
\end{align}

but, since $\{u_j\}$ form an orthonormal basis, we can write $\hat{u}_1 = \sum_{l\geq1}w_lu_l$, so

\begin{align}
\Pi_{11} &= \sum_{j\geq1} \lambda_j \langle u_j, \sum_{l\geq1}w_lu_l \rangle ^2 \\
&= \sum_{j\geq1} \lambda_j w_j^2
\end{align}

Recalling the constraint:
\begin{align}
\Sigma_{11} = \mathbb{E}_{x}\big[\hat{u}_1(x) \hat{u}_1(x)\big] = \langle \hat{u}_1, \hat{u}_1 \rangle = \sum_{j\geq1} w_j^2 = 1,
\end{align}

the objective must be maximised by taking $w_1 = 1$ and $w_j = 0, j>1$. Therefore, the optimum is found when $\hat{u}_1 = u_1$.

For $j=2$, we now have the additional orthogonality constraint:
\begin{align}
\Pi_{12} &= \mathbb{E}_{x,x'}\big[k(x,x')\hat{u}_1(x)\hat{u}_2(x')\big] \\
&= \mathbb{E}_{x,x'}\big[k(x,x')u_1(x)\hat{u}_2(x')\big] \\
&= \sum_{j\geq1} \lambda_j \langle u_j, u_1 \rangle \langle u_j, \hat{u}_2 \rangle \\
&= \lambda_1 \langle u_1, \hat{u}_2 \rangle \\
&= 0
\end{align}

So, for $j=2$ we end up solving the same maximisation problem as for $j=1$, but restricted to the orthogonal subspace of $u_1$. Repeating the same logic for $j>2$ completes the proof.
:::

In practice, turn the orthogonality constraints into penalties:

\begin{align}
\max_{u_j} \ \Pi_{jj} - \sum_{i < j} \frac{\Pi_{ij}^2}{\Pi_{ii}} \quad\text{s.t.}\quad \Sigma_{jj} = 1,\   \forall i < j \\
\end{align}

Where the scaling factor of $\Pi_{ii}$ is introduced into the denominator so that the two terms are on the same scale.

#### Minibatching

At each optimisation step, $\Pi_{ij}$ is estimated by Monte Carlo integration with a minibatch of samples $\m{X} = \{x_b\}_{b=1}^{B}$:

\begin{align}
\tilde{\Pi}_{ij} &= \frac{1}{B^2}\sum_{b=1}^{B}\sum_{b'=1}^{B}\left[ k(x_b, x_{b'})\hat{u}_i(x_b)\hat{u}_j(x_{b'}) \right] \\
&= \frac{1}{B^2} {\v{u}_i^{\m{X}}}^\top\  \m{K}_{\m{X}, \m{X}}\  \v{u}_j^{\m{X}}
\end{align}

where $\v{u}_j^{\m{X}} = \left[ u_j(x_1), \dots, u_j(x_B)\right]^\top \in \mathbb{R}^B$ is the concatenated output of $u_j$ on the minibatch.

Tracking $\tilde{\Pi}_{jj}$ with an exponential moving average (EMA) gives an estimate of $\lambda_j$.

#### Batchnorm

Of course, there is still another constraint to satisfy, normalisation: $\Sigma_{jj} = 1$. In the minibatch setting, this is estimated as $\hat{\Sigma}_{jj} = \frac{1}{B}  {\v{u}_j^{\m{X}}}^\top  \v{u}_j^{\m{X}}$, so we want to have 
$$\| {\v{u}_j^{\m{X}}} \|^2_2 \approx B$$
This is enforced by adding an L2 batch normalisation layer to the end of each neural net:

$${\v{u}_j^{\m{X}}} = \frac{{\v{h}_j^{\m{X}}}}{\| {\v{h}_j^{\m{X}}} \|_2} \cdot B$$

where ${\v{h}_j^{\m{X}}}$ is the output of the penultimate layer for the minibatch $\m{X}$.

#### The Training Loss and Gradients

\begin{align}
\min_{\theta_1,\dots, \theta_k} \ell &= - \frac{1}{B^2} \sum_{j=1}^k \left( {\v{u}_j^{\m{X}}}^\top\  \m{K}_{\m{X}, \m{X}}\  \v{u}_j^{\m{X}} - \sum_{i=1}^{j-1} \frac{(\mathrm{sg}({\v{u}_i^{\m{X}}}^\top)\  \m{K}_{\m{X}, \m{X}}\  \v{u}_j^{\m{X}})^2}{\mathrm{sg}({\v{u}_i^{\m{X}}}^\top\  \m{K}_{\m{X}, \m{X}}\  \v{u}_i^{\m{X}})}\right) \\
\nabla_{\theta_j} \ell &= \frac{2}{B^2} \m{K}_{\m{X}, \m{X}} \left(\v{u}_j^{\m{X}} - \sum_{i=1}^{j-1} \frac{({\v{u}_i^{\m{X}}}^\top\  \m{K}_{\m{X}, \m{X}}\  \v{u}_j^{\m{X}})^2}{{\v{u}_i^{\m{X}}}^\top\  \m{K}_{\m{X}, \m{X}}\  \v{u}_i^{\m{X}}}\v{u}_j^{\m{X}}\right) \cdot \nabla_{\theta_j} \v{u}_j^{\m{X}}
\end{align}

where $\mathrm{sg}$ denotes the "stop gradient" operation. Notably, the term in the gradient within the brackets is exactly the Gram-Schmidt orthogonalisation of $\v{u}_j^{\m{X}}$ with respect to all $\v{u}_j^{\m{X}}, i $ 


### Eigendecomposition of Modern NN Kernels

The NNGP and NTK kernels can be very expensive, or even analytically intractable to compute. A common approach is to estimate them by Monte Carlo sampling with finite width networks. This is straightforward for the NNGP:

$$
k_{NNGP}(x,x') = \mathbb{E}_{\theta \sim p(\theta)} \big[ g(x,\theta) g(x,\theta)^\top \big] \approx \frac{1}{S}\sum_{s=1}^S g(x,\theta_s) g(x,\theta_s)^\top
$$

The NTK is not so simple to estimate, as it involves the outer product of the Jacobian of the network. For large models, this is very expensive to compute and store exactly. Here we focus on estimating the "empirical NTK" of a finite network, given by

$$
k_{NTK}(x,x') = \nabla_\theta g(x, \theta) \nabla_\theta^\top g(x, \theta).
$$

Observing that, for any $p(\v{v})$ such that $\mathbb{E}_{\v{v} \sim p(\v{v})}\left[ \v{v} \v{v}^\top \right]$ = $\m{I}_{dim(\theta)}$:

$$
k_{NTK}(x,x') = \mathbb{E}_{\v{v} \sim p(\v{v})} \left[ \left( \nabla_\theta g(x, \theta)\v{v} \right) \left( \nabla_\theta g(x, \theta)\v{v}\right) ^\top \right]
$$

Taking a Taylor expansion, $\nabla_\theta g(x, \theta)\v{v} \approx (g(x, \theta + \varepsilon\v{v}) - g(x, \theta)) / \varepsilon$, and

$$
k_{NTK}(x,x') \approx \frac{1}{S} \sum_{s=1}^S \left[ \left( \frac{ g(x, \theta + \varepsilon\v{v}) - g(x, \theta) }{\varepsilon} \right) \left( \frac{ g(x, \theta + \varepsilon\v{v}) - g(x, \theta) }{\varepsilon} \right) ^\top \right].
$$

Both of these approximations require $S$ forward passes of the network with different parameter settings. To obtain good approximations in practice, $S$ must be large (the authors use $S=4000$). The hope is to be able to approximate this kernel with $k \ll S$ eigenfunctions to significantly speed up evaluation at test time.

![](https://i.imgur.com/b2WgfDI.png)

<!-- ![](https://i.imgur.com/cwiAbGz.png) -->

![](https://i.imgur.com/fKF6Ef5.png)


#### Scaling up the Linearised Laplace Approximation (LLA)

The LLA is a method to obtain predictive uncertainty estimates for neural networks trained by MAP. The approximate posterior in function space is given by:

$$
\cl{GP}(f | g(x, \v{\theta}_{\mathrm{MAP}}), \nabla_\theta g(x, \v{\theta}_{\mathrm{MAP}}) \m{\Sigma} \nabla_\theta g(x, \v{\theta}_{\mathrm{MAP}}) ^\top)
$$

where $\m{\Sigma}$ is the inverse Gauss-Newton matrix (an approximation to the Hessian). $\m{\Sigma}$ has size $\mathrm{dim}(\v{\theta}) \times \mathrm{dim}(\v{\theta})$, so for a large network this is expensive to invert. It has the form

$$
\m{\Sigma}^{-1} = \sum_i \nabla_\theta g(x_i, \v{\theta}_{\mathrm{MAP}}) ^\top \m{\Lambda}_i \nabla_\theta g(x_i, \v{\theta}_{\mathrm{MAP}}) + \frac{1}{\sigma_0^2} \m{I}_{\mathrm{dim}(\v{\theta})} \\
\mathrm{where} \quad \m{\Lambda}_i = \nabla^2_{\v{f}\v{f}} \log p(y_i | \v{f}) |_{\v{g} = g(x_i | \v{\theta}_{\mathrm{MAP}})} \in \mathbb{R}^{N_{out} \times N_{out}}
$$

This covariance matrix can be approximated using the eigenfunction approximation to the NTK, but now evaluated at the MAP parameters:

$$
k_{NTK}(x,x') = \nabla_\theta g(x, \v{\theta}_{\mathrm{MAP}}) \nabla_\theta^\top g(x, \v{\theta}_{\mathrm{MAP}}) \approx \v{\psi}(x_i)^\top \v{\psi}(x_i)\\
\mathrm{where} \quad \v{\psi}(x) = \left[ \sqrt{\hat{\lambda}_1}u_1(x), \dots, \sqrt{\hat{\lambda}_k}u_k(x)  \right]
$$

The LLA posterior can now be approximated as:

$$
\cl{GP}(f | g(x, \v{\theta}_{\mathrm{MAP}}), \v{\psi}(x) \left[ \sum_i \v{\psi}(x_i)^\top \m{\Lambda}_i  \v{\psi}(x_i) + \frac{1}{\sigma_0^2}\m{I}_k\right]^{-1} \v{\psi}(x) ^\top)
$$

where the matrix to be inverted now only has dimension $k \times k$.


Compared to other scalable LLA methods, this method performs favourably on NLL and Expected Calibration Error (ECE):
![](https://i.imgur.com/uZq528N.png)


### Structured Representation Learning via Contrastive Learning

In contrastive learning, a clean data point $\v{\bar{x}}$ is used to generate random augmentations according to some distribution $p(\v{x} | \v{\bar{x}})$. The goal of contrastive learning is then to learn similar (in some sense) representations for augmentations which are generated from the same clean data point. One way of phrasing this problem is as learning a kernel function which measures similarity of data points according to the augmentation distribution $p(\v{x} | \v{\bar{x}})$:

$$
k(\v{x}, \v{x'}) := \frac{p(\v{x}, \v{x'})}{p(\v{x})p(\v{x'})}\\
\mathrm{where} \quad p(\v{x}, \v{x'}) := \mathbb{E}_{p_d(\v{\bar{x}})} \left[ p(\v{x} | \v{\bar{x}}) p(\v{x'} | \v{\bar{x}}) \right]
$$

and $p_d(\v{\bar{x}})$ is the clean data distribution. $p(\v{x})$ and $p(\v{x'})$ are the respective marginals of $p(\v{x}, \v{x'})$.

This kernel gives a measure of how likely is is that $\v{x}$ and $\v{x'}$ were generated from the same clean data point.

Plugging this kernel into the eigenfunction loss:

\begin{align}
\Pi_{ij} &= \mathbb{E}_{p(x)}\mathbb{E}_{p(x')}\big[k(x,x')\hat{u}_i(x)\hat{u}_j(x')\big] \\
&= \mathbb{E}_{p(x)}\mathbb{E}_{p(x')}\left[\frac{p(\v{x}, \v{x'})}{p(\v{x})p(\v{x'})}\hat{u}_i(x)\hat{u}_j(x')\right] \\
&= \mathbb{E}_{p(x, x')}\left[\hat{u}_i(x)\hat{u}_j(x')\right] \\
&\approx \frac{1}{b}\sum_{i=1}^{b} \hat{u}_i(x_i)\hat{u}_j(x'_i)
\end{align}

where $x_i$ and $x'_i$ here are two independent samples generated from the same clean data point $\bar{x}_i$. This loss can then be optimised in exactly the same way as the standard NeuralEF.

A benefit of this approach is that the feature representations (eigenfunctions) are ordered in terms of importance. This means that they are well suited to be used as ***adaptive length codes***. An application of this is in image retrieval where a shorter code can lead to much lower time and storage requirements for retrieval of the top $M$ closest examples, while longer codes provide better accuracy when possible.

![](https://i.imgur.com/kRqvfPh.png)
