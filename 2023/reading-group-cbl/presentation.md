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
Learning the principal eigenfunctions of an integral operator defined by a kernel and a data measure is at the core of many machine learning problems. Traditional non-parametric solutions based on the Nyström formula suffer from scalability issues. Recent work has shown that parametric approaches, i.e., training neural networks to approximate the eigenfunctions, can lead to satisfactory results.

In particular, Spectral Inference Networks (SpIN) offer a scalable method to approximate eigenfunctions of symmetric operators on high-dimensional function spaces using bi-level optimization methods and gradient masking (Pfau et al., 2019).

NeuralEF improves on SpIN, focusing on approximating eigenfunction expansions of kernels. The method is then applied to modern neural-network based kernels (GP-NN and NTK) as well as scaling up the linearised Laplace approximation for deep networks (Deng et al., 2022a).

Some self-supervised learning methods can be expressed in terms of approximating a kernel, so NeuralEF can be used to learn structured representations (Deng et al., 2022b).



#### References

David Pfau, Stig Petersen, Ashish Agarwal, David G. T. Barrett,and Kimberly L. Stachenfeld. "Spectral inference networks: Unifying deep and spectral learning." ICLR (2019).

Zhijie Deng, Jiaxin Shi, and Jun Zhu. "NeuralEF: Deconstructing kernels by deep neural networks." ICML  (2022).

Zhijie Deng, Jiaxin Shi, Hao Zhang, Peng Cui, Cewu Lu, Jun Zhu. "Neural Eigenfunctions Are Structured Representation Learners." arXiv preprint arXiv:2210.12637 (2022).


---

### Introduction

- Spectral methods are ubiquitous in physics and machine learning.
    - Physics: PDEs (heat and wave equations),
    - Machine Learning: 

Spectral Theorems

:::danger
**TODO**: Add some examples
:::


### Overview of reading group

This reading group will cover three papers in chronological order that build on each other.

1. Pfau et al. (2019). "Spectral inference networks: Unifying deep and spectral learning."
    - Introduces the problem and formulates a neural network objective.
    - Focus is not necessarily on kernel operators.
2. Deng et al. (2022). "NeuralEF: Deconstructing kernels by deep neural networks."
    - Improve upon SpIN algorithm (more computationally efficient and simpler).
    - Apply the method to popular DNN-inspired kernels.
3. Deng,  et al. (2022). Neural Eigenfunctions Are Structured Representation Learners
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
because we normalize the vector to lie on the unit hypersphere: $\frac{\v{u}}{\sqrt{\v{u}^\top\v{u}}} \in \cl{S}^d$. This expression is known as the Rayleigh quotient and equals the eigenvalue corresponding to the eigenvector at its maximum.

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

### From eigenvectors to eigenfunctions

The above derivation is very similar to (kernel) PCA where $\m{A} = \m{X} \m{X}^\top$ or $\m{A} = k(\m{X}, \m{X})$ for $\m{X} \in \mathbb{R}^{N\times D}$ a feature matrix.

$\m{A}$ is a $N \times N$ matrix so the optimization can become very expensive. Also unclear what to do with new points?

Nystrom approximation is not very handy as for a new datapoint
$$
u(x) = \sum k(x_n, x) u(x_n)
$$
which is computationally expensive.


Suppose that instead of a matrix $\m{A}$ we have a symmetric (not necessarily positive definite) kernel $k(x, x)$ where $x$ and $x'$ are in some measurable space $\Omega$, which could be either continuous or discrete.

Define inner product on $\Omega$ w.r.t. to input density $p(x)$ as
$$
\langle f, g \rangle = \int_\Omega f(x) g(x) p(x) \d x = \mathbb{E}_{x\sim p} [f(x)g(x)]
$$
and a kernel operator $\cl{K}$ as
$$
\cl{K} f = \langle k(x,\cdot), f\rangle = \int k(x, x') f(x') p(x') \d x'
$$

we are interested in finding the top $N$ eigenfunctions of the operator $\cl{K}$ for which hold $\cl{K}u = \lambda u$ where $u:\Omega \rightarrow \mathbb{R}$.t


In vector space:

$$
\max_{\m{U}} \tr\big[ (\m{U}^\top\m{U})^{-1} \m{U}^\top \m{A} \m{U} \big]
\iff \max_{\ \ \ \ \ \m{U} \\ \m{U}^\top \m{U} = \m{I}} \tr\big[  \m{U}^\top \m{A} \m{U} \big]
$$

In function space. Let $\v{u}:\cl{X}\rightarrow\mathbb{R}^N$ represent the N eigenfunctions (or a linear combination thereof, which means that they share the same span).

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
which gives the following objective
$$
\begin{equation}\label{test}
\cl{L}(\theta) = \max_{\v{u}_\theta} \tr\left[ \m{\Sigma}^{-1} \m{\Pi}\right]
\end{equation}
$$


Problems:
- Since the objective is invariant to linear transformation of the features $\v{u}(x)$, optimizing it will only give a function that spans the top K eigenfunctions of $\cl{K}$.
- The unconstrained objective doesn't lead to an ordering nor orthogonality of the eigenfunctions. They only have the same span.
- Minibatch estimations of $\cl{L}$ do **not** lead to unbiased estimates of the gradients. We can not use traditional SGD.


### Bi-level optimization

The objective expression is a nonlinear function of multiple expectations, so naively replacing $\m{\Sigma}, \m{\Pi}$ and their gradients with empirical estimates will be biased.

Reframe this as a bilevel optimization.

Bilevel stochastic optimization is the problem of simultaneously solving two coupled minimization.

Traditional optimization $\max_x f(x)$ s.t. $g(x) \le 0$.

Bi-level optimization the objective depends on the 'decision' of another actor whose decision depends on our action...

:::success
**Example: Toll Setting**
 Consider a network of highways that is operated by the government. The government wants to maximize its revenues by choosing the optimal toll setting for the highways. However, the government can maximize its revenues only by taking the highway users' problem into account. For any given tax structure the highway users solve their own optimization problem, where they minimize their traveling costs by deciding between utilizing the highways or an alternative route
:::

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

Bilevel stochastic problems are common in machine learning and include actor-critic methods, generative adversarial networks and imitation learning.

### Ordering through gradient masking

We can use the invariance of trace to cyclic permutation to rewrite the objective
$$
\tr \left(\m{\Sigma}^{-1} \m{\Pi}\right) = \tr\left( \m{L}^{-1} \m{\Pi} \m{L}^{-\top}\right)
$$
where $\m{\Sigma} = \m{L}\m{L}^\top$.

Then, let $\m{\Lambda} \triangleq \m{L}^{-1} \m{\Pi} \m{L}^{-\top}$.


This matrix has the convenient property that the upper left n × n block only depends on the first n eigenfunctions



![](https://i.imgur.com/MPVJ5BA.png)




| Forward  | Backward |
| -------- | -------- |
| ![](https://i.imgur.com/LoMExMr.png)     | ![](https://i.imgur.com/uOq28OS.png)     |


-------

![](https://i.imgur.com/TbDHRyh.png)

## Kernel methods

Kernel methods project data into a high-dimensional feature space $\cl{H}$ to enable linear manipulation of nonlinear data.  In other words, let $\phi: \cl{X} \rightarrow \cl{H}$ be the feature map, then $k:\cl{X} \times \cl{X} \rightarrow \mathbb{R}$ linearly depends on the feature representations of the inputs
$$
k(x, x') = \langle \phi(x), \phi(x') \rangle
$$
Often we don't know $\cl{H}$ (it is typically infinite dimensional). The subtlety is that we can leverage the "kernel trick" to bypass the need of specifying $\cl{H}$ explicitly.

There is a rich family of kernels. 

Classic kernels: Squared Exponential, Matérn Family, Linear, Polynomial, etc. They may easily fail when processing real-world data like images and texts due to inadequate expressive power and the curse of dimensionality. Thereby, various modern kernels which encapsulate

Modern kernel have been developed on the inductive biases of NN architectures. 
Let $g(\cdot, \theta): \cl{X} \rightarrow \mathbb{R}^{N_{out}}$ denote a function represented by an NN with weights $\theta \sim p(\theta)$. Then a **Neural Network Gaussian Process (NNGP)** kernel is defined as
$$
k(x,x') = \mathbb{E}_{\theta \sim p(\theta)} \big[ g(x,\theta) g(x,\theta)^\top \big]
$$
which can be computed analytically when the number of neurons in the layer goes to infinity. There are many classic papers who do this Neal (1998), Cho and Saul (2009), ...

A **Neural Tangent Kernel** (NTK) is defined as
$$
k(x,x') = \mathbb{E}_{\theta \sim p(\theta)} \big[\nabla_\theta g(x, \theta) \nabla_\theta^\top g(x, \theta) \big]
$$
by taking a first order Taylor approximation of the NN w.r.t. the weights.
$$
g(x, \color{orange}{\theta}) = g(x, {\theta_0}) + \nabla_\theta^\top g(x, \theta_0) (\color{orange}{\theta} - \theta_0)
$$
Linear model w.r.t. $\theta$ with basis functions $\phi(x)  = \nabla_\theta g(x, \theta)$.

**Topic of next week's reading group...**.


## Kernel approximations

An idea to scale up kernel methods is to approximate the kernel with the inner product of some explicit vector representations of the data
$$
k(x,x') \approx \v{\nu}(x)^\top \v{\nu}(x),
$$
where $\v{\nu}: \cl{X} \rightarrow \mathbb{R}^M$. There are two broad approaches...

### Bochner's theorem

Bochner's theorem for stationary kernels
$$
k(r) = \int S(\omega) \exp(-i \omega r) \d \omega.
$$

Random Fourier Feature approximation, given the spectrum of a stationary kernel $S(\omega)$, approximate the integral using a monte carlo estimation:
$$
k(r) \approx \sum \exp(-i \omega_k r),\quad\text{with}\quad \omega_k \sim S(\omega)
$$

- Can only be used for stationary kernel
- Scales exponentially bad with dimension

There has been a lot of research 

Mercer decomposition

$$
k(x, x') = \sum_i \lambda_i \phi_i(x) \phi_i(x')
$$

Bochner's theorem for stationary kernels
->

