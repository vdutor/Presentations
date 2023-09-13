# Conditional Diffusion Models

$$
\newcommand{\d}{\mathrm{d}}
\newcommand{\v}[1]{\boldsymbol{#1}}
$$

![](https://hackmd.io/_uploads/Syd45BX63.jpg)
*Source: [Yang Song's blogpost](https://yang-song.net/blog/2021/score/), 2021*

Denoising diffusion models are a powerful class of generative models where one gradually adds noise to a data sample until the sample becomes approximately Gaussian distributed. One then learns the time reversal of this process which can generate a sample approximately distributed according to the target data distribution. Recently a lot of interest has been raised in how one can condition the learned reverse process to generate samples subject to a given condition.

## Examples

Conditioning on 'meta data'. Assume we have a diffusion model trained on images but we want to *guide* the generation to follow a text prompt (e.g., "A capybara on a surfboard"). We are thus interested in sampling from the conditional $p(image|text)$. Most famous example is DALLe 2: 

![](https://hackmd.io/_uploads/Sy2wTrQa2.png)

Alternatively, we may be interested in *inverse problems* which aim to recover $x$ from the measurement $y$ related through a likelihood model $p(y|x)$. There are many examples here:

#### Images

![](https://hackmd.io/_uploads/ByvehL7T2.png)

*Source: Chung et al., Diffusion Posterior Sampling for General Noisy Inverse Problems, ICLR 2023.*

Image inpainting example. Let $x$ be an image and $y$ a collection of pixels at location (mask) ${m}$. We create $P_m$ as a projection matrix which selects the pixels corresponding to the mask $m$. The likelihood takes the form
$$
p(y | x, m) = \delta( x[m] = y) = \delta( P_m x = y).
$$


#### Motif-Scaffolding problems in small molecules

![](https://hackmd.io/_uploads/Synh1S9an.png)
*Source: Kieran Didi, Conditional Generative Modelling with Applications in Protein Design, MPhil Cambridge 2023*

<!--
![](https://hackmd.io/_uploads/ry4KSz9T3.png)
*Source: Trippe, Yim et al., Diffusion probabilistic modeling of protein backbones in 3d for the motif-scaffolding problem, ICLR 2023.*
-->



Conceptually similar to image inpainting, we are given a part of the molecule (i.e. the motif) and are tasked to conditionally generate plausible scaffolds. However, in this setting we might only know the value $y$ but not their exact location within the protein or small molecule. In other words the mask is also latent.

Additionally, we might want to condition on other properties such as solvability, stability, efficaciousness, etc.

#### Functions

Assume we have a generative model over functions $p(f)$, we are typically interested in the conditional after observing data $y$, leading to $p(f \mid y)$. Ideally we have flexibility in our choice of likelihood $p(y | f)$, such as Gaussian or Bernoulli.

In our paper we assumed a Gaussian likelihood with almost no noise variance $\sigma^2 \ll 1$ out of necessity. Indeed, in all the plots you'll see the functions go through the datapoints. This problem started my interest in this topic.

![](https://hackmd.io/_uploads/HyPV3Lma3.png)


## Score-Based Diffusion Models

The examples above are based on diffusion models. We briefly recap the most important parts of this model.

### Forward process

![](https://yang-song.net/assets/img/score/perturb_vp.gif)
*Source: [Yang Song's blogpost](https://yang-song.net/blog/2021/score/), 2021*

Samples $x\in \mathbb{R}^d$ from $p_{data}(x)$ are progressively perturbed through a continuous-time diffusion process expressed as a Ornstein-Uhlenbeck (OU) stochastic differential equation:
$$
\d {x}_t = f(t) {x}_t \d t + g(t) \d {W}_t
$$
where we have the drift $f(t) x_t$ and diffusion $g(t)$ terms and $W_t$ Brownian motion, which can conceptually be thought of as $dW/dt \sim \mathcal{N}(0, dt)$.

**Forward density** The linearity of the drift in OU processes allows the analytic computation of the forward density, which is the probability of the state being $x_t$ at time $t$ given the initial state $x_0$ at $t=0$:
$$
p_t(x_t \mid x_0) = \mathcal{N}(x_t \mid s(t) x_0, \sigma^2(t) I).
$$

**EM** We can use the Euler–Maruyama method for the approximate numerical solution of a stochastic differential equation. We discretize a continuous interval, say $[0,T]$, into $N$ equal subintervals with width $\Delta T = T/N$. Starting from $x_0 \sim p_{data}$, we can recursively run the SDE forward
$$
x_{i+1} = x_i +  f(t_i)  x_i \Delta T + g(t_i) z_i,
$$
where $t_i = i \Delta T$ and $z_i \sim \mathcal{N}(0, \Delta T)$.

:::info
**Example: 2D Gaussian**

<!--
![](https://hackmd.io/_uploads/r1tBZm9Tn.png)
-->

Let $x \in \mathbb{R}^2$ and have a **known** data distribution given by
$$
p_{data}(x) = \mathcal{N}\Big(x \mid
\begin{bmatrix} 0 \\ 0 \end{bmatrix},
\begin{bmatrix} 1 & 0.9 \\ 0.9 & 1 \end{bmatrix}\Big) := \mathcal{N}(x \mid \mu_0, \Sigma_0).
$$
Then running the forward process from $x_0 \sim p_{data}$ looks like:
![](https://hackmd.io/_uploads/rJVKWZqah.png)
:::

The drift and diffusion coefficients of the OU process are chosen such that $\lim_{t\rightarrow \infty} p_t \approx \mathcal{N}(0, I)$.


### Reverse process

![](https://yang-song.net/assets/img/score/denoise_vp.gif)
*Source: [Yang Song's blogpost](https://yang-song.net/blog/2021/score/), 2021*

The reverse process gradually removes the noise from data. Given a noisy sample $x_T$ from the marginal distribution $p_T$, the backward process evolves from $t=T$ to $t=0$ following the SDE
$$
\d {x}_t = \left( f(t)x_t - g^2(t) \nabla_x \log p_t (x) \right) \d t + g(t) \d W^-
$$
where $W^{-}$ is the Wiener process running backwards and $dt$ represents infinitesimally small negative steps. In the reverse process, the drift now requires the gradient w.r.t. x of the (log) **marginal** forward density $\nabla_x \log p_t(x)$, which we refer to as the (Stein) **score**.

It can be shown that the marginals of the forward and reverse process are identical.

Two approximations:
- Learn the score $s_\theta(x_t, t) \approx \nabla_{x_t} \log p_t(x_t)$
- $p_{ref} = \mathcal{N}(0, I) \approx p_T$.

We train a neural network to approximate the score using "score-matching"
$$
L(\theta) = \mathbb{E}_{x_0, t, x_t} \left[\| s_\theta(x_t, t) - \nabla_x \log p_t(x_t \mid x_0) \|^2\right].
$$

:::info
**Example (cont'd): 2D Gaussian**

However, in our example we can compute the score **analytically**, as the data distribution is Gaussian, leading to a Gaussian marginal density.
\begin{align}
p_t(x_t) &= \int p_{data}(x_0) p_t(x_t \mid x_0) \mathrm{d}x_0 \\
&= \int \mathcal{N}(x_0\mid \mu_0, \Sigma_0)\,\mathcal{N}(x_t \mid s(t) x_0, \sigma^2(t) I) \mathrm{d}x_0 \\
&= \mathcal{N}(x_t \mid s(t) \mu_0, s^2(t) \Sigma_0 + \sigma^2(t) I)
\end{align}

Using the EM method we run the backward process from $t=T \rightarrow 0$:
![](https://hackmd.io/_uploads/S1Lv_QcTh.png)
:::

## Sampling

We've seen that by learning the score and reversing the SDE we can generate samples from approximately $p_{data}$. These are the main ideas behind score-based generative models which are powering many of the latest advances in image, video or audio generation.

However, people familiar with Langevin dynamics 

#### Langevin dynamics

Consider an SDE given by
$$
\d {x}_t = \nabla_x \log p({x}_t) \d t + \sqrt{2} \d {W}_t
$$
it can be shown that for $t\rightarrow \infty$ the distribution of $x_t$ is given by $p({x})$. In other words, if we have the score of a distribution we can generate samples from it. First-order EM discretization of continuous SDE. For a small stepsize $\delta t$
$$
{x}_{i+1} = {x}_i + \delta t \nabla \log p({x}_i) + \sqrt{2}{z}_i,\quad z_i \sim \mathcal{N}(0,\delta t {I})
$$

Problem with naive application of Langevin dynamics (without noise perturbation) is that it is very likely that we initialize our chain in a low density area, where the score is badly approximated. Therefore, having an inaccurate score-based model will derail Langevin dynamics from the very beginning of the procedure, preventing it from generating high quality samples that are representative of the data.

![](https://yang-song.net/assets/img/score/pitfalls.jpg)
*Source: [Yang Song's blogpost](https://yang-song.net/blog/2021/score/), 2021*

#### Predictor-corrector samplers

The main idea in Song et al. (2021) is to combine the reverse process (approximated using a numerical solver) with Langevin dynamics to correct the trajectory.

At each time step, the numerical SDE solver first gives an estimate of the sample at the next time step, playing the role of a “predictor”. Then, the score-based MCMC approach corrects the marginal distribution of the estimated sample, playing the role of a “corrector” (Song et al. 2021).


![](https://hackmd.io/_uploads/S1JHwN9a2.png)
*Source: Mathieu et al. Geometric Neural Diffusion Processes, preprint 2023.*

Time reversal (blue) and corrector steps (red) projecting the samples back onto the dynamics.


## Conditional sampling

:::info
**Example (cont'd): 2D Gaussian**

Returning back to our two-dimensional Gaussian data distribution example. An inverse problem, akin to image inpainting, is requiring the second dimension of $x$ to take specific value. Let $x = (x^{(1)}, x^{(2)})$ then we are interested in sampling from
$$
p(x^{(1)} \mid x^{(2)} = y).
$$

How can we modify our (reverse) diffusion process such that at time $t=0$ we sample from the conditional $p(x | y)$ rather than $p(x)$? Visually our samples throughout the trajectory will look like:

![](https://hackmd.io/_uploads/ryDimS9an.png)

![](https://hackmd.io/_uploads/ry4P7B5p3.png)
:::


#### Replacement method

The replacement method is the most widely used approach for approximate conditional sampling in an unconditionally trained diffusion model. In the inpainting problem, it replaces the observed dimensions of intermediate samples $x_t[m]$, with a noisy version of observation $y$. However, it is a heuristic approximation and can lead to inconsistency between inpainted region and observed region. Additionally, the replacement method is applicable only to inpainting problems, and thus not for general likelihoods $p(y \mid x)$.

![](https://hackmd.io/_uploads/BJpI_Bcp2.png)


### Conditional score

In order to perform a posteriori conditional sampling we need to modify the reverse process by using the conditional score
$$
\d {x}_t = \left( f(t)x_t - g^2(t) {\color{orange}{\nabla_{x_t} \log p_t (x_t|y)}} \right) \d t + g(t) \d W^-
$$
**proof: explain**

We can decompose the conditional score as

\begin{align}
\nabla_{x_t }\log p_t (x_t|y) &= \nabla_{x_t }\log \frac{p(y|x_t) p_t(x_t)}{p(y)} \\
&= \nabla_{x_t} \log p(y | x_t) + \nabla_{x_t} \log p_t(x_t)
\end{align}

### Amortized learning: training with conditional information

- Classifier guidance

Dhariwal and Nichol (2021) proposed classifier guidance.

Train an **additional** neural network $h_\phi$ to predict $y$ (or embedding thereof) from noisy versions $x_t$. The neural network thus learns $h_\phi \approx p(y | x_t)$. The reverse process then takes steps following
$$
\nabla_{x_t }\log p_t (x_t|y) \approx \nabla_{x_t} \log h_\phi(y, x_t, t) + s_\theta(x_t, t)
$$

Pro: we can use an unconditionally trained score network $s_\theta$ and separately train a 'classifier' for the conditioning. This method is mainly used used for $p(image|text)$ .

Cons: A downside of classifier guidance is that it requires an additional classifier model and thus complicates the training pipeline. This model has to be trained on noisy data $x_t$, so it is not possible to plug in a standard pre-trained classifier.

- Classifier free

Instead of training a separate classifier model, Ho and Salimans (2021) choose to train an unconditional denoising diffusion model $p(x)$ parameterized through a score estimator $s_\theta(x_t, t)$ together with the conditional model $p(x|y)$ parameterized through $s_\theta(x_t, y, t)$.

Immediately train a score network with conditional information. They parameterise a neural network $s_\theta(x_t, y, t)$ where $y$ can be the conditional information and




### Classifier-Guidance







### Replacement

### Reconstruction guidance
