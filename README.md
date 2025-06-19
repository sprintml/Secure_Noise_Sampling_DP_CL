# Secure Noise Sampling for Differentially Private Collaborative Learning

Link: [https://eprint.iacr.org/2025/1025](https://eprint.iacr.org/2025/1025)

Differentially private stochastic gradient descent (DP-SGD) trains machine learning (ML) models with formal privacy guarantees for the training set by adding random noise to gradient updates. In collaborative learning (CL), where multiple parties jointly train a model, noise addition occurs either (i) before or (ii) during secure gradient aggregation. The first option is deployed in distributed DP methods, which require greater amounts of total noise to achieve security, resulting in degraded model utility. The second approach preserves model utility but requires a secure multiparty computation (MPC) protocol. Existing methods for MPC noise generation require tens to hundreds of seconds of runtime per noise sample because of the number of parties involved. This makes them impractical for collaborative learning, which often requires thousands or more samples of noise in each training step.

We present a novel protocol for MPC noise sampling tailored to the collaborative learning setting. It works by constructing an approximation of the distribution of interest which can be efficiently sampled by a series of table lookups. Our method achieves significant runtime improvements and requires much less communication compared to previous work, especially at higher numbers of parties. It is also highly flexible – while previous MPC sampling methods tend to be optimized for specific distributions, we prove that our method can generically sample
noise from statistically close approximations of arbitrary discrete distributions. This makes it compatible with a wide variety of DP mechanisms. Our experiments demonstrate the efficiency and utility of our method applied to a discrete Gaussian mechanism for differentially private collaborative learning. For 16 parties, we achieve a runtime of 0.06 seconds and 11.59 MB total communication per sample, a 230× runtime improvement and 3× less communication compared to the prior state-of-the-art for sampling from discrete Gaussian distribution in MPC.

# Code

To reproduce the experiments for Secure MPC, see `dpCode/`



To reproduce the experiments for Differentially Private Collaborative ML training, see `DP_FL_prompts/`

To see an implementation of Algorithm 4 for fitting a set of lookup tables to a discrete probability distribution, see `dice_ensemble_code/`
