---
layout: archive
title: "Project"
permalink: /portfolio/
author_profile: true
---

## PySVAR

PySVAR is a mini package (or a bag-of-codes) designed for SVAR estimation across multiple identification schemes. It's simple—there will be virtually no learning curve if you're familiar with Sklearn. Simply input the parameters, and voila!

### Usage
We begin with one of the simplest possible identification method: the Cholesky identification, used by Kilian in the 2009 AER paper. Assuming $e_t=A_0^{−1}\epsilon_t$, where $e_t$ represents the reduced form errors and $\epsilon_t$ denotes the structural shocks and $A_0^{-1}$ is defined as

$$
e_t=\begin{pmatrix}
e_t^{\Delta\text{prod}}\\
e_t^{\text{rea}}\\
e_t^{\text{rop}}
\end{pmatrix}=\begin{bmatrix}
a_{11} & 0 & 0\\
a_{21} & a_{22} & 0\\
a_{31} & a_{32} & a_{33}
\end{bmatrix}\begin{pmatrix}
\epsilon_t^{\text{oil supply shock}}\\
\epsilon_t^{\text{aggregate demand shock}}\\
\epsilon_t^{\text{oil specific-demand shock}}
\end{pmatrix}
$$

To model the above problem,begin by importing `RecursiveIdentification`. Next, create an agent instance, utilizing the appropriate parameters
``` python
recr = RecursiveIdentification(data=kdata, var_names=vname, shock_names=sname, date_frequency='M', lag_order=24)
```

Once the instance is initialized, invoke `.identify()` for estimation and `.bootstrap()` to calculate the confidence interval
```python
recr.identify()
recr.bootstrap(seed=3906)
```

After identification, one can use `.irf()` or `.vd()` to calculate the impulse responses and variance decomposition, respectively. Similar to the approach taken by Kilian in his paper, I also calculate the cumulative response of $\Delta \text{prod}$ as follows:
```python
mdls = [recr]
for m in mdls:
    m.irf_point_estimate[0, :] = -np.cumsum(m.irf_point_estimate[0, :])
    m.irf_point_estimate[3, :] = np.cumsum(m.irf_point_estimate[3, :])
    m.irf_point_estimate[6, :] = np.cumsum(m.irf_point_estimate[6, :])
    m.irf_point_estimate[1, :] = -m.irf_point_estimate[1, :]
    m.irf_point_estimate[2, :] = -m.irf_point_estimate[2, :]
    for _ in range(m.irf_mat_full.shape[0]):
        m.irf_mat_full[_, 0, :] = -np.cumsum(m.irf_mat_full[_, 0, :])
        m.irf_mat_full[_, 3, :] = np.cumsum(m.irf_mat_full[_, 3, :])
        m.irf_mat_full[_, 6, :] = np.cumsum(m.irf_mat_full[_, 6, :])
        m.irf_mat_full[_, 1, :] = -m.irf_mat_full[_, 1, :]
        m.irf_mat_full[_, 2, :] = -m.irf_mat_full[_, 2, :]
   m.plot_irf(h=15, var_list=n, sigs=95, with_ci=True)
```

where the point estimate and confidence interval are stored in `irf_point_estimate` and `irf_mat_full`, respectively.  Lastly call `.plot_irf()` to plot the impulse response. Here is the plot:

![suppy](../images/suppy.pdf)
