import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.interpolate import interp1d
import plotly.graph_objects as go
import pandas as pd
N = 40
z_max = 30
H0 = 70
b_space = [0.001, 0.1, 0.3, 0.7, 1.0]
lamb_space = [0.5,0.8,1.1,1.4,1.8]
omega = 0.3
j_indices = np.arange(N)
x = np.cos(np.pi*j_indices/(N-1))
z = z_max / 2 * (1.0 - x)
dxdz = -2.0 / z_max
Dx = np.zeros((N,N))
c = np.ones(N)
c[0] = c[-1] =2
for j in range (N):
  for k in range(N):
    if j!=k:
      Dx[j,k] = (c[j]*(-1)**(j+k))/(c[k]*(x[j]-x[k]))
for i in range(N):
    Dx[i,i] = -np.sum(Dx[i, :])
Dxx = Dx@Dx
def Hu_Sawicki(R,b_val,lamb):
    R_safe = np.maximum(R, 1e-5)
    f = R * (1.0 - ((2.0 * lamb) / (b_val * lamb + R_safe)))
    f_R = 1.0 - (2.0 * b_val * (lamb ** 2)) / ((b_val * lamb + R_safe) ** 2)
    f_RR = (4.0 * b_val * (lamb ** 2)) / ((b_val * lamb + R_safe) ** 3)
    return f, f_R, f_RR
def r_residual(b,lamb):
    def residual(E):
        residuals = np.ones(N)
        E_x = Dx@E
        E_xx = Dxx@E
        E_p = E_x*(dxdz)
        E_pp = E_xx*(dxdz**2)
        for i in range(1, N - 1):
            R = 6.0 * (2.0 * E[i] ** 2 - (1.0 + z[i]) * E[i] * E_p[i])
            R_p = 6.0 * (3.0 * E[i] * E_p[i] - (1.0 + z[i]) * E_p[i] ** 2 - (1.0 + z[i]) * E[i] * E_pp[i])
            f, f_R, f_RR = Hu_Sawicki(R, b,lamb)

            lhs = f_R
            rhs = omega * (1.0 + z[i]) ** 3 - (f_R * R - f) / 6.0 - (1.0 + z[i]) * R_p * f_RR

                # Assign the scalar result to the current node index
            residuals[i] = lhs - rhs

        residuals[0] = E[0] - 1.0
        residuals[-1] = E[-1] - E_p[-1] * 62.0 / 3.0
        return residuals
    return residual
fig, ax = plt.subplots(figsize=(10, 6))
E_guess = np.sqrt(omega * (1.0 + z) ** 3 + (1.0 - omega))
for b in b_space:
  for lamb in lamb_space:
    res_func = r_residual(b, lamb)
    E_sol = root(res_func, E_guess, method='lm')
    if E_sol.success:
        E_guess = E_sol.x  # use previous solution as next guess
        sort_idx = np.argsort(z)
        ax.plot(z[sort_idx], H0 * E_sol.x[sort_idx],
                label=rf"b = {b}, $\lambda$ = 0.5")
    else:
        print(f"Failed to converge for b={b}")

z_data = np.array([
    0.070, 0.090, 0.120, 0.170, 0.179, 0.199, 0.200, 0.270, 0.280, 0.352,
    0.380, 0.400, 0.440, 0.470, 0.478, 0.480, 0.524, 0.593, 0.680, 0.781,
    0.875, 0.880, 0.900, 1.037, 1.260, 1.300, 1.363, 1.430, 1.530, 1.750, 1.965
])

H_data = np.array([
    69.0,  69.0,  68.6,  83.0,  75.0,  75.0,  72.9,  77.0,  88.8,  83.0,
    83.0,  95.0,  92.6,  89.0,  80.9,  97.0,  84.3,  104.0, 92.0,  105.0,
    109.4, 114.0, 117.0, 154.0, 135.0, 168.0, 160.0, 177.0, 140.0, 202.0, 186.5
])

err_data = np.array([
    19.6,  12.0,  26.2,  8.0,   4.0,   5.0,   29.6,  14.0,  11.2,  14.0,
    13.5,  17.0,  7.8,   34.0,  9.0,   62.0,  11.9,  13.0,  8.0,   12.0,
    19.2,  14.2,  23.0,  20.0,  65.0,  17.0,  33.6,  18.0,  14.0,  40.0, 50.4
])
ax.set_xlabel('$z$')
ax.errorbar(z_data,H_data,yerr = err_data)
ax.set_title(r"$H(z)$ vs $z$ for fixed value of $b$ from Chebyshev Method")
ax.set_ylabel('$H(z)$')
ax.set_xlim(-0.05, 2.1)
ax.set_ylim(30, 260)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), ncol=2, fontsize=8, borderaxespad=0)
ax.grid(True)
fig.tight_layout()
plt.show()

!pip install emcee corner
import emcee
import corner
def log_prior(theta):
  b,H0_vals,om,lamb = theta
  if not (0.0001 <= b <= 2.0):
      return -np.inf
  if not (0.5 <= lamb <= 2.0):
      return -np.inf
  if not (0.1 <= om <= 0.9):
      return -np.inf  
  if not (55.0 <= H0_val <= 85.0):
      return -np.inf
  lp  = -0.5 * ((H0_val - 70.0) / 5.0) ** 2
  lp += -0.5 * ((om    - 0.30 ) / 0.1) ** 2
  lp += -0.5 * ((b     - 0.1  ) / 0.5) ** 2
  lp += -0.5 * ((lamb  - 1.2  ) / 0.5) ** 2
  return lp
def log_likelihood(theta):
  b,H0_vals,om,lamb = theta
  try:
      E_g = np.sqrt(om * (1 + z) ** 3 + (1 - om))
      res_func = r_residual(b, lamb)
      E_sol = root(res_func, E_g, method='lm')
      if not E_sol.success:
         return -np.inf
      E_interp = interp1d(z[sort_idx], E_sol.x[sort_idx],
                            kind='cubic',
                            bounds_error=False,
                            fill_value='extrapolate')
      H_model = H0_val * E_interp(z_data)
      chi2 = np.sum(((H_data - H_model) / err_data) ** 2)
      return -0.5 * chi2
  except Exception:
      return -np.inf
def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll
dimensions = 4
walkers = 32
burn_steps = 300
n_steps = 1500
exp_mean_vals = [0.1,69,0.3,1.2]
step_size = [0.05, 0.5, 0.01,0.1]
val = exp_mean_vals + step_size*np.random.rand(walkers,dimensions)

sample_space = emcee.EnsembleSampler(walkers,dimensions,log_posterior)
state = sample_space.run_mcmc(val, burn_steps, progress=True)
sampler.reset()

sample_space.run_mcmc(state, n_steps, progress=True)
samples = sample_space.get_chain(flat=True)
b_fit, H0_fit, om_fit, lamb_fit = np.median(samples, axis=0)
b_err,  H0_err,om_err, lamb_err = np.std(samples, axis=0)
print(f"b        = {b_fit} ± {b_err:}")
print(f"lamb       = {lamb_fit} ± {lamb_err:}")
print(f"om     = {om_fit} ± {om_err}")
print(f"H0       = {H0_fit} ± {H0_err}")
labels = [r"$b$", r"$H_0$", r"$\Omega_{m0}$", r"$\lambda$"]
fig_corner = corner.corner(
    samples,
    labels=labels,
    truths=[b_fit, lamb_fit, om_fit, H0_fit],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12}
)
plt.show()

    
  
