#Using Finite Difference Method
import numpy as np
from scipy.optimize import root
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

N = 400
z_max = 30
z = np.linspace(0,z_max,N)
Omega_m0 = float(input("Enter the dimensionless matter density: "))
lamb = 1.2
step_size = z_max/(N-1)
b = 1
E_guess = (Omega_m0*(1+z)**3+(1-Omega_m0))**(1/2)
def Hu_Sawicki(R_vals):
    R_safe = np.maximum(R_vals,1e-5)
    f = R_safe*(1-((2*lamb)/(b*lamb+R_safe)))
    f_R = 1 - 2*b*lamb**2/((b*lamb+R_safe)**2)
    f_RR = 4*b*lamb**2/((b*lamb+R_safe)**3)
    return f,f_R,f_RR
def residual(E):
    residuals = np.zeros(N)
    for i in range(1,N-1):
        E_p = (E[i+1]-E[i-1])/(2*step_size)
        E_pp = (E[i+1]-2*E[i]+E[i-1])/step_size**2
        R = 6*(2*E[i]**2 - (1+z[i])*E[i]*E_p)
        R_p = 6*(3*E[i]*E_p-(1+z[i])*E_p**2-(1+z[i])*E[i]*E_pp)
        f, f_R, f_RR = Hu_Sawicki(R)
        lhs = f_R
        rhs = Omega_m0*(1+z[i])**3 - (f_R*R-f)/6 -(1+z[i])*R_p*f_RR
        residuals[i] = lhs - rhs
    E_p_last = (E[-1]-E[-2])/step_size
    residuals[0] = E[0] - 1
    residuals[N-1] = E[N-1] - E_p_last*62/3
    return residuals
solution = root(residual,E_guess)
plt.scatter(z,solution.x,label = 'Actual Solution')
plt.plot(z,E_guess, label = 'Guess solution', color = 'orange')
plt.legend()
plt.xlabel('z')
plt.ylabel('E(z)')
plt.title("E(z) vs Z")
plt.grid(True)
plt.show()

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

Omega_vals = np.linspace(0.24,0.34,10)
H0_vals = np.linspace(65,75,10)
best_chi2 = float('inf')
best_omega = 0.35
best_H0 = 70.0
for Om in Omega_vals:
    E_g = (Om*(1+z)**3+(1-Om))**0.5
    lamb = 1-Om
    solution = root(residual,E_g)
    E_sol = solution.x
    E_interp = interp1d(z, E_sol, kind='cubic')
    for H0_test in H0_vals:
        H_model_point = H0_test * E_interp(z_data)

        # --- CHANGE 2: Corrected Chi^2 operator grouping brackets ---
        chi2 = np.sum(((H_data - H_model_point) / err_data) ** 2)

        if chi2 < best_chi2:
            best_chi2 = chi2
            best_H0 = H0_test
            best_omega = Om

print("Best Omega =" , best_omega)
print("Best H0 =", best_H0)
print("Best_chi", best_chi2)

omega_m0 = best_omega
E_guess_best = (omega_m0*(1+z)**3+(1-omega_m0))**0.5
best_solution = root(residual,E_guess_best,method = 'hybr')
plt.plot(z,best_H0*best_solution.x,label = 'Best Solution')
plt.scatter(z_data,H_data, label = 'Data points')
plt.errorbar(z_data,H_data,yerr = err_data, label = 'Error bar')
plt.xlim(0, 2.3)
plt.ylim(30, 260)
plt.grid(True)
plt.xlabel('Z')
plt.ylabel('H(z)')
plt.title("H(z) vs Z by Hu-Sawicki Model for b=1")
plt.legend()
plt.show()
