 import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Simulación de 3I/ATLAS bajo CMG–LCE + Emisión Ni(CO)4
print("Iniciando simulación 3I/ATLAS...")

# Parámetros
G = 6.67430e-11
M_sun = 1.989e30
mu = 1e-20
r0 = 1e12
v0 = 3e4
tau = 1e7
f = 1e-5
t_span = (0, 3e7)

# Ni(CO)4
k_emission = 1e20
threshold_rho_dot = 1e-36

# Funciones Ψ
def Psi(t): return np.sin(2 * np.pi * f * t) * np.exp(-t / tau)
def dPsi_dt(t): return 2 * np.pi * f * np.cos(2 * np.pi * f * t) * np.exp(-t / tau) - (1/tau) * Psi(t)
def ddPsi_dt2(t):
    term1 = - (2 * np.pi * f)**2 * Psi(t)
    term2 = - (1/tau) * 2 * np.pi * f * np.cos(2 * np.pi * f * t) * np.exp(-t / tau)
    term4 = (1/tau)**2 * Psi(t)
    return term1 + 2*term2 + term4

# ODE
def ode(t, y):
    r, v = y
    a_grav = -G * M_sun / r**2
    a_Psi = -mu * dPsi_dt(t) * ddPsi_dt2(t)
    return [v, a_grav + a_Psi]

# Resolver
sol = solve_ivp(ode, t_span, [r0, -v0], method='RK45', t_eval=np.linspace(t_span[0], t_span[1], 1000))

# Cálculos
a_Psi_t = -mu * np.array([dPsi_dt(tt) for tt in sol.t]) * np.array([ddPsi_dt2(tt) for tt in sol.t])
rho_dot = -mu * np.array([dPsi_dt(tt) for tt in sol.t]) * np.array([ddPsi_dt2(tt) for tt in sol.t])
I_Ni = k_emission * np.maximum(0, np.abs(rho_dot) - threshold_rho_dot)

# Gráfico
fig, ax1 = plt.subplots(figsize=(10, 8))
ax1.plot(sol.t / 86400, sol.y[0] / 1.496e11, label='Posición (AU)', color='blue')
ax1.set_xlabel('Tiempo (días)')
ax1.set_ylabel('Posición (AU)', color='blue')
ax1.legend(loc='upper left')
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(sol.t / 86400, a_Psi_t, label='a_Ψ', color='red', ls='--')
ax2.set_ylabel('a_Ψ (m/s²)', color='red')
ax2.legend(loc='upper right')

ax3 = fig.add_axes([0.125, 0.1, 0.775, 0.1])
ax3.plot(sol.t / 86400, I_Ni, label='I_Ni(CO)₄', color='green')
ax3.set_ylabel('I_Ni')
ax3.legend(loc='upper right')
ax3.grid(alpha=0.3)

plt.title('3I/ATLAS: Trayectoria + Ni(CO)₄ — CMG-LCE')
plt.tight_layout()
plt.savefig('atlas_simulation_with_NiCO4.png')
plt.show()

print("Simulación completada. Gráfico guardado: atlas_simulation_with_NiCO4.png")