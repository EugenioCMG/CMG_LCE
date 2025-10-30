# -*- coding: utf-8 -*-
"""
SIMULACIÓN BIG MEMORY - CMG-LCE
Ciclo cósmico completo: Expansión → Estabilización → Contracción → Reinicio

Autor: Eugenio Oliva Sánchez
DOI: 10.5281/zenodo.17460207
"""

import numpy as np
import matplotlib.pyplot as plt

# ======================
# PARÁMETROS FÍSICOS
# ======================
mu = 1.0
tau_exp = 1.0
tau_stab = 2.0
tau_cont = 1.5
t_total = tau_exp + tau_stab + tau_cont

# Tiempo
t = np.linspace(0, t_total, 10000)
dt = t[1] - t[0]

# ======================
# DEFINICIÓN DE Ψ(t) POR FASES
# ======================
Psi = np.zeros_like(t)
dPsi = np.zeros_like(t)
ddPsi = np.zeros_like(t)

# FASE 1: EXPANSIÓN
t1 = t < tau_exp
Psi[t1] = 3.0 * np.sin(2 * np.pi * t[t1] / tau_exp) * np.exp(-0.1 * t[t1])
dPsi[t1] = np.gradient(Psi[t1], dt)
ddPsi[t1] = np.gradient(dPsi[t1], dt)

# FASE 2: ESTABILIZACIÓN
t2 = (t >= tau_exp) & (t < tau_exp + tau_stab)
freq_stab = 5.0
Psi[t2] = 2.0 * np.sin(2 * np.pi * freq_stab * (t[t2] - tau_exp))
dPsi[t2] = np.gradient(Psi[t2], dt)
ddPsi[t2] = np.gradient(dPsi[t2], dt)

# FASE 3: CONTRACCIÓN
t3 = t >= tau_exp + tau_stab
Psi[t3] = 0.5 * np.exp(-(t[t3] - (tau_exp + tau_stab)) / 0.3)
dPsi[t3] = np.gradient(Psi[t3], dt)
ddPsi[t3] = np.gradient(dPsi[t3], dt)

# ======================
# LEY LCE
# ======================
rho_dot = -mu * dPsi * ddPsi
rho_Psi = np.cumsum(rho_dot) * dt
rho_Psi -= rho_Psi[0]

# ======================
# ACELERACIÓN
# ======================
a_Psi = Psi**2

# ======================
# GRÁFICO
# ======================
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axs[0].plot(t, Psi, color='purple', lw=2)
axs[0].axvspan(0, tau_exp, alpha=0.2, color='red', label='Expansión')
axs[0].axvspan(tau_exp, tau_exp+tau_stab, alpha=0.2, color='gold', label='Estabilización')
axs[0].axvspan(tau_exp+tau_stab, t_total, alpha=0.2, color='blue', label='Contracción')
axs[0].set_ylabel(r'$\Psi(t)$')
axs[0].set_title('BIG MEMORY CYCLE — CMG-LCE')
axs[0].legend()
axs[0].grid(alpha=0.3)

axs[1].plot(t, rho_dot, color='green', lw=1.5)
axs[1].axhline(0, color='k', ls='--')
axs[1].set_ylabel(r'$\dot{\rho}_\Psi$')
axs[1].grid(alpha=0.3)

axs[2].plot(t, rho_Psi, color='darkorange', lw=2)
axs[2].set_ylabel(r'$\rho_\Psi(t)$')
axs[2].grid(alpha=0.3)

axs[3].plot(t, a_Psi, color='crimson', lw=2)
axs[3].set_ylabel(r'$a_\Psi \propto \Psi^2$')
axs[3].set_xlabel('Tiempo (unidades normalizadas)')
axs[3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('simulations/big_memory_cycle.png')
plt.show()

print(f"ρ_Ψ final: {rho_Psi[-1]:.3f}")

