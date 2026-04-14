import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from cobra.io import read_sbml_model
from cobra import Model


# ============================================================
# 1) YOUR ADAPTIVE MODEL (UNCHANGED)
# ============================================================
class AdaptiveMetabolicSimulator:
    def __init__(self, params, initial_conditions, t, mode='single'):
        self.params = params
        self.initial_conditions = initial_conditions
        self.t = t
        self.mode = mode

    def _single_species_model(self, y, t):
        v_gal = self.params['v_gal']
        v_eth = self.params['v_eth']
        K_gal = self.params['K_gal']
        K_eth = self.params['K_eth']
        Y = self.params['Y']
        Q = self.params['Q']
        delta = self.params['delta']
        d = self.params['d']

        n, c_gal, c_eth, a_gal, a_eth = y
        r_gal = c_gal / (K_gal + c_gal)
        r_eth = c_eth / (K_eth + c_eth)

        dn_dt = n * (v_gal * a_gal * r_gal + v_eth * a_eth * r_eth - delta)
        dc_gal_dt = -n * a_gal * r_gal
        dc_eth_dt = -n * a_eth * r_eth + Y * n * a_gal * r_gal

        total_uptake = a_gal + a_eth
        theta = 1 if total_uptake >= Q else 0
        penalty = theta * (total_uptake / Q) * (v_gal * r_gal * a_gal + v_eth * r_eth * a_eth)

        da_gal_dt = a_gal * d * delta * (v_gal * r_gal - penalty)
        da_eth_dt = a_eth * d * delta * (v_eth * r_eth - penalty)

        return [dn_dt, dc_gal_dt, dc_eth_dt, da_gal_dt, da_eth_dt]

    def run(self, plot=False):
        if self.mode == 'single':
            sol = odeint(self._single_species_model, self.initial_conditions, self.t)
        else:
            raise ValueError("This exporter currently expects mode='single'.")

        self.solution = sol

        if plot:
            self.plot()

        return self.t, sol

    def plot(self):
        sol = self.solution
        n, c_gal, c_eth, a_gal, a_eth = sol.T

        plt.figure(figsize=(10, 8))

        plt.subplot(3, 1, 1)
        plt.plot(self.t, n, color='black', label='Population')
        plt.yscale('log')
        plt.ylabel('Cells/mL')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(self.t, c_gal, label='Galactose', color='blue')
        plt.plot(self.t, c_eth, label='Ethanol', color='red')
        plt.ylabel('Resource')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(self.t, a_gal, label='Galactose Strategy', linestyle='--', color='blue')
        plt.plot(self.t, a_eth, label='Ethanol Strategy', linestyle='--', color='red')
        plt.xlabel('Time (h)')
        plt.ylabel('Strategy')
        plt.legend()

        plt.tight_layout()
        plt.savefig("adaptive_simulation.png", dpi=300, bbox_inches="tight")
        plt.close()


# ============================================================
# 2) EXPORTER: ADAPTIVE CRM -> predictions.csv
# ============================================================
def build_predictions_from_adaptive(
    t: np.ndarray,
    sol: np.ndarray,
    params: dict,
    gDW_per_cell: float = 4.765e-12,
    concentration_scale: float = 1000.0,
    output_csv: str = "predictions.csv",
) -> pd.DataFrame:
    """
    Convert adaptive CRM single-mode trajectory into a predictions dataframe
    for guided dFBA.

    Important:
    - This does NOT change the simulation.
    - It only derives extra columns from the already computed trajectory.
    - Biomass curve remains exactly the same as the adaptive model output.

    Parameters
    ----------
    t : np.ndarray
        Time points from the adaptive simulator.
    sol : np.ndarray
        Solution array with columns [n, c_gal, c_eth, a_gal, a_eth].
    params : dict
        Adaptive model parameter dictionary.
    gDW_per_cell : float
        Biomass conversion.
    concentration_scale : float
        Optional scaling factor to map CRM concentration units into something
        more useful for guided dFBA. Use 1.0 if you want raw CRM units.
    output_csv : str
        Filename for predictions CSV.
    """
    t = np.asarray(t, dtype=float)
    sol = np.asarray(sol, dtype=float)

    v_gal = float(params["v_gal"])
    v_eth = float(params["v_eth"])
    K_gal = float(params["K_gal"])
    K_eth = float(params["K_eth"])
    Y = float(params["Y"])
    delta = float(params["delta"])

    n = sol[:, 0].astype(float)              # cells/mL
    c_gal_raw = sol[:, 1].astype(float)
    c_eth_raw = sol[:, 2].astype(float)
    a_gal = sol[:, 3].astype(float)
    a_eth = sol[:, 4].astype(float)

    # Keep raw CRM state
    c_gal = c_gal_raw.copy()
    c_eth = c_eth_raw.copy()

    # Optional rescaled concentrations for dFBA guidance
    galactose_scaled = c_gal_raw * concentration_scale
    ethanol_scaled = c_eth_raw * concentration_scale

    # Monod terms from the CRM itself
    r_gal = c_gal_raw / (K_gal + c_gal_raw + 1e-12)
    r_eth = c_eth_raw / (K_eth + c_eth_raw + 1e-12)

    # Exact CRM growth-rate expression
    mu = v_gal * a_gal * r_gal + v_eth * a_eth * r_eth - delta

    # Strategy-weighted uptake proxies from CRM
    u_gal_proxy = a_gal * r_gal
    u_eth_proxy = a_eth * r_eth

    # Biomass conversion for dFBA-facing outputs
    biomass_gDW_L = n * 1000.0 * gDW_per_cell

    # Derivative-based state changes
    dt_mean = float(np.mean(np.diff(t)))
    dn_dt = np.gradient(n, dt_mean)
    dgal_dt = np.gradient(galactose_scaled, dt_mean)
    deth_dt = np.gradient(ethanol_scaled, dt_mean)

    # Convert to per-biomass rates for dFBA guidance
    q_gal = np.zeros_like(n)
    q_eth_uptake = np.zeros_like(n)
    q_eth_secretion = np.zeros_like(n)

    valid = biomass_gDW_L > 1e-18
    q_gal[valid] = np.maximum(-dgal_dt[valid] / biomass_gDW_L[valid], 0.0)

    eth_net = np.zeros_like(n)
    eth_net[valid] = deth_dt[valid] / biomass_gDW_L[valid]
    q_eth_uptake[valid] = np.maximum(-eth_net[valid], 0.0)
    q_eth_secretion[valid] = np.maximum(eth_net[valid], 0.0)

    # A smooth gate showing preference shift toward ethanol
    gate_eth = a_eth / (a_gal + a_eth + 1e-12)

    # Optional effective uptake capacities guided by strategies
    # These are not changing your CRM output; they are just exported summaries
    vmax_gal_eff = a_gal * v_gal
    vmax_eth_eff = a_eth * v_eth

    pred = pd.DataFrame({
        "time": t,
        "cells_per_ml": n,
        "biomass": biomass_gDW_L,                 # gDW/L for dFBA
        "galactose": galactose_scaled,           # scaled concentration for dFBA guidance
        "ethanol": ethanol_scaled,               # scaled concentration for dFBA guidance
        "galactose_raw": c_gal,                  # original CRM state
        "ethanol_raw": c_eth,                    # original CRM state
        "a_gal": a_gal,
        "a_eth": a_eth,
        "r_gal": r_gal,
        "r_eth": r_eth,
        "mu": mu,
        "dn_dt": dn_dt,
        "u_gal_proxy": u_gal_proxy,
        "u_eth_proxy": u_eth_proxy,
        "q_gal": q_gal,
        "q_eth_uptake": q_eth_uptake,
        "q_eth_secretion": q_eth_secretion,
        "vmax_gal_eff": vmax_gal_eff,
        "vmax_eth_eff": vmax_eth_eff,
        "yield_ethanol_from_gal": Y,
        "gate_eth": gate_eth,
    })

    pred.to_csv(output_csv, index=False)
    return pred


# ============================================================
# 3) EXAMPLE MAIN
# ============================================================
def main():
    params_single = {
        'v_gal': 1.20e10,
        'v_eth': 1.25e10,
        'K_gal': 1.e-3,
        'K_eth': 9.67e-3,
        'Y': 0.53,
        'Q': 2.18e-5,
        'delta': 2.15e-6,
        'd': 2.0e-6,
    }

    initial_conditions_single = [1e6, 5e-3, 0.0, 1.25e-11, 4.75e-12]
    t = np.linspace(0, 71, 1000)

    sim_single = AdaptiveMetabolicSimulator(
        params_single,
        initial_conditions_single,
        t,
        mode='single'
    )

    sim_t, sim_sol = sim_single.run(plot=True)

    predictions = build_predictions_from_adaptive(
        t=sim_t,
        sol=sim_sol,
        params=params_single,
        gDW_per_cell=4.765e-12,
        concentration_scale=1000.0,
        output_csv="predictions.csv",
    )

    print(predictions.head())
    print("\nSaved predictions.csv")

    # Verify biomass curve is unchanged
    plt.figure(figsize=(8, 6))
    plt.plot(sim_t, sim_sol[:, 0], label="Adaptive biomass (original)")
    plt.plot(sim_t, predictions["cells_per_ml"], "--", label="Exported biomass")
    plt.xlabel("Time (h)")
    plt.ylabel("Cells/mL")
    plt.title("Biomass check: unchanged")
    plt.legend()
    plt.tight_layout()
    plt.savefig("adaptive_simulation.png", dpi=300, bbox_inches="tight")
    plt.close()

    return predictions


if __name__ == "__main__":
    predictions = main()