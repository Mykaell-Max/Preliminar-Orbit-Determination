import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import least_squares, minimize, differential_evolution
import warnings
from propagation.propagation import propagate_orbit
from utils.orbital_mechanics import estimate_angular_velocity
from .gauss_iod import gauss_iod_method
from utils.logging_config import setup_logger

logger = setup_logger(__name__)

def differential_correction(R, v_earth, rho_hat, t, mu_sun, perturbations=None):
    """
    Applies differential correction to refine the initial estimate of the object's orbit

    Args:
        R (list): List of position vectors (earth)
        v_earth (list): List of velocity vectors (earth)
        rho_hat (list): List of unit direction vectors
        t (list): List of datetime objects
        mu_sun (float): Gravitational parameter of the Sun
    
    Returns:
        np.array: Refined position vector
        np.array: Refined velocity vector
    """
    r_initial, v_initial = gauss_iod_method(R, rho_hat, t)   
   
    if r_initial is None:
        best_result = None
        best_cost = float('inf')
        r_earth_sun = np.linalg.norm(R[0])
        bounds = [ # muito restritivo?
            # Posição: região dentro de 5 AU do Sol
            (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0),
            # Velocidade: até 0.07
            (-0.07, 0.07), (-0.07, 0.07), (-0.07, 0.07)
        ]
        def objective(params):
            r0, v0 = params[:3], params[3:]
            r_norm = np.linalg.norm(r0)
            if r_norm < 0.1 or r_norm > 10.0:
                return 1e10
         
            try:
                r_pred, _ = propagate_orbit(r0, v0, t, t[0], mu_sun)
                cost = 0
                weights = np.linspace(1.0, 0.8, len(t))
            
                for i in range(len(t)):
                    rho_vec = r_pred[i] - R[i]
                    rho_norm = np.linalg.norm(rho_vec)
                    if rho_norm < 1e-10:
                        return 1e10
                    rho_pred = rho_vec / rho_norm
                    ang_error = np.arccos(np.clip(np.dot(rho_pred, rho_hat[i]), -1.0, 1.0))
                    cost += weights[i] * ang_error**2
                r_norm = np.linalg.norm(r0)
                v_norm = np.linalg.norm(v0)
                energy = v_norm**2 / 2 - mu_sun / r_norm
                h = np.cross(r0, v0)
                h_norm = np.linalg.norm(h)
                if h_norm > 1e-10:
                    e_vec = np.cross(v0, h) / mu_sun - r0 / r_norm
                e = np.linalg.norm(e_vec)
                if energy > 0:  
                    cost += 200 * energy  
                elif -mu_sun / (2 * energy) > 5.0: 
                    cost += 50 * (-mu_sun / (2 * energy) - 5.0)
                if e > 0.95: 
                    cost += 100 * (e - 0.95)**2
                return cost
            except Exception:
                return 1e10
        for trial in range(7): 
            r_scale = 0.7 + 0.3 * trial 
            v_scale = 0.008 + 0.008 * trial
            x0 = np.zeros(6)
            random_offset = np.random.uniform(-0.05, 0.05, 3) if trial > 0 else np.zeros(3)
            x0[:3] = R[0] + r_scale * (rho_hat[0] + random_offset)
            rho_hat_dot = estimate_angular_velocity(rho_hat, t)
            v_offset = np.random.uniform(-0.003, 0.003, 3) if trial > 0 else np.zeros(3)
            x0[3:] = v_earth[0] + v_scale * rho_hat[0] + r_scale * rho_hat_dot + v_offset
            try:
                methods = ['Nelder-Mead', 'Powell'] if trial < 2 else ['Nelder-Mead']
                for method in methods:
                    result = minimize(objective, x0, method=method, bounds=bounds, options={'maxiter': 1500})
                    if result.fun < best_cost:
                        best_cost = result.fun
                        best_result = result.x
            except:
                continue
        if best_result is not None:
            r_initial, v_initial = best_result[:3], best_result[3:]
        else:
            # Se tudo falhar -> tentar otimização global 
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                result = differential_evolution(objective, bounds, popsize=25, maxiter=40, strategy='best1bin', tol=1e-7)
                r_initial, v_initial = result.x[:3], result.x[3:]
            except:
                logger.error("Falha na otimização global.")
                return None, None
    if r_initial is None:
        return None, None
   
    def residuals(params):
        r0, v0 = params[:3], params[3:]
        r_norm = np.linalg.norm(r0)
        v_norm = np.linalg.norm(v0)
        energy = 0.5 * v_norm**2 - mu_sun / r_norm
        h = np.cross(r0, v0)
        h_norm = np.linalg.norm(h)
        penalty = 0
        if h_norm > 1e-10:
            a = -mu_sun / (2 * energy) if energy < 0 else float('inf')
            e_vec = np.cross(v0, h) / mu_sun - r0 / r_norm
            e = np.linalg.norm(e_vec)
            if energy > 0:  # Órbita hiperbólica
                penalty += 500 * energy
            elif a < 0.3:  # Órbita muito pequena
                penalty += 100 * (0.3 - a)**2
            elif a > 5.0:  # Órbita muito grande
                penalty += 50 * (a - 5.0)**2
            if e > 0.98:  # Excentricidade muito alta
                penalty += 200 * (e - 0.98)**2
        else:
            penalty += 1000
        try:
            r_pred, _ = propagate_orbit(r0, v0, t, t[0], mu_sun, perturbations)
        except Exception as e:
            return np.ones(len(t) * 3) * 1e8 + penalty
        weights = np.linspace(1.2, 0.8, len(t))
        resid = []
        for i in range(len(t)):
            rho_vec = r_pred[i] - R[i]
            rho_norm = np.linalg.norm(rho_vec)
            if rho_norm < 1e-10:
                return np.ones(len(t) * 3) * 1e8 + penalty
            rho_pred = rho_vec / rho_norm
            resid.append((rho_pred - rho_hat[i]) * weights[i])
        flat_resid = np.concatenate(resid)
        if penalty > 0:
            flat_resid = flat_resid + penalty * np.ones_like(flat_resid) / len(flat_resid)
        return flat_resid
   
    # rfinamento final com least_squares usando bounds mais amplos
    initial_params = np.hstack((r_initial, v_initial))
    bounds = ([-15, -15, -15, -0.15, -0.15, -0.15], [15, 15, 15, 0.15, 0.15, 0.15])
   
    try:
        result = least_squares(residuals, initial_params, method='trf', bounds=bounds, ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=10000, verbose=0)
        r_final, v_final = result.x[:3], result.x[3:]
        r_norm = np.linalg.norm(r_final)
        v_norm = np.linalg.norm(v_final)
        energy = 0.5 * v_norm**2 - mu_sun / r_norm
        a = -mu_sun / (2 * energy) if energy < 0 else float('inf')
        h = np.cross(r_final, v_final)
        h_norm = np.linalg.norm(h)
        if h_norm > 1e-10:
            e_vec = np.cross(v_final, h) / mu_sun - r_final / r_norm
            e = np.linalg.norm(e_vec)
            i = np.degrees(np.arccos(abs(h[2]) / h_norm))
            if energy > 0: 
                logger.warning(f"Solução hiperbólica: energy={energy:.8f}, e={e:.6f}")
                if energy > 1e-4: 
                    return None, None
            elif a > 10.0:
                logger.warning(f"Solução com semi-eixo maior excessivo: a={a:.6f} AU")
                return None, None
            elif e > 0.99:
                logger.warning(f"Solução com excentricidade excessiva: e={e:.6f}")
                return None, None
            logger.info(f"Solução aceita: a={a:.6f} AU, e={e:.6f}, i={i:.6f}°")
            return r_final, v_final
        else:
            logger.warning("Solução com momento angular quase nulo.")
            return None, None
    except Exception as e:
        logger.error(f"Erro no refinamento: {e}")
        return None, None
