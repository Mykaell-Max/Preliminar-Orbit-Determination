import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils.orbital_mechanics import calculate_orbital_elements
from utils.constants import MU_SUN
from utils.logging_config import setup_logger

logger = setup_logger(__name__)


def better_f_and_g(tau, r_mag, mu):
    """Enhanced Lagrange coefficients with higher order terms for eccentric orbits"""
    u = mu / r_mag**3
    sqrt_u = np.sqrt(u)
    
    # very short arcs 
    if abs(tau * sqrt_u) < 0.1:
        f = 1 - 0.5 * u * tau**2 + (1/24) * u**2 * tau**4 - (1/720) * u**3 * tau**6
        g = tau - (1/6) * u * tau**3 + (1/120) * u**2 * tau**5 - (1/5040) * u**3 * tau**7
    else:
        # for longer arcs
        psi = tau**2 * u
        c2 = (1 - np.cos(np.sqrt(psi))) / psi if psi > 0 else \
            (np.cosh(np.sqrt(-psi)) - 1) / (-psi) if psi < 0 else 0.5
        c3 = (np.sqrt(psi) - np.sin(np.sqrt(psi))) / (psi * np.sqrt(psi)) if psi > 0 else \
            (np.sinh(np.sqrt(-psi)) - np.sqrt(-psi)) / ((-psi) * np.sqrt(-psi)) if psi < 0 else 1/6
        f = 1 - psi * c2
        g = tau * (1 - psi * c3)
    return f, g


def setup_matrices(R, rho_hat):
    """Setup the matrices required for Gauss method"""
    D0 = np.column_stack([rho_hat[0], rho_hat[1], rho_hat[2]])
    D = np.zeros((3, 3))
    D[0, 0] = np.linalg.det(np.column_stack([rho_hat[1], rho_hat[2], R[1]]))
    D[0, 1] = np.linalg.det(np.column_stack([rho_hat[0], rho_hat[2], R[1]]))
    D[0, 2] = np.linalg.det(np.column_stack([rho_hat[0], rho_hat[1], R[1]]))
    D[1, 0] = np.linalg.det(np.column_stack([rho_hat[1], rho_hat[2], R[0]]))
    D[1, 1] = np.linalg.det(np.column_stack([rho_hat[0], rho_hat[2], R[0]]))
    D[1, 2] = np.linalg.det(np.column_stack([rho_hat[0], rho_hat[1], R[0]]))
    D[2, 0] = np.linalg.det(np.column_stack([rho_hat[1], rho_hat[2], R[2]]))
    D[2, 1] = np.linalg.det(np.column_stack([rho_hat[0], rho_hat[2], R[2]]))
    D[2, 2] = np.linalg.det(np.column_stack([rho_hat[0], rho_hat[1], R[2]]))
    D0_det = np.linalg.det(D0)

    if abs(D0_det) < 1e-8: 
        logger.warning("Observações quase coplanares. Tentando correção...")
        D0_det = np.sign(D0_det) * 1e-8 

    A = np.zeros((3, 3))
    b = np.zeros(3)
    A[0, 0] = -D[0, 0] / D0_det
    A[0, 1] = D[0, 1] / D0_det
    A[0, 2] = -D[0, 2] / D0_det
    A[1, 0] = -D[1, 0] / D0_det
    A[1, 1] = D[1, 1] / D0_det
    A[1, 2] = -D[1, 2] / D0_det
    A[2, 0] = -D[2, 0] / D0_det
    A[2, 1] = D[2, 1] / D0_det
    A[2, 2] = -D[2, 2] / D0_det
    b[0] = -np.dot(rho_hat[1], R[0])
    b[1] = -np.dot(rho_hat[1], R[1])
    b[2] = -np.dot(rho_hat[1], R[2])
    return A, b


def direct_solution(A, b, R, rho_hat, tau1, tau3, mu):
    """Try to find a direct solution via matrix inversion"""
    try:
        A_reg = A + np.eye(3) * 1e-5
        rho = np.linalg.solve(A_reg, b)
        logger.info(f"Distâncias calculadas: {rho[0]:.4f}, {rho[1]:.4f}, {rho[2]:.4f} UA")
        
        if all(0.1 <= r <= 10 for r in rho):
            r1 = R[0] + rho[0] * rho_hat[0]
            r2 = R[1] + rho[1] * rho_hat[1]
            r3 = R[2] + rho[2] * rho_hat[2]
            r2_mag = np.linalg.norm(r2)
            
            f1, g1 = better_f_and_g(tau1, r2_mag, mu)
            f3, g3 = better_f_and_g(tau3, r2_mag, mu)
            
            v2_from_1 = (r1 - f1 * r2) / g1
            v2_from_3 = (r3 - f3 * r2) / g3
            delta_v = np.linalg.norm(v2_from_1 - v2_from_3)
            
            if delta_v < 0.01: 
                v = (v2_from_1 + v2_from_3) / 2
                logger.info(f"Solução direta com boa consistência (delta_v = {delta_v:.6f} UA/dia)")
                return r2, v, delta_v
        else:
            logger.info("Solução direta produziu distâncias não razoáveis, tentando método iterativo...")
    except np.linalg.LinAlgError:
        logger.warning("Erro na solução direta, tentando método iterativo...")
    return None, None, float('inf')


def iterative_solution(R, rho_hat, tau1, tau3, mu, rho2_initial):
    """Find solution through iterations"""
    rho2 = rho2_initial
    best_iteration_delta_v = float('inf')
    best_iteration_result = None
    
    logger.info(f"Iniciando solução iterativa com rho2_initial = {rho2_initial:.4f}")

    for iteration in range(10): 
        r2 = R[1] + rho2 * rho_hat[1]
        r2_mag = np.linalg.norm(r2)
        
        f1, g1 = better_f_and_g(tau1, r2_mag, mu)
        f3, g3 = better_f_and_g(tau3, r2_mag, mu)
        
        if abs(g1) < 1e-10 or abs(g3) < 1e-10:
            logger.warning(f"Coeficientes g muito pequenos: g1={g1:.2e}, g3={g3:.2e}. Ajustando...")
            g1 = np.sign(g1) * max(abs(g1), 1e-10)
            g3 = np.sign(g3) * max(abs(g3), 1e-10)
        
        rho1 = np.dot(R[1] - R[0], rho_hat[1]) / np.dot(rho_hat[0], rho_hat[1]) + rho2 * np.dot(rho_hat[1], rho_hat[0])
        rho3 = np.dot(R[1] - R[2], rho_hat[1]) / np.dot(rho_hat[2], rho_hat[1]) + rho2 * np.dot(rho_hat[1], rho_hat[2])
        
        rho1 = np.clip(rho1, 0.1, 10.0)
        rho3 = np.clip(rho3, 0.1, 10.0)
        
        r1 = R[0] + rho1 * rho_hat[0]
        r3 = R[2] + rho3 * rho_hat[2]
        
        try:
            v2_from_1 = (r1 - f1 * r2) / g1
            v2_from_3 = (r3 - f3 * r2) / g3
            
            if not np.all(np.isfinite(v2_from_1)) or not np.all(np.isfinite(v2_from_3)):
                raise ValueError("Velocidades infinitas detectadas")
                
            v2 = (v2_from_1 + v2_from_3) / 2
            delta_v = np.linalg.norm(v2_from_1 - v2_from_3)
            
            if delta_v < best_iteration_delta_v:
                best_iteration_delta_v = delta_v
                best_iteration_result = (r2.copy(), v2.copy(), delta_v)
                
            if delta_v < 0.0001:  # ~1.7 m/s, muito bom
                logger.info(f"Convergiu em {iteration+1} iterações!")
                break
                
            dot_product = np.dot(v2_from_3 - v2_from_1, r2)
            adjustment = 0.15 * (1.0 / (iteration + 1))
            rho2_new = rho2 * (1 + adjustment * np.tanh(dot_product * min(0.1, delta_v)))
            
            if not np.isfinite(rho2_new) or abs(rho2_new - rho2) > 0.5:
                logger.warning("Ajuste muito grande, limitando...")
                rho2_new = rho2 + np.sign(rho2_new - rho2) * 0.1
                
            rho2 = np.clip(rho2_new, 0.1, 10.0)
            
        except Exception as e:
            logger.error(f"Erro no cálculo da velocidade: {e}")
            break
    
    return best_iteration_result


def validate_orbit(r, v, delta_v, mu):
    """Validate the orbital elements and provide information about the orbit"""
    try:
        a, e, inc, omega, w, theta = calculate_orbital_elements(r, v, mu)
        
        if not np.isfinite(a) or not np.isfinite(e) or not np.isfinite(inc):
            logger.warning("Elementos orbitais infinitos detectados, ajustando...")
            if not np.isfinite(a):
                a = 100 if a > 0 else -100  
            if not np.isfinite(e):
                e = 0.99 if a > 0 else 1.01 
            if not np.isfinite(inc):
                inc = 0 
                
        if e > 1000:
            logger.warning(f"Excentricidade extremamente alta ({e:.4f}), ajustando para um valor mais razoável")
            e = np.clip(e, 0, 10.0)
            
        logger.info(f"Solução pelo método de Gauss: a = {a:.4f} UA, e = {e:.4f}, i = {inc:.4f}°")
        logger.info(f"Velocidade orbital: {np.linalg.norm(v)*1731:.1f} km/h")
        
        orbit_valid = False
        if a > 0 and a < 100 and e < 0.99:
            logger.info("Órbita elíptica razoável detectada")
            orbit_valid = True
        elif a < 0 and e > 1.0 and e < 10:
            logger.info("Órbita hiperbólica detectada - possível objeto em trajetória de escape")
            orbit_valid = True
        elif abs(e - 1.0) < 0.01 and a > 100:
            logger.info("Órbita aproximadamente parabólica detectada")
            orbit_valid = True
            
        if orbit_valid:
            return True, r, v
        else:
            logger.warning(f"Órbita não razoável: a = {a:.4f}, e = {e:.4f}")
            if a > 100: 
                logger.warning("Semieixo maior muito grande - possível objeto muito distante ou erro numérico")
            elif a < -100:  
                logger.warning("Semieixo maior muito negativo - trajetória hiperbólica extrema ou erro numérico")
            elif e >= 0.99 and   e < 1.5:
                logger.warning("Excentricidade próxima ou pouco acima de 1 - verificar se é um cometa")
            elif e >= 1.5:  
                logger.warning("Excentricidade muito alta - provável erro numérico")

            if delta_v > 0.01:
                logger.warning(f"Alta inconsistência entre velocidades (delta_v = {delta_v:.6f} UA/dia)")
                logger.warning  ("Provável causa: arco muito curto ou observações imprecisas")

            if delta_v < 0.1: 
                logger.info("Retornando solução de melhor esforço!")
                return True, r  , v
                
            return False, None, None
            
    except Exception as e:
        logger.error(f"Erro no cálculo de elementos orbitais: {e}")
        logger.warning("Retornando vetores estado, mas elementos orbitais não puderam ser calculados")
        return True, r, v


def gauss_iod_method(R, rho_hat, t, mu=MU_SUN):
    """
    Gauss Initial Orbit Determination method, optimized for short arcs.
    
    Args:
        R (list): Lista de vetores posição da Terra [r1, r2, r3] (AU)
        rho_hat (list): Lista de vetores de direção unitários [rho_hat1, rho_hat2, rho_hat3]
        t (list): Lista de tempos de observação [t1, t2, t3]
        mu (float): Parâmetro gravitacional (AU^3/day^2)
    
    Returns:
        np.array: Vetor posição inicial r2 no tempo t2
        np.array: Vetor velocidade inicial v2 no tempo t2
    """
    
    if len(R) < 3:
        logger.error("Método de Gauss requer pelo menos 3 observações")
        return None, None
    
    if len(R) > 3:
        indices = [0, len(R) // 2, -1]
        R = [R[i] for i in indices]
        rho_hat = [rho_hat[i] for i in indices]
        t = [t[i] for i in indices]
    
    tau1 = (t[0] - t[1]).total_seconds() / 86400.0
    tau3 = (t[2] - t[1]).total_seconds() / 86400.0

    A, b = setup_matrices(R, rho_hat)
    
    starting_points = [1.0, 1.5, 2.0, 2.5]
    best_solution = None
    best_delta_v = float('inf')
    
    direct_r, direct_v, direct_delta_v = direct_solution(A, b, R, rho_hat, tau1, tau3, mu)
    if direct_r is not None and direct_delta_v < best_delta_v:
        best_solution = (direct_r, direct_v, direct_delta_v)
        best_delta_v = direct_delta_v
    
    for rho2_initial in starting_points:
        try:
            iteration_result = iterative_solution(R, rho_hat, tau1, tau3, mu, rho2_initial)
            
            if iteration_result is not None:
                r_iter, v_iter, delta_v_iter = iteration_result
                if delta_v_iter < best_delta_v:
                    best_solution = iteration_result
                    best_delta_v = delta_v_iter
                    logger.info(f"Nova melhor solução encontrada (delta_v = {best_delta_v:.6f} UA/dia)")
        except Exception as e:
            logger.error(f"Erro na tentativa com rho2_initial = {rho2_initial}: {e}")
    
    if best_solution is None:
        logger.error("Não foi possível encontrar uma solução estável.")
        return None, None
    
    r, v, delta_v = best_solution
    is_valid, final_r, final_v = validate_orbit(r, v, delta_v, mu)
    
    return final_r, final_v