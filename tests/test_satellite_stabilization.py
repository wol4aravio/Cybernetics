import scipy.integrate
import numpy as np

from cybernetics.real.satellite_stabilization import SatelliteStabilization


def test_smoke():
    satellite = SatelliteStabilization(
        u1 = lambda t, s: 0.0,
        u2 = lambda t, s: 0.2 * s[0] * s[2],
        u3 = lambda t, s: -s[0] * s[1])

    p0, q0, r0 = 24, 16, 16    
    
    sol = scipy.integrate.solve_ivp(
        fun=satellite.model_eq_np, 
        t_span=[0, 1], 
        y0=[p0, q0, r0, 0], 
        method='RK45', 
        t_eval=np.linspace(0, 1, num=101))

    assert np.all(sol.y[0] == p0)
