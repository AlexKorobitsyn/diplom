import numpy as np
from scipy.integrate import solve_ivp
from models.base_model import GrowthModel
from logic.system_equations import system_equations
import plotly.graph_objects as go


class ResourceLimitedModel(GrowthModel):
    NAME = "Система с ресурсными ограничениями"

    DEFAULT_PARAMS = {
        'M0': 10,
        'rho': 0.05,
        'alpha': 0.33,
        'beta': 0.9,
        'mu': 0.02,
        'p0': 1.0,
        'a': 1.0,
        'u_val': 0.2,
        'v_val': 0.2,
        't_span': (0, 50),
        't_eval': np.linspace(0, 50, 1000),
        'x0': [0.9, 0.5, 1.0, 0.2, 0.2, 0.2]
    }

    def simulate(self, params: dict):
        u_val = params['u_val']
        v_val = params['v_val']

        def u_func(t, state): return u_val
        def v_func(t, state): return v_val

        ode_params = {
            'alpha': params['alpha'],
            'beta': params['beta'],
            'mu': params['mu'],
            'M0': params['M0'],
            'rho': params['rho'],
            'p0': params['p0'],
            'a': params['a'],
            'u_func': u_func,
            'v_func': v_func
        }

        sol = solve_ivp(
            fun=lambda t, y: system_equations(t, y, ode_params),
            t_span=params['t_span'],
            y0=params['x0'],
            t_eval=params['t_eval'],
            vectorized=False
        )
        self.solution = sol
    def steady_state(self, **params):
        pass

    def plot_results(self):
        if self.solution is None:
            return

        x1, x2, x3 = self.solution.y[0], self.solution.y[1], self.solution.y[2]

        fig = go.Figure(data=[go.Scatter3d(
            x=x1, y=x2, z=x3,
            mode='lines',
            line=dict(color='blue', width=3)
        )])

        fig.update_layout(
            title='Фазовый портрет: x₁, x₂, x₃',
            scene=dict(
                xaxis_title='x₁',
                yaxis_title='x₂',
                zaxis_title='x₃'
            )
        )
        fig.show()
