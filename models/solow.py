# models/solow.py
import numpy as np
from scipy.integrate import solve_ivp
from .base_model import GrowthModel
import matplotlib.pyplot as plt


class SolowModel(GrowthModel):
    DEFAULT_PARAMS = {
        's': 0.3,
        'alpha': 0.33,
        'n': 0.01,
        'g': 0.02,
        'delta': 0.05
    }

    def steady_state(self, **params):
        p = {**self.DEFAULT_PARAMS, **params}
        s = p['s']
        alpha = p['alpha']
        n = p['n']
        g = p['g']
        delta = p['delta']

        return (s / (n + g + delta)) ** (1 / (1 - alpha))

    def simulate(self, k0, t_span=(0, 100), **params):
        p = {**self.DEFAULT_PARAMS, **params}

        def dk_dt(t, k):
            return (p['s'] * self.production(k, p['alpha'])
                    - (p['n'] + p['g'] + p['delta']) * k)

        sol = solve_ivp(dk_dt, t_span, [k0], t_eval=np.linspace(*t_span, 500))
        return sol.t, sol.y[0]

    def phase_diagram(self, k0_list=None, t_span=(0, 100), n_arrows=20, labels=True, legend=True):
        self.s = self.DEFAULT_PARAMS['s']
        self.alpha = self.DEFAULT_PARAMS['alpha']
        self.n = self.DEFAULT_PARAMS['n']
        self.g = self.DEFAULT_PARAMS['g']
        self.delta = self.DEFAULT_PARAMS['delta']
        if k0_list is None:
            k_star = self.steady_state()
            k0_list = [0.5 * k_star, 0.8 * k_star, 1.2 * k_star, 1.5 * k_star]

        k_star = self.steady_state()
        plt.figure(figsize=(10, 6))

        k_min = min(k0_list) * 0.8
        k_max = max(k0_list) * 1.2
        t = np.linspace(t_span[0], t_span[1], 20)
        k = np.linspace(k_min, k_max, 15)
        T, K = np.meshgrid(t, k)

        dkdt = self.s * K ** self.alpha - (self.n + self.g + self.delta) * K
        dt = np.ones_like(dkdt)

        magnitude = np.sqrt(dt ** 2 + dkdt ** 2)
        dt_normalized = dt / magnitude
        dkdt_normalized = dkdt / magnitude

        plt.quiver(T, K, dt_normalized, dkdt_normalized,
                   color='gray', scale=20, width=0.003, alpha=0.7)

        for k0 in k0_list:
            t_vals, k_vals = self.simulate(k0, t_span=t_span)
            line, = plt.plot(t_vals, k_vals, linewidth=2,
                             label=f'$k_0={k0:.2f}$')

            if len(t_vals) > 1:
                plt.arrow(t_vals[5], k_vals[5],
                          t_vals[10] - t_vals[5], k_vals[10] - k_vals[5],
                          shape='full', color=line.get_color(),
                          length_includes_head=True,
                          head_width=0.8, head_length=1.5, alpha=0.8)

                plt.arrow(t_vals[-10], k_vals[-10],
                          t_vals[-5] - t_vals[-10], k_vals[-5] - k_vals[-10],
                          shape='full', color=line.get_color(),
                          length_includes_head=True,
                          head_width=0.8, head_length=1.5, alpha=0.8)

        plt.axhline(y=k_star, color='r', linestyle='--',
                    label='$k^*$ (уст. сост.)')

        if labels:
            plt.title("График выхода на устойчивый уровень капиталовооружённости")
            plt.xlabel("Время $t$")
            plt.ylabel("Капитал на эффективного работника $k(t)$")
        if legend:
            plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()