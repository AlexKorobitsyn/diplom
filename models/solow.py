#models/solow.py
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

    def __init__(self):
        self.delta = 0.05
        self.g = 0.02
        self.n = 0.01
        self.alpha = 0.33
        self.s = 0.3

    def steady_state(self, **params):
        self.s = params.get('s', self.DEFAULT_PARAMS['s'])
        self.alpha = params.get('alpha', self.DEFAULT_PARAMS['alpha'])
        self.n = params.get('n', self.DEFAULT_PARAMS['n'])
        self.g = params.get('g', self.DEFAULT_PARAMS['g'])
        self.delta = params.get('delta', self.DEFAULT_PARAMS['delta'])

        return ((self.n + self.g + self.delta) / self.s) ** (1 / (self.alpha-1))

    def simulate(self, k0, t_span=(0, 100), **params):
        p = {**self.DEFAULT_PARAMS, **params}
        def dk_dt(t, k):
            return (p['s'] * self.production(k, p['alpha'])
                    - (p['n'] + p['g'] + p['delta']) * k)
        if isinstance(k0, float) or isinstance(k0, int):
            sol = solve_ivp(dk_dt, t_span, [k0], t_eval=np.linspace(*t_span, 500))
        else:
            sol = solve_ivp(dk_dt, t_span, k0, t_eval=np.linspace(*t_span, 500))
        return sol.t, sol.y[0]


    def phase_diagram(self, k0_list=None, t_span=(0, 100), n_arrows=20, labels=True, legend=True):
        print(type(k0_list))
        """
        Фазовая диаграмма модели Солоу: Показывает, как капитал k(t) стремится к стационарному значению k*. 
        Добавляет стрелки, указывающие направление движения k со временем.
        Параметры: k0_list: список начальных значений капитала. 
        t_span: диапазон времени. n_arrows: количество стрелочек на поле.
        labels: отображать ли подписи. legend: отображать ли легенду.
        """
        if k0_list is None:
            k0_list = [0.5 * self.steady_state(), 0.8 * self.steady_state(),
                       1.2 * self.steady_state(), 1.5 * self.steady_state()]

        k_star = self.steady_state()

        plt.figure(figsize=(10, 6))

        t = np.linspace(*t_span, 500)
        our_max =  2 * k_star
        for k0 in k0_list:
            print(k0)
            _, k_traj = self.simulate(k0, t_span=t_span)
            our_max = max(our_max, k_traj.max())
            plt.plot(t, k_traj, label=f'$k_0={k0:.2f}$')

        k_vals = np.linspace(0.1, our_max, n_arrows)
        t_vals = np.linspace(*t_span, n_arrows)
        T, K = np.meshgrid(t_vals, k_vals)

        dkdt = self.s * K ** self.alpha - (self.n + self.g + self.delta) * K
        dt = np.ones_like(dkdt)  # по времени шаг всегда вперед

        magnitude = np.sqrt(dt ** 2 + dkdt ** 2)
        dt /= magnitude
        dkdt /= magnitude

        plt.quiver(T, K, dt, dkdt, angles='xy', alpha=0.4, pivot='mid', color='gray')

        # Линия стационарного состояния
        plt.axhline(y=k_star, color='r', linestyle='--', label='$k^*$ (уст. сост.)')

        if labels:
            plt.title("Фазовый портрет модели Солоу")
            plt.xlabel("Время $t$")
            plt.ylabel("Капитал на эффективного работника $k(t)$")
        if legend:
            plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
