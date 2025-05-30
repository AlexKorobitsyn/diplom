# models/ak.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from .base_model import GrowthModel


class AKModel(GrowthModel):

    DEFAULT_PARAMS = {
        's': 0.3,  # Норма сбережений
        'n': 0.01,  # Темп роста населения
        'g': 0.02,  # Темп технологического прогресса
        'delta': 0.05,  # Норма амортизации
        'A': 1.0  # Технологический параметр (производительность)
    }

    def steady_state(self, **params):
        """
        Вычисление стационарного состояния для AK-модели
        В AK-модели нет стационарного состояния в традиционном смысле,
        возвращаем теоретическое значение для визуализации
        """
        p = {**self.DEFAULT_PARAMS, **params}
        return p['s'] * p['A'] / (p['n'] + p['g'] + p['delta'])

    def simulate(self, k0, t_span=(0, 100), **params):
        p = {**self.DEFAULT_PARAMS, **params}

        def dk_dt(t, k):
            return p['s'] * p['A'] * k - (p['n'] + p['g'] + p['delta']) * k

        sol = solve_ivp(dk_dt, t_span, [k0], t_eval=np.linspace(*t_span, 500))
        return sol.t, sol.y[0]

    def plot_results(self, t, k, k_star, model_name):

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t, k, label='Траектория капитала')
        plt.title(f"AK-модель: {model_name}")
        plt.ylabel("Капитал (k)")
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(t, np.log(k), label='Лог капитала')
        plt.xlabel("Время (t)")
        plt.ylabel("ln(k)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()