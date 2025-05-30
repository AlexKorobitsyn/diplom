# models/base_model.py
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class GrowthModel(ABC):

    @staticmethod
    def production(k, alpha):
        return k ** alpha

    @abstractmethod
    def steady_state(self, **params):
        pass

    @abstractmethod
    def simulate(self, k0, t_span, **params):
        #Симуляция траектории
        pass

    @staticmethod
    def plot_results(t_values, k_values, k_star, model_name):
        plt.figure(figsize=(10, 6))

        # Настройки стрелок
        arrow_kwargs = {
            'length_includes_head': True,
            'head_width': 0.5,
            'head_length': 1.0,
            'fc': 'k',
            'ec': 'k'
        }

        for i, (t, k) in enumerate(zip(t_values, k_values)):
            line, = plt.plot(t, k, label=f"Траектория k₀ = {k[0]:.2f}")

            if len(t) > 1:
                # Стрелка в начале
                dx = t[1] - t[0]
                dy = k[1] - k[0]
                plt.arrow(t[0], k[0], dx * 0.8, dy * 0.8, **arrow_kwargs)

                # Стрелка в конце
                dx = t[-1] - t[-2]
                dy = k[-1] - k[-2]
                plt.arrow(t[-2], k[-2], dx * 0.8, dy * 0.8, **arrow_kwargs)

        plt.axhline(k_star, color='red', linestyle='--', label=f"k* ≈ {k_star:.3f}")

        plt.title(f"Траектории капитала в модели {model_name}")
        plt.xlabel("Время (t)")
        plt.ylabel("Капитал на работника k(t)")
        plt.legend()
        plt.grid(True)
        plt.show()