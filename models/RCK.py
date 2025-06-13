import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy import interpolate
from numpy import linalg as la
from models.base_model import GrowthModel


class RCKModel(GrowthModel):
    DEFAULT_PARAMS = {
        'rho': 0.02,
        'alpha': 0.33,
        'delta': 0.05,
        'theta': 2.0,
        'g': 0.02,
        'n': 0.01
    }

    def __init__(self, **params):
        self.rho = self.DEFAULT_PARAMS['rho']
        self.alpha = self.DEFAULT_PARAMS['alpha']
        self.delta = self.DEFAULT_PARAMS['delta']
        self.theta = self.DEFAULT_PARAMS['theta']
        self.g = self.DEFAULT_PARAMS['g']
        self.n = self.DEFAULT_PARAMS['n']

        self.trajectories = []
        self.update_params(**params)

    def update_params(self, **params):
        p = {**self.DEFAULT_PARAMS, **params}
        self.alpha = p.get('alpha', self.DEFAULT_PARAMS['alpha'])
        self.delta = p.get('delta', self.DEFAULT_PARAMS['delta'])
        self.n = p.get('n', self.DEFAULT_PARAMS['n'])
        self.g = p.get('g', self.DEFAULT_PARAMS['g'])
        self.rho = p.get('rho', self.DEFAULT_PARAMS['rho'])
        self.theta = p.get('theta', self.DEFAULT_PARAMS['theta'])

        try:
            denominator = (self.delta + self.n + self.theta + self.rho * self.g)
            if denominator > 0:
                self.k_star = (self.alpha / denominator) ** (1 / (1 - self.alpha))
                self.c_star = self.f(self.k_star) - (self.n + self.g + self.delta) * self.k_star
            else:
                self.k_star = 1.0
                self.c_star = 0.5
        except:
            self.k_star = 1.0
            self.c_star = 0.5

        try:
            self.k_max = (1 / (self.n + self.g + self.delta)) ** (1 / (1 - self.alpha))
            if self.k_max < self.k_star * 2:
                self.k_max = self.k_star * 2
        except:
            self.k_max = self.k_star * 2 if self.k_star > 0 else 10.0

    def find_steady_state(self):
        """Находим стационарное состояние с защитой"""
        try:
            denominator = (self.delta + self.n + self.theta + self.rho * self.g)
            if denominator <= 0:
                return 1.0, 0.5

            k_star = (self.alpha / denominator) ** (1 / (1 - self.alpha))
            c_star = self.f(k_star) - (self.n + self.g + self.delta) * k_star
            return k_star, c_star
        except:
            return 1.0, 0.5

    def steady_state(self, **params):
        """Реализация абстрактного метода"""
        if params:
            self.update_params(**params)
        return self.k_star

    def f(self, k):
        """Производственная функция с защитой"""
        try:
            if k <= 1e-10:
                return 0.0
            return k ** self.alpha
        except:
            return 0.0

    def dk_dt(self, k, c):
        try:
            return self.f(k) - c - (self.delta + self.n + self.g) * k
        except:
            return 0.0

    def dc_dt(self, k, c):
        """Функция изменения потребления с защитой"""
        try:
            if k <= 1e-10:
                return 0.0

            mpk = self.alpha * k ** (self.alpha - 1)
            return c / self.rho * (mpk - self.theta - (self.n + self.delta) - self.rho * self.g)
        except:
            return 0.0

    def dc_dk(self, c, k):
        """Производная dc/dk для ОДУ седловой траектории"""
        try:
            dk = self.dk_dt(k, c)
            if abs(dk) < 1e-10:
                return 0.0
            return self.dc_dt(k, c) / dk
        except:
            return 0.0

    def Jacobian(self, k, c):
        """Вычисление якобиана системы с защитой"""
        try:
            if k <= 1e-10:
                k = 1e-10

            df_dk = self.alpha * k ** (self.alpha - 1) - (self.n + self.delta + self.g)
            df_dc = -1.0

            dg_dk = (c / self.rho) * self.alpha * (self.alpha - 1) * k ** (self.alpha - 2)
            dg_dc = (1.0 / self.rho) * (
                        self.alpha * k ** (self.alpha - 1) - self.theta - self.n - self.delta - self.rho * self.g)

            return np.array([[df_dk, df_dc], [dg_dk, dg_dc]])
        except:
            return np.array([[0, -1], [0, 0]])

    def saddle_path_slope(self):
        """Вычисление наклона седловой траектории с защитой"""
        try:
            J = self.Jacobian(self.k_star, self.c_star)

            w, v = la.eig(J)

            real_parts = np.real(w)
            neg_indices = np.where(real_parts < 0)[0]

            if len(neg_indices) == 0:
                index = np.argmin(real_parts)
            else:
                index = neg_indices[0]

            eigenvector = v[:, index]

            if abs(eigenvector[0]) < 1e-10:
                return 0.0
            return np.real(eigenvector[1] / eigenvector[0])
        except:
            return -0.1

    def system(self, t, y):
        """Система дифференциальных уравнений модели с защитой"""
        try:
            k, c = y
            if k < 1e-10:
                k = 1e-10
                dkdt = 0
            else:
                dkdt = self.dk_dt(k, c)

            dcdt = self.dc_dt(k, c)
            return [dkdt, dcdt]
        except:
            return [0, 0]

    def prepare_interpolate_function(self, eps=1e-5, npoints=400):
        """Расчет седловой траектории с улучшенной обработкой ошибок"""
        try:
            if self.k_star <= eps or self.k_max <= self.k_star + eps:
                raise ValueError("Invalid steady state values")

            k_below = np.linspace(1e-5, self.k_star - eps, npoints // 2)
            k_above = np.linspace(self.k_star + eps, self.k_max, npoints // 2)
            k = np.concatenate((k_below, k_above))

            slope = self.saddle_path_slope()

            c_below = odeint(
                lambda c, k: self.dc_dk(c, k),
                self.c_star - eps * slope,
                k_below[::-1],
                atol=1e-4, rtol=1e-2
            )[:, 0][::-1]

            c_above = odeint(
                lambda c, k: self.dc_dk(c, k),
                self.c_star + eps * slope,
                k_above,
                atol=1e-4, rtol=1e-2
            )[:, 0]

            c = np.concatenate((c_below, c_above))
            mask = np.isfinite(c) & (c > 0)

            if np.sum(mask) < 10:
                raise ValueError("Not enough valid points")

            self.interpolate_f = interpolate.interp1d(
                k[mask], c[mask], kind='linear',
                fill_value="extrapolate", bounds_error=False
            )
        except:
            k_lin = np.linspace(0.01, self.k_max, npoints)
            c_lin = np.linspace(0.01, self.c_star * 1.5, npoints)
            self.interpolate_f = interpolate.interp1d(
                k_lin, c_lin, kind='linear',
                fill_value="extrapolate", bounds_error=False
            )

    def simulate(self, k0, t_span=(0, 50), **params):
        """Симуляция траектории с защитой от бесконечного выполнения"""
        self.update_params(**params)

        try:
            self.prepare_interpolate_function()
            c0 = float(self.interpolate_f(k0))
        except:
            c0 = self.c_star * 0.8 if hasattr(self, 'c_star') else 0.5

        try:
            sol = solve_ivp(
                self.system, t_span, [k0, c0],
                t_eval=np.linspace(*t_span, 100),
                method='RK45',
                rtol=1e-3,
                atol=1e-5,
                max_step=0.5
            )

            trajectory = {
                't': sol.t,
                'k': sol.y[0],
                'c': sol.y[1],
                'k0': k0,
                'c0': c0
            }
            self.trajectories.append(trajectory)
            return trajectory
        except Exception as e:
            print(f"Simulation error: {e}")
            return {
                't': np.array([0, t_span[1]]),
                'k': np.array([k0, k0]),
                'c': np.array([c0, c0]),
                'k0': k0,
                'c0': c0
            }

    def phase_diagram(self, npoints=100, arrows=True, n_arrows=20, labels=True, legend=True):
        """Профессиональный фазовый портрет с улучшенным векторным полем"""
        plt.figure(figsize=(10, 8))
        k_min, k_max = 0, max(self.k_star*2, 1.2)
        c_min, c_max = 0, max(2.0, self.c_star*1.5)

        k_vals = np.linspace(0.01, k_max, npoints)
        c_dk0 = [self.f(k) - (self.n + self.g + self.delta) * k for k in k_vals]
        plt.plot(k_vals, c_dk0, 'b-', linewidth=2.5, label=r'$\dot{k}=0$')

        plt.axvline(self.k_star, color='r', linestyle='--', linewidth=2, label=r'$\dot{c}=0$')

        try:
            saddle_c = self.interpolate_f(k_vals)
            plt.plot(k_vals, saddle_c, 'g-', linewidth=2.5, alpha=0.9, label='Седловая траектория')
        except:
            pass

        plt.plot(self.k_star, self.c_star, 'o', markersize=10,
                 markerfacecolor='gold', markeredgecolor='black',
                 markeredgewidth=1.5, label='Стационарное состояние', zorder=10)

        if arrows:
            k_grid = np.linspace(0.1, k_max, n_arrows)
            c_grid = np.linspace(0.1, c_max, n_arrows)
            K, C = np.meshgrid(k_grid, c_grid)

            dk = np.zeros_like(K)
            dc = np.zeros_like(C)

            for i in range(K.shape[0]):
                for j in range(K.shape[1]):
                    k_val = K[i, j]
                    c_val = C[i, j]
                    dk[i, j] = self.dk_dt(k_val, c_val)
                    dc[i, j] = self.dc_dt(k_val, c_val)

            norm = np.sqrt(dk ** 2 + dc ** 2)
            scale_factor = 0.8 * min(
                (k_max - k_min) / n_arrows,
                (c_max - c_min) / n_arrows
            )

            dk_norm = np.zeros_like(dk)
            dc_norm = np.zeros_like(dc)

            non_zero_mask = norm > 0
            dk_norm[non_zero_mask] = dk[non_zero_mask] / norm[non_zero_mask] * scale_factor
            dc_norm[non_zero_mask] = dc[non_zero_mask] / norm[non_zero_mask] * scale_factor

            color_norm = np.log(norm + 1)

            plt.quiver(
                K, C,
                dk_norm, dc_norm,
                color_norm,
                angles='xy', scale_units='xy', scale=1,
                width=0.004, headwidth=4, headlength=5,
                headaxislength=4.5, cmap='viridis', alpha=0.8
            )
            plt.colorbar(label='Скорость изменения', shrink=0.7)

        #
        # for i, traj in enumerate(self.trajectories):
        #     try:
        #         k = np.clip(traj['k'], k_min, k_max)
        #         c = np.clip(traj['c'], c_min, c_max)
        #
        #
        #         color = 'purple' if i % 2 == 0 else 'darkorange'
        #         plt.plot(k, c, color=color, linewidth=2.5, alpha=0.9)
        #
        #         if len(k) > 10:
        #             plt.arrow(k[0], c[0],
        #                       k[5] - k[0], c[5] - c[0],
        #                       shape='full', color=color,
        #                       length_includes_head=True,
        #                       head_width=0.04, head_length=0.08,
        #                       alpha=0.9, linewidth=0.5, zorder=5)
        #
        #             if len(k) > 20:
        #                 plt.arrow(k[-6], c[-6],
        #                           k[-1] - k[-6], c[-1] - c[-6],
        #                           shape='full', color=color,
        #                           length_includes_head=True,
        #                           head_width=0.04, head_length=0.08,
        #                           alpha=0.9, linewidth=0.5, zorder=5)
        #     except:
        #         continue

        plt.xlim(k_min, k_max)
        plt.ylim(c_min, c_max)

        title = f'Фазовый портрет модели Рамсея-Купманса-Касса'
        plt.title(title, fontsize=16, pad=15)
        plt.xlabel('Капиталовооружённость ($k$)', fontsize=14, labelpad=10)
        plt.ylabel('Потребление ($c$) на душу населения', fontsize=14, labelpad=10)

        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)

        if legend:
            leg = plt.legend(loc='upper right', fontsize=12, frameon=True)
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_alpha(0.95)
            leg.get_frame().set_edgecolor('gray')
            leg.get_frame().set_linewidth(0.8)

        plt.tight_layout()
        plt.savefig('rck_phase_diagram.png', dpi=300, bbox_inches='tight')
        plt.show()