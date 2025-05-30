#models/RCK.py
import numpy as np

import matplotlib.pyplot as plt
from fontTools.varLib.interpolate_layout import interpolate_layout
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy import interpolate
from numpy import linalg as la

from mpl_toolkits.mplot3d import Axes3D

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
        self.c0 = None
        self.k0 = None
        p = {**self.DEFAULT_PARAMS, **params}

        self.alpha = p.get('alpha', self.DEFAULT_PARAMS['alpha'])
        self.delta = p.get('delta', self.DEFAULT_PARAMS['delta'])
        self.n = p.get('n', self.DEFAULT_PARAMS['n'])
        self.g = p.get('g', self.DEFAULT_PARAMS['g'])
        self.rho = p.get('rho', self.DEFAULT_PARAMS['rho'])
        self.theta = p.get('theta', self.DEFAULT_PARAMS['theta'])
        self.k_max = (1/(self.n+self.g+self.delta))**(1/(1-self.alpha))
        self.interpolate_f = None

    def find_k_star_solo_rck(self):
        k_star = (self.alpha / (self.delta + self.n + self.theta + self.rho * self.g)) ** (1 / (1 - self.alpha))
        c_star = self.f(k_star) - (self.n + self.g + self.delta) * k_star
        return [k_star, c_star]


    def f(self, k):
        return (k**self.alpha)


    def f_prime(self, k):
        return np.where(k > 0,self.alpha * k ** (self.alpha - 1), 0.01)

    def dk_dt(self, k, c):
        return self.f(k) - c - (self.delta + self.n + self.g) * k

    def dc_dt(self, k, c):
        return c/self.rho*(self.alpha*k**(self.alpha - 1) - self.theta -(self.n + self.delta) -self.rho*self.g)

    def find_c0(self, k0):
        return self.f(k0) - (self.delta + self.n + self.g) * k0

    def system(self, t, y):
        k, c = y
        if k <= 0:
            k = 1e-6
        if c <= 0:
            c = 1e-6
        dkdt = k ** self.alpha - c - (self.n + self.delta) * k
        dcdt = c / self.rho * (
                    self.alpha * k ** (self.alpha - 1) - self.theta - self.rho * self.g - self.n - self.delta)
        return [dkdt, dcdt]


    def dc_dk(self, c, k):
        return self.dc_dt(k,c)/self.dk_dt(k,c)
    def steady_state(self, **params):
        #точка
        return None
    def Jacobian(self, k, c):
        J = np.array([[1 / self.rho * (self.alpha * k ** (self.alpha - 1) -self.theta - self.n - self.delta - self.g),c / self.rho * \
                       self.alpha * (self.alpha - 1) * k ** (self.alpha - 2)],
                      [-1,
                       self.alpha * k ** (self.alpha - 1) - \
                       (self.g + self.n + self.delta)]])

        return (J)
    def saddle_path_slope(self):
        k_star, c_star = self.find_k_star_solo_rck()
        J = self.Jacobian(k_star, c_star)
        w, v = la.eig(J)
        min_eig = np.argsort(w)[0]

        slope = v[0, min_eig] / v[1, min_eig]

        return (slope)

    def prepare_interpolate_function(self, eps = 10**(-8), npoints = 400):
        k_star, c_star = self.find_k_star_solo_rck()
        if(k_star-eps<0.12):
            small = 1e-6
            k_below = np.linspace(small, k_star - eps, npoints)
        else:
            k_below = np.linspace(k_star - eps, 0.0001, npoints)
        k_above = np.linspace(k_star + eps, self.k_max, npoints)
        k = np.concatenate((k_below, k_above)).flatten()


        # решение ОДУ
        c_below = odeint(self.dc_dk,
                         c_star - eps * self.saddle_path_slope(), k_below)
        c_above = odeint(self.dc_dk,
                         c_star + eps * self.saddle_path_slope(), k_above)

        c = np.concatenate((c_below, c_above)).flatten()
        k = np.concatenate((k_below, k_above)).flatten()

        mask = np.isfinite(c) & np.isreal(c)
        c = c[mask]
        k = k[mask]
        c = np.real(c)  # float64

        self.interpolate_f = interpolate.interp1d(k, c, kind='linear', fill_value='extrapolate')

    def simulate(self, k0, t_span=(0, 150), **params):
        # Обновляем параметры при симуляции, если переданы
        p = {**self.DEFAULT_PARAMS, **params}
        self.__init__(**p)
        self.k0 = k0
        self.c0 = self.find_c0(k0)

    def phase_diagram(self, npoints=100, arrows=True, n_arrows=16, labels=True, legend=True):
        self.prepare_interpolate_function()
        print(self.find_k_star_solo_rck())
        k_lin = np.linspace(0.01, self.k_max, npoints)
        plt.figure(figsize=(10, 6))

        # Траектория \dot{k} = 0 (нулевой рост капитала)
        c0_vals = self.find_c0(k_lin)
        plt.plot(k_lin, c0_vals, label='$\\dot{k}=0$ траектория (капитал не изменяется)', color='blue')

        # Стационарное состояние
        k_star, c_star = self.find_k_star_solo_rck()
        plt.axvline(k_star, color='red', linestyle='--', label='$\\dot{c}=0$ траектория (потребление не изменяется)')

        # Седловая траектория
        try:
           # print("ВАЖНООО", k_lin, self.interpolate_f(k_lin))
            plt.plot(k_lin, self.interpolate_f(k_lin), label='Седловая траектория', color='green')
        except Exception as e:
            print("Ошибка при построении седловой траектории:", e)

        plt.plot(k_star, c_star, '*r', label='Стационарное состояние')

        # Векторное поле
        if arrows:
            x = np.linspace(k_lin[0], k_lin[-1], n_arrows)
            y = np.linspace(min(self.interpolate_f(k_lin[0]), min(c0_vals)), max(self.interpolate_f(k_lin[-1]), max(c0_vals)), n_arrows)
            X, Y = np.meshgrid(x, y)

            # Вычисляем производные
            DK = self.dk_dt(X, Y)
            DC = self.dc_dt(X, Y)

            M = np.hypot(DK, DC)
            M[M == 0] = 1.0
            DK /= M
            DC /= M

            plt.quiver(X, Y, DK, DC, M, pivot='mid', alpha=0.3)

        # Подписи
        if labels:
            plt.title('Фазовый портрет модели Рамсея-Купманса-Касса')
            plt.xlabel('Капитал на душу населения (k)')
            plt.ylabel('Потребление на душу населения (c)')

        if legend:
            plt.legend()

        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_results(self, t_values, k_values, k_star, model_name):
       # self.make_3d_graph()
        self.phase_diagram(npoints=120, arrows= True, n_arrows = 12)

    def make_3d_graph(self):


        t_span = (0, 150)
        t_eval = np.linspace(t_span[0], t_span[1], 500)
        print(self.k0,self.c0)
        sol = solve_ivp(self.system, t_span, [self.k0, self.c0], t_eval=t_eval, method='BDF')
        k_st, c_st = self.find_k_star_solo_rck()

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(sol.t, sol.y[0], sol.y[1], label='Траектория изменения капиталовооружённости и потребления на душу населения')
        ax.plot(sol.t, [k_st]*len(sol.t), [c_st]*len(sol.t), '--', label='Стационарное состояние')
        ax.set_xlabel('Время (t)')
        ax.set_ylabel('Капитал на душу населения (k)')
        ax.set_zlabel('Потребление на душу населения (c)')
        ax.set_title('Траектория капитала и потребления в модели РКК')
        ax.legend()

        plt.ion()
        plt.show()
        plt.pause(0.1)
