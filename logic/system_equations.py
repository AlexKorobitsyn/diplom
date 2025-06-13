import numpy as np


def system_equations(t, state, params):
    """
    Правая часть системы уравнений (3.14) из статьи.
    state: [x1, x2, x3, z1, z2, z3]
    params: словарь с параметрами модели
    """
    x1, x2, x3, z1, z2, z3 = state

    # Распаковка параметров
    alpha = params['alpha']
    beta = params['beta']
    mu = params['mu']
    M0 = params['M0']
    rho = params['rho']
    p0 = params['p0']
    a = params['a']

    # Управление (здесь можно заменить на закон управления)
    u = params['u_func'](t, state)
    v = params['v_func'](t, state)

    # Вспомогательные функции A и B (объявить отдельно при необходимости)
    def A(x1, x3):
        return (1 / x1) - (alpha * beta * x3) / (1 - alpha)

    def B(z2, z3):
        return z2 + alpha * z3

    Ax = A(x1, x3)
    Bz = B(z2, z3)

    # Уравнения для x
    dx1 = (beta / alpha) * x1 - (1 / M0) * x2
    dx2 = x2 * (Ax + (1 / Bz) - (1 / z2) + x3 - ((alpha * beta - beta) / (alpha * (1 - alpha))) - mu)
    dx3 = alpha * x3 * (Ax + (1 / Bz) - (1 / z2) - (beta - alpha * beta) / (alpha * (1 - alpha)))

    # Уравнения для z
    dz1 = rho * z1 - (1 / M0) * (x2 / x1) * z1 - Ax * Bz
    dz2 = rho * z2 + (1 / M0) * (x1 / x2) * z1 - 1
    dz3 = rho * z3 - x3 * z2 + ((1 - alpha) / alpha) * Ax * Bz + (1 / alpha)

    return [dx1, dx2, dx3, dz1, dz2, dz3]
