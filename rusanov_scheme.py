import numpy as np


def initial_condition_u(a, x0, X):
    return a * np.sin(2 * np.pi * x0 / X + np.pi / 4)


def initial_condition_h(b, g, u):
    return (u + b) ** 2 / (4 * g)


def initial_condition_b(x, before_step, after_step):
    if x < 0:
        return before_step
    return after_step


def F_func(q, u, g, h):
    s = np.array(q * u + g * h ** 2 / 2)
    F = np.vstack((q, s))
    return F


def rusanov_scheme(u, f, q, h, R, dots):
    u_j12_1 = np.zeros(dots, dtype=float)
    for i in range(2, dots - 1):
        u_j12_1[i] = (u[i] - u[i - 1]) / 2 - R * (f[i + 1] - f[i]) / 3  # что такое R?
    u_j_2 = np.zeros(dots, dtype=float)
    f = F_func(q, u_j12_1, g, h)
    for i in range(2, dots - 1):
        u_j_2[i] = u[i] - 2 / 3 * R * (f[i])


# ______________________________________________________ вот тут пока вопрос


if __name__ == "__main__":
    a = 2
    b = 10
    X = 10  #

    T1 = 0.5  #
    T2 = 1
    T3 = 2
    h1 = 5  # до ступеньки
    h0 = 1  # после ступеньки
    g = 10
    x_start = -5  # int(input())
    x_end = x_start + X

    x0 = np.linspace(x_start, x_end, 1000)
    # bx = np.array([h1 if xi < 0 else h0 for xi in x0])
    u = initial_condition_u(a, x0, X)  # скорость??
    u[0] = 0
    u[1] = 1
    h = initial_condition_h(b, g, u)  # глубина

    q = h * u  # расход
    F = F_func(q, u, g, h)
    U0 = np.vstack((h, q))  # функция, которая задает матрицу (h q)^T

    CFL = 0.5
    delta_t = CFL * (X / len(x0)) / max(u + (h1 - h0) ** (1 / 2))
    print(delta_t)
# _____________________________________________________________________________
