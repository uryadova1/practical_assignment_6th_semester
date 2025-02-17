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


def rusanov_scheme(un, fn, q, h, R, dots, C):
    u_j12 = np.zeros(dots, dtype=float)
    u_j12[0] = un[0]
    for i in range(1, dots - 1):
        u_j12[i] = (un[i] - un[i - 1]) / 2 - R * (fn[i + 1] - fn[i]) / 3
    # ______________________________________ t/3

    u2 = np.zeros(dots, dtype=float)
    u2[0] = un[0]
    f1 = F_func(q, u_j12, g, h)
    for i in range(1, dots - 1):
        u2[i] = un[i] - 2 / 3 * R * (f1[i + 1] - f1[i - 1])
    # ______________________________________ 2t/3

    u_n_plus_1 = np.zeros(dots, dtype=float)
    f2 = F_func(q, u2, g, h)
    w = np.zeros(dots, dtype=float)
    w[0] = un[0]
    w[1] = un[1]
    for i in range(2, dots - 2):
        w[i] = un[i + 2] - 4 * un[i + 1] + 6 * un[i] - 4 * un[i - 1] + un[i - 2]

    for i in range(2, dots - 2):
        u_n_plus_1 = un[i] - R * (7 / 24 * (fn[i + 1] - fn[i - 1]) - 1 / 12 * (fn[i + 2] - fn[i - 2])) - 3 * 8 * R * (
                    f2[i + 1] - f2[i - 1]) - C * w / 24
    #че то на границах надо сделать
    return u_n_plus_1

def runge_error(u_h, u_h2, p):
    error = np.linalg.norm(u_h2[::2] - u_h, ord=2) / (2**p - 1)
    return error

def compute_order(u_h, u_h2, u_exact):
    error_h = np.linalg.norm(u_h - u_exact, ord=2)
    error_h2 = np.linalg.norm(u_h2[::2] - u_exact, ord=2)
    p = np.log2(abs(error_h / error_h2))
    return abs(p)


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
    n = 1000
    delta_x = X / n

    x0 = np.linspace(x_start, x_end, n)
    # bx = np.array([h1 if xi < 0 else h0 for xi in x0])
    u = initial_condition_u(a, x0, X)  # скорость??
    u[0] = 0
    u[1] = 1
    h = initial_condition_h(b, g, u)  # глубина

    q = h * u  # расход
    F = F_func(q, u, g, h)
    U0 = np.vstack((h, q))  # функция, которая задает матрицу (h q)^T

    CFL = 0.5
    delta_t = CFL * (X / len(x0)) / max(u + (g * (h1 - h0)) ** (1 / 2))
    R = delta_t / delta_x
    print(delta_t)
# _____________________________________________________________________________
