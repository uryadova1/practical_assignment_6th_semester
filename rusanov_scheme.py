import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

g = 10


def periodical_u(a, x0, X):
    return a * np.sin(2 * np.pi * x0 / X + np.pi / 4)


def periodical_h(b, u):
    return (u + b) ** 2 / (4 * g)


def step_u(n):
    return np.zeros(n, dtype=float)


def step_h(x, h0, h1):
    return np.array([h1 if x[i] <= 0 else h0 for i in range(len(x))])


def F_func(q, h):
    global g
    return np.array(q ** 2 / h + g * h ** 2 / 2)  # приведено к виду q^2/h+gh^2/2, чтобы не использовать u


def an_sol_problem_1(x, h1, h0, q1, q0, T):
    global g
    h2 = 2.54  # для перепада глубин 5_1
    # h2 = 3.95 # для перепада глубин 10_1
    # h2 = 6.2 # для перепада глубин 20_1
    v0 = q0 / h0
    c1 = np.sqrt(g * h1)
    b = np.sqrt(g * (h0 + h2) / (2 * h2 * h0))
    v2 = b * (h2 - h0)
    q2 = h2 * v2
    D = q2 / (h2 - h0)
    c2 = np.sqrt(g * h2)
    ve = -c1
    vc = v2 - c2

    xe = ve * (T)
    xc = vc * (T)
    xd = D * (T)
    h3 = lambda x: 1 / (9 * g) * (x / T - (v2 + 2 * c2)) ** 2
    q3 = lambda x: 1 / (9 * g) * (x / T - (v2 + 2 * c2)) ** 2 * 1 / 3 * (2 * x / T + (v2 + 2 * c2))
    deep = np.piecewise(x, [x < xe, x >= xe, x >= xc, x >= xd], [h1, h3, h2, h0])
    flow = np.piecewise(x, [x < xe, x >= xe, x >= xc, x >= xd], [q1, q3, q2, q0])

    return deep, flow


def rusanov_scheme_p(q_n, h_n, R, C, n):
    # вот здесь периодические условия
    h_n = np.concatenate([[h_n[n - 3]], [h_n[n - 2]], h_n, [h_n[1]], [h_n[2]]])
    q_n = np.concatenate([[q_n[n - 3]], [q_n[n - 2]], q_n, [q_n[1]], [q_n[2]]])

    f_1 = F_func(q_n, h_n)  # задаем q^2/h+gh^2/2

    h_1 = (h_n[:-1] + h_n[1:]) / 2 - R * (q_n[1:] - q_n[:-1]) / 3  # n - 1
    q_1 = (q_n[:-1] + q_n[1:]) / 2 - R * (f_1[1:] - f_1[:-1]) / 3

    f_2 = F_func(q_1, h_1)

    h_2 = h_n[1:-1] - 2 / 3 * R * (q_1[1:] - q_1[:-1])
    q_2 = q_n[1:-1] - 2 / 3 * R * (f_2[1:] - f_2[:-1])  # n-2

    w_h = h_n[4:] - h_n[3:-1] * 4 + h_n[2:-2] * 6 - h_n[1:-3] * 4 + h_n[:-4]
    w_q = q_n[4:] - q_n[3:-1] * 4 + q_n[2:-2] * 6 - q_n[1:-3] * 4 + q_n[:-4]

    f_3 = F_func(q_2, h_2)
    h_3 = h_n[2:-2] - R * (7 / 24 * (q_n[3:-1] - q_n[1:-3]) - (2 / 24) * (q_n[4:] - q_n[:-4])) - 3 / 8 * R * (
            q_2[2:] - q_2[:-2]) - w_h * C / 24
    q_3 = q_n[2:-2] - R * ((7 / 24) * (f_1[3:-1] - f_1[1:-3]) - (2 / 24) * (f_1[4:] - f_1[:-4])) - 3 / 8 * R * (
            f_3[2:] - f_3[:-2]) - w_q * C / 24

    return h_3, q_3


def rusanov_scheme_s(q_n, h_n, R, C):  # сама схема
    f_1 = F_func(q_n, h_n)  # задаем q^2/h+gh^2/2

    h_1 = (h_n[:-1] + h_n[1:]) / 2 - R * (q_n[1:] - q_n[:-1]) / 3  # n - 1
    q_1 = (q_n[:-1] + q_n[1:]) / 2 - R * (f_1[1:] - f_1[:-1]) / 3

    f_2 = F_func(q_1, h_1)

    h_2 = h_n[1:-1] - 2 / 3 * R * (q_1[1:] - q_1[:-1])
    q_2 = q_n[1:-1] - 2 / 3 * R * (f_2[1:] - f_2[:-1])  # n-2

    w_h = h_n[4:] - h_n[3:-1] * 4 + h_n[2:-2] * 6 - h_n[1:-3] * 4 + h_n[:-4]
    w_q = q_n[4:] - q_n[3:-1] * 4 + q_n[2:-2] * 6 - q_n[1:-3] * 4 + q_n[:-4]

    f_3 = F_func(q_2, h_2)
    h_3 = h_n[2:-2] - R * (7 / 24 * (q_n[3:-1] - q_n[1:-3]) - (2 / 24) * (q_n[4:] - q_n[:-4])) - 3 / 8 * R * (
            q_2[2:] - q_2[:-2]) - w_h * C / 24
    q_3 = q_n[2:-2] - R * ((7 / 24) * (f_1[3:-1] - f_1[1:-3]) - (2 / 24) * (f_1[4:] - f_1[:-4])) - 3 / 8 * R * (
            f_3[2:] - f_3[:-2]) - w_q * C / 24

    h_3 = np.concatenate([[h_n[0]], [h_n[1]], h_3, [h_n[-2]], [h_n[-1]]])
    q_3 = np.concatenate([[q_n[0]], [q_n[1]], q_3, [q_n[-2]], [q_n[-1]]])

    return h_3, q_3


def calculate_p(h1, h2, h3, q1, q2, q3):
    # norm1, norm2 = np.zeros(len(h1)), np.zeros(len(h1))
    print(len(q1), len(q2), len(q3))
    # for i, val in enumerate(h1, 0):
    #     norm1[i] = ((val - h2[2 * i]) ** 2 + (q1[i] - q2[2 * i]) ** 2) ** (1 / 2)
    #     norm2[i] = ((h2[2 * i] - h3[4 * i]) ** 2 + (q2[i * 2] - q3[4 * i]) ** 2) ** (1 / 2)

    h2 = np.array(h2[::2])
    h3 = np.array(h3[::4])

    q2 = np.array(q2[::2])
    q3 = np.array(q3[::4])

    norm1 = ((h1 - h2) ** 2 + (q1 - q2) ** 2) ** (1 / 2)
    norm2 = ((h2 - h3) ** 2 + (q2 - q3) ** 2) ** (1 / 2)
    p = np.log2(abs(norm1 / norm2))
    # p = np.log(norm1/norm2)/np.log(2)
    return p


def runge_error(u_h, u_h2, p):
    error = np.linalg.norm(u_h2[::2] - u_h, ord=2) / (2 ** p - 1)
    return error


def rusanov_scheme_for_step(C, X, x_start, x_end, T, CFL, n0, H0, H1):
    n = n0 + 4
    delta_x = X / n

    x0 = np.linspace(x_start, x_end,
                     n)
    u0 = np.zeros(n)  # step_u(n)
    h0 = step_h(x0, H0, H1)

    q0 = h0 * u0
    h_n, q_n, u_n = h0.copy(), q0.copy(), u0.copy()

    max_lambda = max(abs(u0) + (g * h0) ** 0.5)  # (max(lambda1, lambda2))
    print(f"max lambda: {max_lambda}")
    delta_t = CFL * delta_x / max_lambda
    R = delta_t / delta_x

    # time_steps = int(T / delta_t) + 1

    step = 0
    time_steps = 100000000

    while step < time_steps:
        h_n, q_n = rusanov_scheme_s(q_n, h_n, R, C)
        u_n = q_n / h_n
        max_lambda = max(abs(u_n) + (g * h_n) ** 0.5)
        delta_t = CFL * delta_x / max_lambda
        # динамическое вычисление шага по времени
        time_steps = int(T / delta_t) + 1
        R = delta_t / delta_x
        # print(f"CFL: {delta_t * max_lambda / delta_x}, R: {R}")
        step += 1
        print(step, time_steps, delta_t)
        if step > 500:
            break
    return x0, h_n


def rusanov_scheme_periodical(a, b, C, X, x_start, x_end, T, CFL, n0):
    file = open("max_lambda.txt", "w")
    n = n0 + 4
    x0 = np.linspace(x_start, x_end, n0)
    u0 = periodical_u(a, x0, X)
    h0 = periodical_h(b, u0)
    q0 = h0 * u0
    h_n, q_n, u_n = h0.copy(), q0.copy(), u0.copy()

    max_lambda = max(abs(u0) + (g * h0) ** 0.5)
    delta_x = X / n
    delta_t = CFL * delta_x / max_lambda
    R = delta_t / delta_x
    time_steps = int(T / delta_t) + 1
    for i in range(time_steps):
        h_n, q_n = rusanov_scheme_p(q_n, h_n, R, C, n0)
        u_n = q_n / h_n
        max_lambda = max(abs(u_n) + (g * h_n) ** 0.5)
        print(max_lambda, file=file)
        delta_t = CFL * delta_x / max_lambda
        # print(delta_t / delta_x * max_lambda)
        R = delta_t / delta_x
    file.close()
    return x0, h_n


def rusanov_scheme_third_task(a, b, C, X, x_start, x_end, delta_t, delta_x, n0, time_steps):
    global g
    n = n0
    x0 = np.linspace(x_start, x_end, n)
    u0 = periodical_u(a, x0, X)
    h0 = periodical_h(b, u0)
    q0 = h0 * u0
    h_n, q_n, u_n = h0.copy(), q0.copy(), u0.copy()
    R = delta_t / delta_x
    print(time_steps)
    for i in range(time_steps):
        h_n, q_n = rusanov_scheme_p(q_n, h_n, R, C, n)
        # u_n = q_n / h_n
    return x0, h_n, q_n
