import numpy as np
import matplotlib.pyplot as plt

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


def rusanov_scheme(q_n, h_n, R, C):  # сама схема
    # print(f"hn: {len(h_n)} qn: {len(q_n)} un: {len(u_n)}")

    f_1 = F_func(q_n, h_n)  # задаем q^2/h+gh^2/2

    # print(f"f1: {len(f_1)}")
    h_1 = (h_n[:-1] + h_n[1:]) / 2 - R * (q_n[1:] - q_n[:-1]) / 3  # n - 1
    q_1 = (q_n[:-1] + q_n[1:]) / 2 - R * (f_1[1:] - f_1[:-1]) / 3

    # print(f"h1: {len(h_1)} q1: {len(q_1)}")

    f_2 = F_func(q_1, h_1)

    h_2 = h_n[1:-1] - 2 / 3 * R * (q_1[1:] - q_1[:-1])
    q_2 = q_n[1:-1] - 2 / 3 * R * (f_2[1:] - f_2[:-1])  # n-2

    # print(f"h2: {len(h_2)} q2: {len(q_2)} f2: {len(f_2)}")

    w_h = h_n[4:] - h_n[3:-1] * 4 + h_n[2:-2] * 6 - h_n[1:-3] * 4 + h_n[:-4]
    w_q = q_n[4:] - q_n[3:-1] * 4 + q_n[2:-2] * 6 - q_n[1:-3] * 4 + q_n[:-4]

    f_3 = F_func(q_2, h_2)
    h_3 = h_n[2:-2] - R * (7 / 24 * (q_n[3:-1] - q_n[1:-3]) - (2 / 24) * (q_n[4:] - q_n[:-4])) - 3 / 8 * R * (
            q_2[2:] - q_2[:-2]) - w_h * C / 24
    # print(f"h3[]: {len(h_3)}")
    q_3 = q_n[2:-2] - R * ((7 / 24) * (f_1[3:-1] - f_1[1:-3]) - (2 / 24) * (f_1[4:] - f_1[:-4])) - 3 / 8 * R * (
            f_3[2:] - f_3[:-2]) - w_q * C / 24

    h_3 = np.concatenate([[h_n[0]], [h_n[1]], h_3, [h_n[-2]], [h_n[-1]]])
    q_3 = np.concatenate([[q_n[0]], [q_n[1]], q_3, [q_n[-2]], [q_n[-1]]])

    return h_3, q_3


def runge_error(u_h, u_h2, p):
    error = np.linalg.norm(u_h2[::2] - u_h, ord=2) / (2 ** p - 1)
    return error


def rusanov_scheme_for_step(C, X, x_start, x_end, T, CFL, n0, H0, H1):
    n = n0 + 4
    delta_x = X / n

    x0 = np.linspace(x_start, x_end,
                     n)
    u0 = step_u(n)
    h0 = step_h(x0, H0, H1)

    q0 = h0 * u0
    h_n, q_n, u_n = h0.copy(), q0.copy(), u0.copy()

    max_lambda = max(abs(u0) + (g * h0) ** 0.5)  # (max(lambda1, lambda2))
    delta_t = CFL * delta_x / max_lambda
    R = delta_t / delta_x

    # time_steps = int(T / delta_t) + 1

    step = 0
    time_steps = 100000000

    while step < time_steps:
        h_n, q_n = rusanov_scheme(q_n, h_n, R, C)
        u_n = q_n / h_n
        max_lambda = max(abs(u_n) + (g * h_n) ** 0.5)
        delta_t = CFL * delta_x / max_lambda
        # динамическое вычисление шага по времени
        time_steps = int(T / delta_t) + 1
        R = delta_t / delta_x
        # print(f"CFL: {delta_t * max_lambda / delta_x}, R: {R}")
        step += 1
        # print(step, time_steps)
        if step > 10000:
            break
    # f.close()
    return x0, h_n


def rusanov_scheme_periodical(a, b, C, X, x_start, x_end, T, CFL, n0):
    f = open("max_lambda.txt", "w")
    n = n0 + 4
    x0 = np.linspace(x_start, x_end, n)
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
        h_n, q_n = rusanov_scheme(q_n, h_n, R, C)
        u_n = q_n / h_n
        u_n[0] = u_n[n0 - 1]
        u_n[1] = u_n[n0]
        u_n[2] = u_n[n0 + 1]
        u_n[n0 + 2] = u_n[3]
        u_n[n0 + 3] = u_n[4]
        max_lambda = max(abs(u_n) + (g * h_n) ** 0.5)
        print(max_lambda, file=f)
        delta_t = CFL * delta_x / max_lambda
        # print(delta_t / delta_x * max_lambda)
        R = delta_t / delta_x
    f.close()
    return x0, h_n


