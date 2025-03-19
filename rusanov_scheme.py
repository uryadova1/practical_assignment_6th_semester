import numpy as np
import matplotlib.pyplot as plt


def initial_condition_u(a, x0, X):
    return a * np.sin(2 * np.pi * x0 / X + np.pi / 4)


def initial_condition_h(b, u):
    global g
    return (u + b) ** 2 / (4 * g)


def step(x):
    return [5 if x[i] <= 0 else 1 for i in range(len(x))] #задаем ступеньку, нужно чтобы потом график нарисовать


def F_func(q, h):
    global g
    # tmp = np.array(q ** 2 / h + g * h ** 2 / 2)
    # print(tmp, file=file)
    return np.array(q ** 2 / h + g * h ** 2 / 2)  #приведено к виду q^2/h+gh^2/2, чтобы не использовать u


def rusanov_scheme(q_n, h_n, u_n, R, C, n): #сама схема
    # print(f"hn: {len(h_n)} qn: {len(q_n)} un: {len(u_n)}")

    f_1 = F_func(q_n, h_n) #задаем q^2/h+gh^2/2

    # print(f"f1: {len(f_1)}")
    h_1 = np.zeros(n, dtype=float)
    q_1 = np.zeros(n, dtype=float)
    h_1[1:] = (h_n[:-1] + h_n[1:]) / 2 - R * (q_n[1:] - q_n[:-1]) / 3  # n - 1
    q_1[1:] = (q_n[:-1] - q_n[1:]) / 2 - R * (f_1[1:] - f_1[:-1])
    h_1[0], q_1[0] = h_n[0], q_n[0]  # восстанавливаем узел 0

    # print(f"h1: {len(h_1)} q1: {len(q_1)}")

    f_2 = F_func(q_1, h_1)
    h_2 = np.zeros(n, dtype=float)
    q_2 = np.zeros(n, dtype=float)
    # print(f"h2: {len(h_2)} q2: {len(q_2)} f2: {len(f_2)}")
    h_2[1:-1] = h_n[1:-1] - 2 / 3 * R * (q_1[1:-1] - q_1[2:])
    q_2[1:-1] = q_n[1:-1] - 2 / 3 * R * (f_1[2:] - f_1[:-2])  # n-2
    h_2[0], q_2[0] = h_1[0], q_1[0]  # восстанавливаем первый и последний узел
    h_2[-1], q_2[-1] = h_1[-1], q_1[-1]

    w = np.zeros(n - 4, dtype=float)
    for k in range(2, n - 4):
        w[k] = u_n[k + 2] - 4 * u_n[k + 1] + 6 * u_n[k] - 4 * u_n[k - 1] + u_n[k - 2] #только ради этого таскаем за собой в функцию вектор u

    h_3 = np.zeros(n, dtype=float)
    q_3 = np.zeros(n, dtype=float)
    h_3[2:-2] = h_n[2:-2] - R * (7 / 24 * (q_n[:-4] - q_n[1:-3]) - (2 / 24) * (q_n[:-4] - q_n[2:-2])) - 3 / 8 * R * (
            q_2[:-4] - q_2[1:-3]) - w * C / 24
    q_3[2:-2] = q_n[2:-2] - R * ((7 / 24) * (f_1[:-4] - f_1[1:-3]) - (2 / 24) * (f_1[:-4] - f_1[2:-2])) - 3 / 8 * R * (
            f_2[:-4] - f_2[1:-3]) - w * 0.102

    h_3[0], q_3[0] = h_2[0], q_2[0]  # восстанавливаем 2 первых и последних узла (надо 3?)
    h_3[-1], q_3[-1] = h_2[-1], q_2[-1]
    h_3[1], q_3[1] = h_2[1], q_2[1]
    h_3[-2], q_3[-2] = h_2[-2], q_2[-2]

    return h_3, q_3


def runge_error(u_h, u_h2, p):
    error = np.linalg.norm(u_h2[::2] - u_h, ord=2) / (2 ** p - 1)
    return error



def rusanov_scheme_for_different_time_limits(a, b, C, X, x_start, x_end, T, CFL, n0):
    n = n0  # int(input('n: '))
    delta_x = X / n

    x0 = np.linspace(x_start, x_end,
                     n)
    u0 = initial_condition_u(a, x0, X)
    h0 = initial_condition_h(b, u0)

    q0 = h0 * u0

    lambda1 = max(abs(u0 - (g * h0) ** 0.5))  # что то не то
    lambda2 = max(abs(u0 + (g * h0) ** 0.5))

    delta_t = CFL * delta_x / (max(lambda1, lambda2))
    R = delta_t / delta_x

    time_steps = int(T / delta_t) + 1

    h_n, q_n, u_n = h0.copy(), q0.copy(), u0.copy()

    for i in range(time_steps):
        h_n, q_n = rusanov_scheme(q_n, h_n, u_n, R, C, n)
        u_n = q_n / h_n

    # _____________________________________________________________________________
    n0 *= 2
    n2 = n0  # int(input('n: ')) + 4
    delta_x2 = X / n2

    x02 = np.linspace(x_start, x_end,
                      n2)  # длина n +4, возможно стоит сами границы еще сдвинуть типа x_start - 2 * delta_x, x_end - 2 * delta_x
    u02 = initial_condition_u(a, x02, X)
    h02 = initial_condition_h(b, u02)

    q02 = h02 * u02

    CFL = 0.5
    lambda12 = max(abs(u02 - (g * h02) ** 0.5))
    lambda22 = max(abs(u02 + (g * h02) ** 0.5))

    delta_t2 = CFL * delta_x2 / (max(lambda12, lambda22))
    R2 = delta_t2 / delta_x2

    time_steps2 = int(T / delta_t2) + 1

    h_n2, q_n2, u_n2 = h02.copy(), q02.copy(), u02.copy()

    for i in range(time_steps2):
        h_n2, q_n2 = rusanov_scheme(q_n2, h_n2, u_n2, R2, C, n2)
        u_n2 = q_n2 / h_n2

    # _____________________________________________________________________________
    p_h = 3
    rung_error = runge_error(h_n, h_n2, p_h)
    print(f"runge error: {rung_error}")

    x_step = np.linspace(x_start, x_end, 1000)
    step_graph = step(x_step)

    plt.figure(figsize=(10, 6))
    plt.plot(x_step, step_graph, label="step", linestyle="--")
    plt.plot(x0, h_n, label="n", linestyle="dotted")
    plt.plot(x02, h_n2, label="2n", linestyle="dotted")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    a = 2
    b = 10
    X = 10  #
    C = 0.5  # float(input())
    n0 = 15  # int(input())
    g = 10

    T1 = 0.5  #
    T2 = 1
    T3 = 2
    h1 = 5
    h0 = 1
    x_start = -5  # int(input())
    x_end = x_start + X
    CFL = 0.5
    rusanov_scheme_for_different_time_limits(a, b, C, X, x_start, x_end, T1, CFL,
                                             n0)  # чтобы можно было запустить для любого Т
