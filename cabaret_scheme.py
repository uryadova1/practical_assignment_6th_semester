import numpy as np

g = 10


def initial_condition_u(x):
    return np.array([-0.5 if xi < 1 or xi > 2 else -2 for xi in x])


def F_func(q, h):
    global g
    return np.array(q ** 2 / h + g * h ** 2 / 2)  # приведено к виду q^2/h+gh^2/2, чтобы не использовать u


# f = u^2/2?


def lambda_func(q, h):
    return q / h #lambda = u


def min_and_max_m(u1, u2, u3):
    min_m = min(u1, u2, u3)
    max_m = max(u1, u2, u3)
    return min_m, max_m


def cabaret_scheme(h_n, q_n, tau, h):
    # первый шаг
    f_1 = F_func(q_n, h_n)
    h_1 = tau / 2 * ((q_n[:-1] - q_n[1:]) / h) + h_n  # h, q
    q_1 = tau / 2 * ((f_1[:-1] - f_1[1:]) / h) + q_n  # q, q^2/h+gh^2/2

    f_2 = F_func(q_1, h_1)

    # второй шаг
    lambda_minus_05 = lambda_func(q_1[:-1], h_1[:-1])
    lambda_plus_05 = lambda_func(q_1[1:], h_1[1:])
    if lambda_minus_05 > 0 and lambda_plus_05 >= 0:
        h_2 = 2 * (h_1[:-1]) - h_n[:-1]  # h
        q_2 = 2 * (q_1[:-1]) - q_n[:-1]  # q
        min_m, max_m = min_and_max_m()
    elif lambda_minus_05 <= 0 and lambda_plus_05 < 0:
        h_2 = 2 * (h_1[1:]) - h_n[1:]  # h
        q_2 = 2 * (q_1[1:]) - q_n[1:]  # q
        min_m, max_m = min_and_max_m()
