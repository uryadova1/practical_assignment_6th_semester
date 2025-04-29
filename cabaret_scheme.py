import numpy as np
import matplotlib.pyplot as plt

g = 10


def initial_condition_u(x):
    return np.array([0.5 if xi < 1 or xi > 2 else 2 for xi in x])


def Fu(u):
    global g
    return u ** 2 / 2


# f = u^2/2?

def F(m, M, u):
    n = len(u)
    new_arr = list()
    for i in range(n):
        if m[i] <= u[i] <= M[i]:
            new_arr.append(u[i])
        elif u[i] < m[i]:
            new_arr.append(m[i])
        else:
            new_arr.append(M[i])
    return np.array(new_arr)



def first_and_third_step(u, f, tau, h):
    return u - tau / 2 * (f[1:] - f[:-1]) / h


def find_min_m(u1, u2):
    n = len(u1)
    m = list()
    for i in range(n):
        m.append(min(u1[i], u2[i], u2[i + 1]))
        print(m[-1])
    return np.array(m)

def find_max_m(u1, u2):
    n = len(u1)
    m = list()
    for i in range(n):
        m.append(max(u1[i], u2[i], u2[i + 1]))
    return np.array(m)



def second_step(u_c, u_s, uU):
    u_n1, m, M = 0, 0, 0
    # if all(i > 0 for i in u_c[:-1]) and all(i >= 0 for i in u_c[1:]):
    u_n1 = 2 * u_c - u_s[:-1]
    m = find_min_m(uU, u_s)
    M = find_max_m(uU, u_s)
    # elif all(i <= 0 for i in u_con[:-1]) and all(i < 0 for i in u_con[1:]):
    u_n1 = F(m, M, u_n1)
    u_n1 = np.concatenate([[u_s[0]], u_n1])
    # print(u_s[0], u_n1[0])
    return u_n1


def cabaret_scheme(u_n_con, u_n_str, tau, h):
    f_n = Fu(u_n_str)
    u_n12_j_12 = first_and_third_step(u_n_con, f_n, tau, h)
    u_n1_j = second_step(u_n12_j_12, u_n_str, u_n_con)
    f_n1_j1 = Fu(u_n1_j)
    u_n1_j12 = first_and_third_step(u_n12_j_12, f_n1_j1, tau, h)
    return u_n1_j12, u_n1_j

def graphics(x1, u1):
    plt.figure(figsize=(10, 6))
    plt.plot(x1, u1,"k o", markersize=3)
    # plt.plot(x2, u2, linestyle="o")
    plt.xlabel("x")
    plt.ylabel("h(x, t)")
    # plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    x0 = 0
    X = 5
    x1 = x0 + X
    h = 0.1
    N = int((x1 - x0)  / h) # ячейки
    tau = 0.0125 * 2
    x_str = np.linspace(x0, x1, N + 1)
    x_con = np.linspace(x0 + h / 2, x1 - h / 2, N)
    u_con0 = initial_condition_u(x_con)
    u_str0 = initial_condition_u(x_str)
    time_steps = 64

    print(u_str0, u_con0)

    print(f"fdt/dx {tau/h}")

    u_con, u_str = u_con0.copy(), u_str0.copy()
    for i in range(time_steps):
        u_con, u_str = cabaret_scheme(u_con, u_str, tau, h)
        if i == 31:
            # print(u_str[0], u_str[1])
            graphics(x_con, u_con)
