import numpy as np

from rusanov_scheme import rusanov_scheme_for_step, step_h, rusanov_scheme_periodical, rusanov_scheme_third_task, \
    an_sol_problem_1, calculate_p
import matplotlib.pyplot as plt

a = 2
b = 10
X = 10
C = 2.8
g = 10

H1 = 5
H0 = 1
x_start_step = -5
x_end_step = x_start_step + X
x_start_periodical = 0
x_end_periodical = x_start_periodical + X
CFL = 0.45


def first_task_graphic(x1, h1, n1, x2, h2, n2, x3, h3, n3):
    x_step = np.linspace(x_start_step, x_end_step, 1000)

    exact_sol, flow = an_sol_problem_1(x_step, H1, H0, 0, 0, 0.4)

    step_graph = step_h(x_step, H0, H1)
    plt.figure(figsize=(10, 6))
    plt.plot(x_step, step_graph, label="step", linestyle="--")
    plt.plot(x_step, exact_sol, label=f"exact", linestyle="-", color="black", linewidth=1)
    plt.plot(x1, h1, label=f"n = {n1}", marker=".", linestyle="")
    # plt.plot(x2, h2, label=f"n = {n2}", marker=".", linestyle="")
    # plt.plot(x3, h3, label=f"n = {n3}", marker=".", linestyle="")
    plt.xlabel("x")
    plt.ylabel("h(x, t)")
    plt.legend()
    plt.grid()
    plt.show()


def second_task_graphic(x1, h1, n1):
    plt.figure(figsize=(10, 6))
    plt.plot(x1, h1, label=f"n = {n1}\nT = 2", linestyle="dotted")
    plt.xlabel("x")
    plt.ylabel("h(x, t)")
    plt.legend()
    plt.grid()
    plt.show()


def first_task():
    T = 0.4  # передать в функцию
    n1 = 50
    n2 = 100
    n3 = 200

    x1, h1 = rusanov_scheme_for_step(C, X, x_start_step, x_end_step, T, CFL, n1, H0, H1)
    x2, h2 = rusanov_scheme_for_step(C, X, x_start_step, x_end_step, T, CFL, n2, H0, H1)
    x3, h3 = rusanov_scheme_for_step(C, X, x_start_step, x_end_step, T, CFL, n3, H0, H1)
    first_task_graphic(x1, h1, n1, x2, h2, n2, x3, h3, n3)


def second_task(T):
    # T = 0.5
    n = 2000
    x, h = rusanov_scheme_periodical(a, b, C, X, x_start_periodical, x_end_periodical, T, CFL, n)
    max_lambda = max([float(i) for i in open("max_lambda.txt", "r").read().split()])
    # print(max_lambda)
    # second_task_graphic(x, h, n)
    return max_lambda


def third_task_graphics(n, t):
    x = np.genfromtxt(f"tmp/time_{t}_n_{n}_x.csv", delimiter=',', names=True)
    h = np.genfromtxt(f"tmp/time_{t}_n_{n}_h.csv", delimiter=',', names=True)
    plt.figure(figsize=(10, 6))
    plt.plot(x, h, label=f"n = {n}\nT = {t}", linestyle="dotted")
    plt.xlabel("x")
    plt.ylabel("h(x, t)")
    plt.legend()
    plt.grid()
    plt.show()


def third_task(T):
    n = [1001, 2001, 4001]
    delta_x1 = X / (n[0] - 1)
    delta_x2 = X / (n[1] - 1)
    delta_x3 = X / (n[2] - 1)
    delta_t1 = 0.05 * delta_x1
    delta_t2 = delta_t1 / 2
    delta_t3 = delta_t1 / 4
    print("T/dt", T / delta_t1, T / delta_t2, T / delta_t3)
    x1, h1, q1 = rusanov_scheme_third_task(a, b, C, X, x_start_periodical, x_end_periodical, delta_t1, delta_x1, n[0],
                                           int(T / delta_t1))
    x2, h2, q2 = rusanov_scheme_third_task(a, b, C, X, x_start_periodical, x_end_periodical, delta_t2, delta_x2, n[1],
                                           int(T / delta_t2))
    x3, h3, q3 = rusanov_scheme_third_task(a, b, C, X, x_start_periodical, x_end_periodical, delta_t3, delta_x3, n[2],
                                           int(T / delta_t3))

    # print(f"x1:", x1[:10])
    # print("x2:", x2[::2][:10])
    # print("x3:", x3[::4][:10])

    p = calculate_p(h1, h2, h3, q1, q2, q3)

    xp = np.linspace(x_start_periodical, x_end_periodical, 1001)
    plt.figure(figsize=(10, 6))
    plt.plot(xp, p, label=f"p", linestyle="dotted")
    plt.plot(x1, h1, 'k o', markersize=0.8, label=f"n = {n[0]}\nT = {T}", )
    plt.xlabel("x")
    plt.ylabel("h(x, t)")
    plt.legend()
    plt.grid()
    plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.plot(x1, h1, label=f"n = {n[0]}\nT = {T}", linestyle="dotted")
    # plt.plot(x2, h2, label=f"n = {n[1]}\nT = {T}", linestyle="dotted")
    # plt.plot(x3, h3, label=f"n = {n[2]}\nT = {T}", linestyle="dotted")
    # plt.xlabel("x")
    # plt.ylabel("h(x, t)")
    # plt.legend()
    # plt.grid()
    # plt.show()


if __name__ == "__main__":
    # first_task()
    T = 0.5
    # max_lambda = second_task(T)  # 8.352979187303404
    # print(max_lambda)
    third_task(T)
