import numpy as np


from rusanov_scheme import rusanov_scheme_for_step, step_h, rusanov_scheme_periodical, rusanov_scheme_third_task, \
    an_sol_problem_1, calculate_p
import matplotlib.pyplot as plt

a = 2
b = 10
X = 10  #
C = 2.5
g = 10

H1 = 5
H0 = 1
x_start = 0
X = 10
x_end = x_start + X
CFL = 0.5 #0.45


def first_task_graphic(x1, h1, n1, x2, h2, n2, x3, h3, n3):
    x_step = np.linspace(x_start, x_end, 1000)

    exact_sol, flow = an_sol_problem_1(x_step, H1, H0,0, 0, 0.4)

    step_graph = step_h(x_step, H0, H1)
    plt.figure(figsize=(10, 6))
    plt.plot(x_step, step_graph, label="step", linestyle="--")
    plt.plot(x_step, exact_sol, label=f"exact", linestyle="-", color="black", linewidth=1)
    plt.plot(x1, h1, label=f"n = {n1}", marker=".",  linestyle="")
    plt.plot(x2, h2, label=f"n = {n2}", marker=".", linestyle="")
    plt.plot(x3, h3, label=f"n = {n3}", marker=".", linestyle="")
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
    T = 0.5 #передать в функцию
    n1 = 50
    n2 = 100
    n3 = 200

    x1, h1 = rusanov_scheme_for_step(C, X, x_start, x_end, T, CFL, n1, H0, H1)
    x2, h2 = rusanov_scheme_for_step(C, X, x_start, x_end, T, CFL, n2, H0, H1)
    x3, h3 = rusanov_scheme_for_step(C, X, x_start, x_end, T, CFL, n3, H0, H1)
    first_task_graphic(x1, h1, n1, x2, h2, n2, x3, h3, n3)

def second_task():
    T = 0.5
    n = 2000
    x, h = rusanov_scheme_periodical(a, b, C, X, x_start, x_end, T, CFL, n)
    max_lambda = max([float(i) for i in open("max_lambda.txt", "r").read().split()])
    print(max_lambda)
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

def third_task(max_lambda):
    T = [0.5, 1, 2]
    n = [1000, 2000, 4000]
    delta_x1 = X / n[0]
    delta_t1 = CFL * delta_x1 / max_lambda
    delta_t2 = delta_t1 / 2
    delta_t3 = delta_t1 / 4
    x1, h1, q1 = rusanov_scheme_third_task(a, b, C, X, x_start, x_end, T[0], delta_t1, n[0])
    x2, h2, q2 = rusanov_scheme_third_task(a, b, C, X, x_start, x_end, T[0], delta_t2, n[1])
    x3, h3, q3 = rusanov_scheme_third_task(a, b, C, X, x_start, x_end, T[0], delta_t3, n[2])

    # u1 = q1/h1
    # u2 = q2/h2
    # u3 = q3/h3

    pq = calculate_p(q1, q2, q3)
    ph = calculate_p(h1, h2, h3)
    p = pq/ph

    print(f"p: {p}")

    xp = np.linspace(x_start, x_end, 1000)
    plt.figure(figsize=(10, 6))
    plt.plot(xp, p, label=f"p", linestyle="dotted")
    plt.xlabel("x")
    plt.ylabel("h(x, t)")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x1, h1, label=f"n = {n[0]}\nT = {T[0]}", linestyle="dotted")
    plt.plot(x2, h2, label=f"n = {n[0]}\nT = {T[0]}", linestyle="dotted")
    plt.plot(x3, h3, label=f"n = {n[0]}\nT = {T[0]}", linestyle="dotted")
    plt.xlabel("x")
    plt.ylabel("h(x, t)")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    first_task()
    # max_lambda = second_task() #8.352979187303404
    # third_task(max_lambda)