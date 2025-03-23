import numpy as np


from rusanov_scheme import rusanov_scheme_for_step, step_h, rusanov_scheme_periodical
import matplotlib.pyplot as plt

a = 2
b = 10
X = 10  #
C = 2.5
g = 10

H1 = 5
H0 = 1
x_start = -5
X = 10
x_end = x_start + X
CFL = 0.5


def first_task_graphic(x1, h1, n1, x2, h2, n2, x3, h3, n3):
    x_step = np.linspace(x_start, x_end, 1000)
    step_graph = step_h(x_step, H0, H1)
    plt.figure(figsize=(10, 6))
    plt.plot(x_step, step_graph, label="step", linestyle="--")
    plt.plot(x1, h1, label=f"n = {n1}", linestyle="dotted")
    plt.plot(x2, h2, label=f"n = {n2}", linestyle="dotted")
    plt.plot(x3, h3, label=f"n = {n3}", linestyle="dotted")
    plt.xlabel("x")
    plt.ylabel("h(x, t)")
    plt.legend()
    plt.grid()
    plt.show()


def second_task_graphic(x1, h1, n1):
    plt.figure(figsize=(10, 6))
    plt.plot(x1, h1, label=f"n = {n1}", linestyle="dotted")
    plt.xlabel("x")
    plt.ylabel("h(x, t)")
    plt.legend()
    plt.grid()
    plt.show()

def first_task():
    T = 0.3
    n1 = 50
    n2 = 100
    n3 = 200

    x1, h1 = rusanov_scheme_for_step(C, X, x_start, x_end, T, CFL, n1, H0, H1)
    x2, h2 = rusanov_scheme_for_step(C, X, x_start, x_end, T, CFL, n2, H0, H1)
    x3, h3 = rusanov_scheme_for_step(C, X, x_start, x_end, T, CFL, n3, H0, H1)
    first_task_graphic(x1, h1, n1, x2, h2, n2, x3, h3, n3)

def second_task():
    T = 0.1
    n = 2000
    x, h = rusanov_scheme_periodical(a, b, C, X, x_start, x_end, T, CFL, n)
    max_lambda = max([float(i) for i in open("max_lambda.txt", "r").read().split()])
    print(max_lambda)
    second_task_graphic(x, h, n)
    return max_lambda

def third_task(max_lamdba):
    T1 = 0.5
    T2 = 1
    T3 = 2
    n1 = 1000
    n2 = 2000
    n3 = 4000
    delta_x1 = X / n1
    delta_t1 = CFL * delta_x1 / max_lambda
    delta_t2 = delta_t1 / 2
    delta_t3 = delta_t1 / 4

if __name__ == "__main__":
    # first_task()
    max_lambda = second_task()