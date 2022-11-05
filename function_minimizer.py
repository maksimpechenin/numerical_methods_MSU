import numpy as np
import math

x0 = np.array([2, 4.01, 2.01])
eps = 10 ** (-6)
tau = 10 ** (-3)
beta, lambd, mu = 1, 0.5, 2


def f(x):
    return (2*x[0]-5) ** 2 + (x[1]-4) ** 2 + (x[2]-2) ** 2


def df(x, f=f, tau=tau):
    x = list(x)
    grad = np.zeros(len(x))
    for i in range(len(x)):
        x[i] += tau
        grad[i] = f(x) / (2 * tau)
        x[i] -= 2 * tau
        grad[i] -= f(x) / (2 * tau)
        x[i] += tau
    return grad


def ddf(x, f=f, h=tau):
    dlina = len(x)
    e = np.eye(dlina)  # единичная матрица
    A = np.zeros((dlina, dlina))
    for i in range(dlina):
        for j in range(dlina):
            A[i][j] = (f(x + (e[i] + e[j]) * h) - f(x + e[j] * h) - f(x + e[i] * h) + f(x)) / (h ** 2)
    return A


def mindrob(xk, f=f):
    alpha = beta
    if f(xk - alpha * df(xk)) - f(xk) >= eps:
        alpha *= lambd
        while f(xk - alpha * df(xk)) - f(xk) >= eps:
            alpha *= lambd
    elif f(xk - alpha * df(xk)) < f(xk):
        alpha *= mu
        while f(xk - alpha * df(xk)) < f(xk - beta * df(xk)):
            alpha *= mu
    else:
        alpha = 0
    return alpha


def gradmethod(xk, f=f, delta=math.sqrt(eps)):
    print('gradient descent')
    alphak = mindrob(xk)
    xkplus1 = xk - alphak * df(xk)
    i = 0
    while np.dot(alphak * df(xk), alphak * df(xk)) > delta or \
        abs(f(xkplus1) - f(xk)) > delta or \
        np.dot(df(xkplus1), df(xkplus1)) > delta:
        i += 1
        alphak = mindrob(xk)
        xtmp = xk
        xk = xkplus1
        xkplus1 = xtmp - alphak * df(xtmp)
    print(i)
    return xkplus1


def goldenratio(xk, f=f, epsilon=eps):
    r = (3 - math.sqrt(5)) / 2
    a, b = -5, 5
    c = a + r * (b - a)
    d = b - r * (b - a)
    while b - a >= epsilon:
        hk = - np.linalg.inv(ddf(xk)) @ df(xk)
        if f(xk + c * hk) >= f(xk + d * hk):
            a = c
            c = d
            d = b - r * (b - a)
        else:
            b = d
            d = c
            c = a + r * (b - a)
    return (a + b) / 2


def newtonmethod(xk, f=f, delta=eps):
    print('newton method')
    alphak = goldenratio(xk)
    hk = - np.linalg.inv(ddf(xk)) @ df(xk)
    xkplus1 = xk + alphak * hk
    i = 0
    while np.dot(alphak * hk, alphak * hk) > delta or \
            abs(f(xkplus1) - f(xk)) > delta or \
            np.dot(df(xkplus1), df(xkplus1)) > delta:
        i += 1
        alphak = goldenratio(xk)
        xtmp = xk
        xk = xkplus1
        hk = - np.linalg.inv(ddf(xtmp)) @ df(xtmp)
        xkplus1 = xtmp + alphak * hk
    print(i)
    return xkplus1



xk = gradmethod(x0)
print(xk)
print('====')
xfinal = newtonmethod(xk)
print(xfinal)
