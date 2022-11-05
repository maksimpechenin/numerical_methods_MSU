import numpy as np
import math

n = 9  # должно быть нечетным!
deltaN = 20  # число шагов, на которое увеличиваеся количество точек разбиения отрезка [a, b] при методе квадратур
a, b = 0, 1
eps = 10 ** (-5)
h = (b - a) / (n - 1)
s = [a + i * h for i in range(n)]


def real_u(x):
    return x + math.exp(-x)


def K(x, t):
    return x * math.exp(t) / 2


def f(x):
    return math.exp(-x)


def matrix_A(h, n, s):
    A = []
    for i in range(n - 1):
        si = s[i]
        si2 = (s[i] + s[i + 1]) / 2
        Ai = [h * K(si, s[0]) / 6, 4 * h * K(si, s[0] / 2 + s[1] / 2) / 6]
        Ai2 = [h * K(si2, s[0]) / 6, 4 * h * K(si2, s[0] / 2 + s[1] / 2) / 6]
        for j in range(1, n - 1):
            Ai.append(4 * h * K(si, s[j] / 2 + s[j + 1] / 2) / 6)
            Ai.append(2 * h * K(si, s[j + 1]) / 6)
            Ai2.append(4 * h * K(si2, s[j] / 2 + s[j + 1] / 2) / 6)
            Ai2.append(2 * h * K(si2, s[j + 1]) / 6)
        Ai.append(h * K(si, s[-1]) / 6)
        Ai2.append(h * K(si2, s[-1]) / 6)
        A.append(Ai)
        A.append(Ai2)
    Alast = [h * K(s[-1], s[0]) / 6, 4 * h * K(s[-1], s[0] / 2 + s[1] / 2) / 6]
    for j in range(1, n - 1):
        Alast.append(4 * h * K(s[-1], s[j] / 2 + s[j + 1] / 2) / 6)
        Alast.append(2 * h * K(s[-1], s[j + 1]) / 6)
    Alast.append(h * K(s[-1], s[n - 1]) / 6)
    A.append(Alast)
    return np.array(A)


def Fvec(s):
    n = len(s)
    result = []
    for i in range(n - 1):
        result.append(f(s[i]))
        result.append(f(s[i] / 2 + s[i + 1] / 2))
    result.append(f(s[-1]))
    return np.array(result)


def find_U(h, n, s):
    A = matrix_A(h, n, s)
    F = Fvec(s)
    return np.linalg.inv(np.eye(len(F)) - A) @ F


def make_Un(x, U, s, h):
    Un = 0
    n = len(s)
    for i in range(n - 1):
        Un += K(x, s[i]) * U[2 * i] + \
              4 * K(x, s[i] / 2 + s[i + 1] / 2) * U[2 * i + 1] + \
              K(x, s[i + 1]) * U[2 * i + 2]
    Un *= h / 6
    Un += f(x)
    return Un


def integral(func1, func2, s, h):
    result = 0
    n = len(s)
    for i in range(n - 1):
        result += (1 / 6) * h * ((func1(s[i]) - func2(s[i])) ** 2 + 4 * (
                    func1(s[i] / 2 + s[i + 1] / 2) - func2(s[i] / 2 + s[i + 1] / 2)) ** 2 + (
                                             func1(s[i + 1]) - func2(s[i + 1])) ** 2)
    return result


def adaptive(func1, func2, eps=eps):
    I = 0
    alpha, beta = a, b
    while alpha < beta:
        hi = beta - alpha
        s = [alpha, beta]
        Ih = integral(func1, func2, s, hi)
        hi2 = hi / 2
        s2 = [alpha, alpha / 2 + beta / 2, beta]
        Ih2 = integral(func1, func2, s=s2, h=hi2)
        DELTAi = (Ih - Ih2) / (1 - 2 ** (-2))
        if DELTAi < eps * hi / (b - a):
            alpha, beta = beta, b
            I += Ih2 + DELTAi
        else:
            beta = (alpha + beta) / 2
    return I


def main(number):
    N, H, S = n, h, s
    U = find_U(H, N, S)

    def Un(x):
        return make_Un(x, U, S, H)

    n_1 = 2*N + 1
    h_1 = (b - a) / (n_1 - 1)
    s_1 = [a + i * h_1 for i in range(n_1)]
    U_1 = find_U(h_1, n_1, s_1)

    def Un_1(x):
        return make_Un(x, U_1, s_1, h_1)

    i = 0
    tmp = math.sqrt(adaptive(Un, Un_1))
    while math.sqrt(adaptive(Un, Un_1)) >= eps:
        print(Un(1))
        print(Un_1(1))
        print(math.sqrt(adaptive(Un, Un_1)), tmp/math.sqrt(adaptive(Un, Un_1)))
        tmp = math.sqrt(adaptive(Un, Un_1))
        print(i)
        i += 1
        N, H, S = n_1, h_1, s_1
        U = find_U(H, N, S)
        n_1 = N + deltaN
        h_1 = (b - a) / (n_1 - 1)
        s_1 = [a + i * h_1 for i in range(n_1)]
        U_1 = find_U(h_1, n_1, s_1)
    return Un(number), Un_1(number)


print(main(5))
print(real_u(5))
