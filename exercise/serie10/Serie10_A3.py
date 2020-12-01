import numpy as np


def jacobi_fun(A, b, x_start):
    L = np.tril(A, k=-1)
    R = np.triu(A, k=1)
    D_inv = np.linalg.inv(1 / A)
    B_j = -D_inv @ (L + R)
    c = D_inv @ b
    x = B_j @ x_start + c
    return x

def gauss_seidel_fun(A, b, x_start):
    L = np.tril(A, k=-1)
    R = np.triu(A, k=1)
    D = np.diag(A)
    DL_inv = np.linalg.inv(D + L)
    B_gs = -DL_inv @ R
    c = DL_inv @ b
    x = B_gs @ x_start + c
    return x



'''
# A Matrix A
# b Vektor Ax=b
# x0 Startvektor
# tol Max fehler
# opt gauss-seidel oder jacobi

'''
def Ibrahim_Karim_Gruppe_S10_Aufg3a(A,b,x0,tol,opt):
    x = np.zeros(len(b)) if x0 is None else x0
    count = 0
    switcher = {
        1: jacobi_fun(A,b,x),
        2: gauss_seidel_fun(A,b,x)
    }
    limit = 10000
    for count in range(1, limit):
        x_new = switcher.get(opt)
        count += 1
        if np.allclose(x, x_new, rtol=tol):
            break
        x = x_new

    return x, count


# xn = Vektor nach n iterationen
# n = Anzahl iterationen
# n2 = Anzahl Schritte gemäss a-priori Abschätzung

A = np.array([[8,5,2],[5,9,1],[4,2,7]])
b = np.array([19,5,34])
x = np.array([0,0,0])
print(Ibrahim_Karim_Gruppe_S10_Aufg3a(A,b,None,1e-4,opt=1))
print(Ibrahim_Karim_Gruppe_S10_Aufg3a(A,b,None,1e-4,opt=2))