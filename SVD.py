import numpy as np
import scipy.linalg
from PIL import Image
import matplotlib.pyplot as plt
import math

A = np.array([[2,0,8,6,0],
              [1,6,0,1,7],
              [5,0,7,4,0],
              [7,0,8,5,0],
              [0,10,0,0,7]]) # matrix to decompose

print('----- Power method -----\n')
def eigen_power(A, it):
    _, x_len = A.shape
    x = np.random.uniform(-1, 1, (x_len, 1))  # initial vector
    # x = x.transpose(1)

    for k in range(1, it):
        z = np.dot(A, x)
        z_norm = np.linalg.norm(z, ord='fro')
        r = z[1]/x[1]
        x = z/z_norm
        print("K = ", k,'x = ',x, 'r = ', r)
    return x, r
#print(eigen_power(A, it=100))
"""inverse power method"""
# x0 = np.array([3,7,-13])
# x0 = np.transpose(x0)
# M = 25

print('----- Iverse power method -----\n')
def eigen_inverse(A, s,  it):
    _, x_len = A.shape
    x = np.random.uniform(-1, 1, (x_len, 1))
    aStar = A - np.identity(x_len)*s
    for k in range(1, it):
        lu = scipy.linalg.lu_factor(aStar)
        z = scipy.linalg.lu_solve(lu, x)
        z_norm = np.linalg.norm(z, ord='fro')
        r = z[0] / x[0]
        x = z / z_norm
        print("K = ", k, 'x = ', x, 'r = ', r)
    return x, r
#print(eigen_inverse(A, it=10))

#
# print('----- Check the result -----\n')
# z = np.linalg.eig(A)
# print(z)

def make_householder(a):
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return H

'''QR decomposition using Householders method to find eigenvalues and eigenvectors'''
def qr(A):
    m, n = A.shape
    Q = np.eye(m) # Return a 2-D array with ones on the diagonal and zeros elsewhere.
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q, A

q, r = qr(A)
print('q:\n', q.round(6))
print('r:\n', r.round(6))

def eig_val_vec (A, it=100):
    m, n = A.shape
    U = np.identity(m)
    for i in range(1, it):
        X = A
        q, r = qr(X)
        A = np.dot(r,q)
        U = np.dot(U,q)
    return np.diag(A), U

def QR_eig_shift (A, it=100, sigma = 1):
    m, n = A.shape
    I = np.identity(m)
    U = np.identity(m)
    for i in range(1, it):
        X = A - sigma*I
        q, r = qr(X)
        A = np.dot(r,q) + sigma*I
        U = np.dot(U,q)
    return np.diag(A), U


""" # image example """

img = Image.open('grumpy-cat-christmas-clipart-small_100.jpg')
imggray = img.convert('LA')
# imggray.show()

imgmat = np.array(list(imggray.getdata(band=0)), float)
imgmat.shape = (imggray.size[1], imggray.size[0])
imgmat = np.matrix(imgmat)
imgmat = np.squeeze(np.asarray(imgmat))
# img_1 = Image.fromarray(imgmat)
# img_1.show()


sigma_1, U_1 = QR_eig_shift(np.dot(A, np.transpose(A)))
idx = sigma_1.argsort()[::-1]
sigma_1 = sigma_1[idx]
U_1 = U_1[:,idx]
sigma_1, V_1 = QR_eig_shift(np.dot(np.transpose(A), A))
idx = sigma_1.argsort()[::-1]
print(idx)
sigma_1 = sigma_1[idx]
V_1 = V_1[:,idx]
# V_1 = modifiedGramSchmidt(V_1)

V_1 = np.transpose(V_1)
sigma_1 = np.sqrt(sigma_1)

# reconstimg_1 = np.dot(U_star[:, :3], np.dot(np.diag(sigma_1[:3]),V_star[:3, :]))

U, sigma, V  = np.linalg.svd(A, full_matrices=True, compute_uv=True)
reconstimg = np.dot(U[:, :3], np.dot(np.diag(sigma[:3]), V[:3, :]))

np.set_printoptions(precision=5)
# print(V_1, V)
# img_2 = Image.fromarray(reconstimg_1)
# img_2.show()
# img_2 = Image.fromarray(reconstimg)
# img_2.show()
def SignFlip(A, U, S, V):
    # 1 left singular vector
    Y = A - np.dot(U, np.dot(np.diag(S), V))
    K = U.shape[1]
    J = Y.shape[1]

    s_left=np.zeros(K, dtype=float)
    s_right=np.zeros(K, dtype=float)

    for k in range(0,K):
        for j in range(0,J):
            s_left[k] = s_left[k] + np.dot(np.dot(np.transpose(U[:,k]),Y[:,j]),np.square(np.dot(np.transpose(U[:,k]),Y[:,j])))

    for k in range(0, K):
        for j in range(0, J):
            s_right[k] = s_right[k] + np.dot(np.dot(np.transpose(V[:, k]), np.transpose(Y[j, :])),
                                                    np.square(np.dot(np.transpose(V[:, k]), np.transpose(Y[j, :]))))
    for k in range(0, K):
        if s_left[k]*s_right[k]<0:
            if np.sqrt(np.square(s_left[k]))<np.sqrt(np.square(s_right[k])):
                s_left[k]=-s_left[k]

            else:
                s_right[k]=-s_right[k]

        # U = U[:,k]*np.sign(s_left[k])
        # V = V[:,k]*np.sign(s_right[k])
    return s_left, s_right

U_star, V_star= SignFlip(A,U_1, sigma_1, V_1)
print(U_star, U_1, U)
