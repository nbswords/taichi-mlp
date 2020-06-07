"""
code reference:https://blog.csdn.net/chuan403082010/article/details/86370551
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from alive_progress import alive_bar


def loaddata(filename):
    fr = open(filename)
    x = []
    y = []
    for line in fr.readlines():
        line = line.strip().split()
        x.append([float(line[0]), float(line[1])])
        y.append([float(line[-1])])
    return np.mat(x), np.mat(y)


# min-max scaling
def scaling(data):
    min = np.min(data, 0)
    max = np.max(data, 0)
    new_data = (data-min)/(max-min)
    return new_data, min, max


def sigmoid(data):
    return 1/(1+np.exp(-data))

# Neural Network calculation


def calc_nn(X, label, n_hidden_dim, alpha, reg_lambda, num_step, anim_inter=100):
    # init. w b
    W1 = np.mat(np.random.randn(2, n_hidden_dim))  # (2,3)
    b1 = np.mat(np.random.randn(1, n_hidden_dim))  # (1,3)
    W2 = np.mat(np.random.randn(n_hidden_dim, 1))  # (3,1)
    b2 = np.mat(np.random.randn(1, 1))  # (1,1)
    w1_save = []
    b1_save = []
    w2_save = []
    b2_save = []
    step = []
    print("Neural Network Calculating...")
    with alive_bar(num_step) as bar:
        for stepi in range(num_step):
            z1 = X*W1 + b1  # (20,2)(2,3)+(1,3)=(20,3)
            a1 = sigmoid(z1)  # (20,3)
            z2 = a1*W2 + b2  # (20,3)(3,1)+(1,1)=(20,1)
            a2 = sigmoid(z2)  # (20,1)
            # BP
            delta2 = a2-label  # (20,1)
            dW2 = a1.T*delta2 + reg_lambda*W2
            #    (3,20)(20,1) = (3,1)
            db2 = np.mat(np.ones((X.shape[0], 1))).T * delta2
            delta1 = np.mat((delta2*W2.T).A*a1.A*(1-a1).A)  # (20,3)
            dW1 = X.T*delta1 + reg_lambda*W1
            db1 = np.sum(delta1, 0)
            # new W,b
            W2 -= alpha*dW2
            b2 -= alpha*db2
            W1 -= alpha*dW1
            b1 -= alpha*db1
            if stepi % anim_inter == 0:
                w1_save.append(W1.copy())
                b1_save.append(b1.copy())
                w2_save.append(W2.copy())
                b2_save.append(b2.copy())
                step.append(stepi)
            bar()

    return W1, b1, W2, b2, w1_save, b1_save, w2_save, b2_save, step


if __name__ == "__main__":

    xmat, ymat = loaddata('taichi_data.txt')
    xmat_s, xmat_min, xmat_max = scaling(xmat)

    anim = 0  # 0: no anim. 1: yes
    # parameters
    # When memory error appear, you can set the parameters here fewer
    if anim == 0:
        W1, b1, W2, b2, w1_save, b1_save, w2_save, b2_save, step = calc_nn(
            xmat_s, ymat, n_hidden_dim=300, alpha=0.05, reg_lambda=0, num_step=50000)
    if anim == 1:
        W1, b1, W2, b2, w1_save, b1_save, w2_save, b2_save, step = calc_nn(
            xmat_s, ymat, n_hidden_dim=300, alpha=0.05, reg_lambda=0, num_step=50000, anim_inter=1000)

    # contour
    contour_x1 = np.arange(-0.5, 10.5, 0.01)
    contour_x2 = np.arange(-0.5, 10.5, 0.01)
    contour_x1, contour_x2 = np.meshgrid(contour_x1, contour_x2)
    plotx_old = np.c_[contour_x1.ravel(), contour_x2.ravel()]
    plotx = (plotx_old-xmat_min)/(xmat_max-xmat_min)
    """
    z1 and z2 are output of single layer NN
    act1 and act2 are output of sigmoid activation function
    """
    # animation
    if anim == 1:
        print("Ploting Taichi animation...")
        with alive_bar(len(w1_save)) as bar:
            for i in range(len(w1_save)):
                plt.clf()
                z1 = plotx*w1_save[i]+b1_save[i]
                act1 = sigmoid(z1)
                z2 = act1*w2_save[i] + b2_save[i]
                act2 = sigmoid(z2)
                ploty_new = np.reshape(act2, contour_x1.shape)
                plt.contourf(contour_x1, contour_x2, ploty_new,
                             3, alpha=0.5, cmap='gray')
                cont = plt.contour(contour_x1, contour_x2, ploty_new, 3)
                plt.clabel(cont, inline=True, fontsize=10)
                plt.scatter(xmat[:, 0][ymat == 0].A, xmat[:, 1][ymat == 0].A,
                            s=50, marker='o', label='0', color='black', cmap='gray')
                plt.scatter(xmat[:, 0][ymat == 1].A, xmat[:, 1][ymat == 1].A,
                            s=50, marker='o', label='1', color='white', cmap='gray')
                plt.grid()
                plt.legend(loc=1)
                plt.title('Taichi step:%s' % np.str(step[i]))
                plt.pause(0.0001)
                bar()
        plt.show()
    if anim == 0:
        print("Ploting taichi...")
        z1 = plotx * W1 + b1
        act1 = sigmoid(z1)
        z2 = act1 * W2 + b2
        act2 = sigmoid(z2)
        ploty_new = np.reshape(act2, contour_x1.shape)
        plt.contourf(contour_x1, contour_x2, ploty_new,
                     1, alpha=0.5, cmap='gray')
        cont = plt.contour(contour_x1, contour_x2, ploty_new, 1)
        plt.clabel(cont, inline=True, fontsize=10)
        plt.scatter(xmat[:, 0][ymat == 0].A, xmat[:, 1][ymat == 0].A,
                    s=50, marker='o', label='0', color='black', cmap='gray')
        plt.scatter(xmat[:, 0][ymat == 1].A, xmat[:, 1][ymat == 1].A,
                    s=50, marker='o', label='1', color='white', cmap='gray')
        plt.grid()
        plt.legend(loc=1)
        plt.title('Taichi')
        plt.show()
