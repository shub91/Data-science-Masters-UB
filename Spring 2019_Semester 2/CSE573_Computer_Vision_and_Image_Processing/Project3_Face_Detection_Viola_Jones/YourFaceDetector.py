"""
Project 3: Face Detection using Viola Jones Algorithm

Name: Shubham Sharma
UBIT: ss628

"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_selection import f_classif, SelectPercentile
import matplotlib.pyplot as plt
from imutils import paths
import sys
import pickle
import os
import math
import json
import cv2

class VJ_alg:
    def __init__(self, N = 10):

        self.N = N # No of weak C used
        self.weight = [] # weight of each weak classifier
        self.wc = [] # weak classifier

    def train(self, tr, ps, ns): # trains the algorith (Viola Jones)

        W = np.zeros(len(tr))
        tr_data = []
        for x in range(len(tr)):
            tr_data.append((compute_integral_im(tr[x][0]), tr[x][1]))
            if tr[x][1] == 1:
                W[x] = 1.0 / (2 * ps)
            else:
                W[x] = 1.0 / (2 * ns)

        ft = self.ft_building(tr_data[0][0].shape)
        X, y = self.ft_apply(ft, tr_data)
        i = SelectPercentile(f_classif, percentile=10).fit(X.T, y).get_support(indices=True)
        X = X[i]
        ft = ft[i]

        for t in range(self.N):
            W = W / np.linalg.norm(W)
            weak_C = self.tr_wc(X, y, ft, W)
            C, err, Acc = self.WC_best(weak_C, W, tr_data)
            B = err / (1.0 - err) # beta value
            for i in range(len(Acc)):
                W[i] = W[i] * (B ** (1 - Acc[i]))
            A = math.log(1.0/B)
            self.weight.append(A)
            self.wc.append(C)

    def tr_wc(self, X, y, ft, W):

        P, N = 0, 0
        for w, l in zip(W, y):
            if l == 1:
                P += w
            else:
                N += w

        C_s = []
        T_ft = X.shape[0]
        for i, feature in enumerate(X):
            app_ft = sorted(zip(W, feature, y), key=lambda x: x[1])
            P, N = 0, 0
            P_W, N_W = 0, 0
            min_err, b_ft, b_T, b_P = float('inf'), None, None, None
            for w, f, l in app_ft:
                err = min(N_W + P - P_W, P_W + N - N_W)
                if err < min_err:
                    min_err = err
                    b_ft = ft[i]
                    b_T = f
                    b_P = 1 if P > N else -1

                if l == 1:
                    P += 1
                    P_W += w
                else:
                    N += 1
                    N_W += w

            C = wk_clf(b_ft[0], b_ft[1], b_T, b_P)
            C_s.append(C)
        return C_s

    def ft_building(self, im_shp):
        print("[INFO] Building feautures from training data to choose the best weak classifier")
        H, W = im_shp
        ft = []
        for w in range(1, W+1):
            for h in range(1, H+1):
                i = 0
                while i + w < W:
                    j = 0
                    while j + h < H:
                        adj = ft_rectangle(i, j, w, h)
                        R = ft_rectangle(i+w, j, w, h)
                        if i + 2 * w < W:
                            ft.append(([R], [adj]))

                        B = ft_rectangle(i, j+h, w, h)
                        if j + 2 * h < H:
                            ft.append(([adj], [B]))

                        R_2 = ft_rectangle(i+2*w, j, w, h)

                        if i + 3 * w < W:
                            ft.append(([R], [R_2, adj]))

                        B_2 = ft_rectangle(i, j+2*h, w, h)
                        if j + 3 * h < H:
                            ft.append(([B], [B_2, adj]))

                        B_R = ft_rectangle(i+w, j+h, w, h)
                        if i + 2 * w < W and j + 2 * h < H:
                            ft.append(([R, B], [adj, B_R]))

                        j += 1
                    i += 1
        return np.array(ft)

    def WC_best(self, C, W, tr_data):
        best_C, best_err, best_Acc = None, float('inf'), None
        for C in C:
            err, Acc = 0, []
            # Calculating average weighted error of each weak classifier
            for data, w in zip(tr_data, W):
                dist = abs(C.im_classifier(data[0]) - data[1])
                Acc.append(dist)
                err += w * dist
            err = err / len(tr_data)
            if err < best_err:
                best_C, best_err, best_Acc = C, err, Acc
        return best_C, best_err, best_Acc

    def ft_apply(self, ft, tr_data):
        print("[INFO] Applying precomputed features during training ... ")
        X = np.zeros((len(ft), len(tr_data)))
        y = np.array(list(map(lambda data: data[1], tr_data)))
        i = 0
        for P_a, N_a in ft:
            feature = lambda ii: sum([position.ft_comp(ii) for position in P_a]) - sum([neg.ft_comp(ii) for neg in N_a])
            X[i] = list(map(lambda data: feature(data[0]), tr_data))
            i += 1
        return X, y

    def im_classifier(self, im):
        T = 0
        ii = compute_integral_im(im)
        for A, C in zip(self.weight, self.wc):
            T += A * C.im_classifier(ii)
        return 1 if T >= 0.5 * sum(self.weight) else 0

    def F_faces(self, D):
        self.A_model(D)
        return self.result

    def A_model(self, dirlist):
        for im in dirlist:
            i = cv2.imread(im, 0)
            self.im_classifier(i)

    def save(self, filename):
        with open(filename + ".pkl", 'wb') as f:
            pickle.dump(self, f)

    def load(F):
        with open(F+".pkl", 'rb') as file:
            return pickle.load(file)

class wk_clf:
    def __init__(self, P_a, N_a, T, P):
        self.P_a = P_a # Positive contributions to the feauture
        self.N_a = N_a # Negative contributions to the feature
        self.T = T # Weak Classifier's threshold
        self.P = P # Weak Classifier's polarity

    def im_classifier(self, x):
        feature = lambda ii: sum([position.ft_comp(ii) for position in self.P_a]) - sum([neg.ft_comp(ii) for neg in self.N_a])
        return 1 if self.P * feature(x) < self.P * self.T else 0

class ft_rectangle:
    def __init__(self, x, y, W, H):
        self.x = x # row-index of the top left corner of the rectangle
        self.y = y # column-index of the top left corner of the rectangle
        self.W = W # Rectangle's W
        self.H = H # Rectangle's H

    def ft_comp(self, ii):
        a = ii[self.y+self.H][self.x+self.W] + ii[self.y][self.x]
        b = ii[self.y+self.H][self.x]+ii[self.y][self.x+self.W]
        return a - b

def compute_integral_im(im): # Calculating integral im matrix
    print("[INFO] Calculating Integral image Matrix ... ")
    #     im = np.hstack((np.zeros((im.shape[0],1)), im))
    #     im = np.vstack((np.zeros((im.shape[1])), im))
    #     print(im)
    ii = np.zeros(im.shape)
    s = np.zeros(im.shape)
    #     print(im.shape[1])
    #     print(len(im))
    #     print(im.shape[0])
    #     print(len(im[1]))
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            s[i][j] = s[i-1][j] + im[i][j] if i-1 >= 0 else im[i][j]
            ii[i][j] = ii[i][j-1] + s[i][j] if j-1 >= 0 else s[i][j]
    #     print(ii)
    return ii

def face_detect(D_list):
    with open("Model_VJ.pkl", "rb") as file:
        M = pickle.load(file)
        return M.F_faces(D_list)

# def main():
    arg = sys.argv[1]
if len(sys.argv) != 2:
    print("Data directory missing or incorrect number of arguments")
    sys.exit(1)
arg = sys.argv[1]
f_list = os.listdir(arg)
f_list = list(map(lambda path: arg + "/" + path, f_list))
R = "results.json"
rfile = face_detect(f_list)

with open(R, "w") as f:
    json.dump(rfile, f)
