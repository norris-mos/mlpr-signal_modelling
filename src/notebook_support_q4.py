import numpy as np
import matplotlib.pyplot as plt

def do_4a(X_shuf_train, y_shuf_train, X_shuf_val, y_shuf_val):
C_grid = np.arange(1, 21)

rmse = {"train": [None for _ in C_grid], "val": [None for _ in C_grid]}

plt.figure(0)
#plt.plot(np.arange(0, 1, 0.05)[-C:], X_shuf_train[0, -C:],'r+', label='linear fit')
plt.plot(np.arange(0, 1, 0.05), X_shuf_train[0],'r+', label='data')
plt.plot(1, y_shuf_train[0],'b+', label='truth')

to_show = [1, 3, 5, 15]

for i, C in enumerate(C_grid):
    ww, rr = np.linalg.lstsq(X_shuf_train[:,-C:], y_shuf_train, rcond=None)[0:2]
    #rmse["train"][i] = rr
    rmse["train"][i] = np.sqrt(((y_shuf_train - (X_shuf_train[:,-C:] @ ww)) ** 2).mean())
    rmse["val"][i] = np.sqrt(((y_shuf_val - (X_shuf_val[:,-C:] @ ww)) ** 2).mean())

    if C in to_show:
        np.random.seed(C * 13)
        col = (np.random.random(), np.random.random(), np.random.random())
        plt.scatter(1, ww @ X_shuf_train[0, -C:], c=col, label=f'C={C}')

plt.legend()
min_tr, min_val = min(rmse["train"]), min(rmse["val"])

# +1 needed below to account for index N of RMSE list corresponding to C=(N+1)
print("Best C for train set: ", np.where(rmse["train"] == min_tr)[0] + 1, "; RMSE: ", min_tr)
print("Best C for train set: ", np.where(rmse["val"] == min_val)[0] + 1, "; RMSE: ", min_val)

plt.figure(1)
plt.plot(C_grid, rmse["train"], label="train")
plt.plot(C_grid, rmse["val"], label="val")
plt.xlim(13, 20)
plt.ylim(0.0025, 0.0030)
plt.xlabel("C")
plt.ylabel("error")
plt.legend()

def do_4b():
    raise NotImplementedError()

def do_4c():
    raise NotImplementedError()
