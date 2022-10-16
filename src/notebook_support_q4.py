import numpy as np
import matplotlib.pyplot as plt
import src.notebook_support_q3 as q3

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
    print("Best C for val set: ", np.where(rmse["val"] == min_val)[0] + 1, "; RMSE: ", min_val)

    plt.figure(1)
    plt.plot(C_grid, rmse["train"], label="train")
    plt.plot(C_grid, rmse["val"], label="val")
    plt.xlim(13, 20)
    plt.ylim(0.0025, 0.0030)
    plt.xlabel("C")
    plt.ylabel("error")
    plt.legend()

def do_4b_3ii(c1,c2,k,X_shuf_train, y_shuf_train, X_shuf_test, y_shuf_test,X_shuf_val,y_shuf_val):
  

    # lowest val model- C = 15
    ww, rr = np.linalg.lstsq(X_shuf_train[:,-c1:], y_shuf_train, rcond=None)[0:2]
    # best polynomial from Q3: C=5,K=2
    vv = q3.make_vv(C=c2,K=k)

    


    # apply to test set and evaluate rmse
    pred1_train = X_shuf_train[:,-c1:] @ ww
    pred2_train = X_shuf_train[:,-c2:] @ vv
    pred1_val = X_shuf_val[:,-c1:] @ ww
    pred2_val = X_shuf_val[:,-c2:] @ vv
    pred1_test = X_shuf_test[:,-c1:] @ ww
    pred2_test = X_shuf_test[:,-c2:] @ vv
    # getting rmse for val and train
    rmse_train1 = np.sqrt(((y_shuf_train - pred1_train)) ** 2).mean()
    rmse_train2 = np.sqrt(((y_shuf_train - pred2_train)) ** 2).mean()
    rmse_val1 = np.sqrt(((y_shuf_val - pred1_val)) ** 2).mean()
    rmse_val2 = np.sqrt(((y_shuf_val - pred2_val)) ** 2).mean()
    # check rmse 
    rmse1 = np.sqrt(((y_shuf_test - pred1_test)) ** 2).mean()
    rmse2 = np.sqrt(((y_shuf_test - pred2_test)) ** 2).mean()

    print(f'The RMSE for the ww model: test: {rmse1} with C={c1} ')
    print(f'The RMSE for the vv model:  test: {rmse2} with C={c2} & k={k}')
    #print(f'The RMSE for the vv model: train:{rmse_train2} val: {rmse_val2} test: {rmse2} with C={c2} & k={k}')

    

def do_4c(c1,X_shuf_val, y_shuf_val,X_shuf_train,y_shuf_train):

    # train the model on train l
    ww, rr = np.linalg.lstsq(X_shuf_train[:,-c1:], y_shuf_train, rcond=None)[0:2]
    pred1 = X_shuf_val[:,-c1:] @ ww
    rmse1 = y_shuf_val - pred1

    print(len(rmse1))
    plt.hist(rmse1,bins=1000)
    
    

