import numpy as np

def Phi_row(K, t):
    return np.asarray([t ** i for i in range(K)])

def Phi(C, K):
    to_return = np.zeros((C, K))
    
    # iterate c through 19/ 20, 18 / 20, ... with C steps
    for i, c in enumerate(np.arange(19 / 20, 1 - (C / 20) - (1 / 100), -1 / 20)):
        to_return[C - i - 1, :] = Phi_row(K=K, t=c)
    
    return to_return

def make_vv(C, K):
    p = Phi(C, K)
    
    return (p @ np.linalg.inv((p.T @ p).T) @ Phi_row(K=K, t=1))

def do_3biii(X_shuf_train):
    ex = X_shuf_train[0] # get example

    # perform linear fit
    X_lin = Phi(C=20, K=2)  # equivalent to linear design matrix in Q2
    ww_lin = np.linalg.lstsq(X_lin, ex, rcond=None)[0]

    # perform polynomial fit
    X_poly = Phi(C=20, K=5)  # equivalent to quartic design matrix in Q2
    ww_poly  = np.linalg.lstsq(X_poly, ex, rcond=None)[0]

    # use v
    v_lin = make_vv(C=20, K=2)
    v_poly = make_vv(C=20, K=5)

    # compare
    print("LSTSQ Lin Fit Prediction:  ", np.dot(ww_lin, Phi_row(K=2, t=1)))
    print("LSTSQ With vv Prediction:  ", v_lin.T @ ex)
    print("")
    print("LSTSQ Poly Fit Prediction: ", np.dot(ww_poly, Phi_row(K=5, t=1)))
    print("LSTSQ Poly vv Prediction: ", v_poly.T @ ex)


def do_3ci(X_shuf_train, y_shuf_train):
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    min_c = 1
    max_c = 20
    step_c = 1

    min_k = 1
    max_k = 5
    step_k = 1

    result = []

    # iterate through context values
    for i, c in enumerate(range(min_c, max_c + 1, step_c)):
        tmp_res = []

        # iterate through poly orders
        for j, k in enumerate(range(min_k, max_k, step_k)):
            vv = make_vv(C=c, K=k)
            pred = X_shuf_train[:,-c:] @ vv
            tmp_res.append(rmse(predictions=pred, targets=y_shuf_train))
        result.append(tmp_res)

    result = np.asarray(result)

    best_c = result.argmin()
    best_k = result[best_c].argmin()

    print("Attempted C: ", list(range(min_c, max_c + 1, step_c)))
    print("Attempted K: ", list(range(min_k, max_k, step_k)))
    print("C=", best_c, "; K=", best_k)

    return result

def do_3cii():
    raise NotImplementedError()