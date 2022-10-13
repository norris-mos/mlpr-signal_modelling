import numpy as np
import matplotlib.pyplot as plt


def do_2a(X_shuf_train, y_shuf_train):
    def phi_quartic(t):
        return np.asarray([1, t, t ** 2, t ** 3, t ** 4])

    t_grid = np.arange(0,1,0.05)

    # plot example
    plt.plot(t_grid, X_shuf_train[0],'r+', label = 'X')
    plt.plot(1, y_shuf_train[0],'b.', label = "y")
    plt.xlim(0, 1.1)

    # perform linear fit
    X_lin = np.vstack([t_grid,np.ones(len(t_grid))]).T
    ww_lin = np.linalg.lstsq(X_lin, X_shuf_train[0],rcond=None)[0]

    # perform polynomial fit
    X_poly = np.asarray([phi_quartic(x) for x in t_grid])
    ww_poly  = np.linalg.lstsq(X_poly, X_shuf_train[0],rcond=None)[0]

    # add fits to plot
    # add t = 1
    t_grid_= np.append(t_grid, 1)
    X_lin = np.vstack([t_grid_,np.ones(len(t_grid_))]).T
    X_poly = np.asarray([phi_quartic(x) for x in t_grid_])

    plt.plot(t_grid_, np.sum(ww_poly * X_poly, axis=1),'g--',label='quartic fit')
    plt.plot(t_grid_, np.sum(ww_lin * X_lin, axis=1),'c--',label='linear fit')
    plt.legend()
    plt.xlabel('t steps')
    plt.ylabel('amplitude')


def do_2c(X_shuf_train, y_shuf_train, K_grid, examples=4):
    def basic_change_K(K, t):
        to_return = []
        for i in range(K + 1):
            to_return.append(t ** i)
        return np.asarray(to_return)
    
    t_grid = np.arange(0,1,0.05)
    t_grid_ = np.append(t_grid, 1)
    colors = list(mcolors.TABLEAU_COLORS.values())

    for ex in range(examples):
        plt.figure(ex)
        plt.plot(t_grid, X_shuf_train[ex], colors[0], label="X")
        plt.plot(1, y_shuf_train[ex], colors[1], label = "y")

        for i, K in enumerate(K_grid):
            X = np.asarray([basic_change_K(K, x) for x in t_grid])
            X_ = np.asarray([basic_change_K(K, x) for x in t_grid_])
            
            ww  = np.linalg.lstsq(X, X_shuf_train[ex],rcond=None)[0]            
            plt.plot(t_grid_, np.sum(ww * X_, axis=1), colors[2 + i], label=f'K={K}')
            
        
        plt.xlim(0, 1.1)
        plt.legend()
        plt.title(f"Example {str(ex)}")
    