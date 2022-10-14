from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def get_and_describe_data() -> np.ndarray:
    to_return = np.load(Path(__file__).parent.parent / "data/amp_data.npz")['amp_data']
    print("Type: ", str(type(to_return)))
    print("Shape: ", to_return.shape)
    print("Min: ", to_return.min())
    print("Max: ", to_return.max())

    return to_return

def do_1a_plot_line_and_hist(amp_data, sz = (10, 5)):
    """_summary_

    Args:
        amp_data (np.ndarray): The data.
        sz (tuple, optional): Figure size. Defaults to (10, 5).

    Returns:
        List[plt.Figure]: The line chart and histogram.
    """
    f_line = plt.figure()

    # line
    axarr = f_line.add_subplot(1, 1, 1)
    plt.plot(np.arange(0, len(amp_data), 1), amp_data)
    plt.xlabel('index')
    plt.ylabel('amplitude')
    plt.ylim(-0.8, 0.8)

    f_hist = plt.figure()

    # hist
    axarr = f_hist.add_subplot(1, 1, 1)
    plt.hist(amp_data, bins=100)
    
    return [f_line, f_hist]

def do_1b_amp_data_to_splits(amp_data, seed = 64):
    # force data to be multiple of 21 in length
    amp_data = amp_data[:-6]

    # reshape
    reshaped = np.reshape(amp_data, (1605394, 21))

    if not np.allclose(reshaped[0], amp_data[:21], rtol=1e-010):
        raise RuntimeError("Reshape process has gone wrong")

    # set seed
    np.random.seed(seed)
    shuffled = np.random.permutation(reshaped)
    train, val, test = np.split(shuffled, [int(0.7*len(shuffled)),int(0.85*len(shuffled))])
    X_shuf_train, y_shuf_train = train[:,:-1],train[:,-1]
    X_shuf_val, y_shuf_val = val[:,:-1],val[:,-1]
    X_shuf_test, y_shuf_test = test[:,:-1],test[:,-1]

    return (
        X_shuf_train, 
        y_shuf_train, 
        X_shuf_val, 
        y_shuf_val,
        X_shuf_test, 
        y_shuf_test
    )

def check_1b_result(
    X_shuf_train_l,
    y_shuf_train_l,
    X_shuf_val_l,
    y_shuf_val_l,
    X_shuf_test_l, 
    y_shuf_test_l,
    amp_data_l
):
    tmp = [y_shuf_train_l, y_shuf_val_l, y_shuf_test_l]
    print("All length checks pass: ", all([
        X_shuf_train_l == y_shuf_train_l,
        X_shuf_val_l == y_shuf_val_l,
        X_shuf_test_l == y_shuf_test_l,
        (amp_data_l - 6) / 21 == sum(tmp)
    ]))
    print("Train / val / test split ", [l / sum(tmp) for l in tmp])
