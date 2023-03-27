import math
import torch
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt


def DGP_sin(train_x, test_x=None):
    # Generate training data
    train_y = 0.5 * torch.sin((2 * math.pi) * train_x) + \
        torch.randn(train_x.size()) * math.sqrt(0.01)
    if test_x is not None:
        test_y = 0.5 * torch.sin((2 * math.pi) * test_x)
        return train_y, test_y
    else:
        return train_y


def DGP_cubic(train_x, test_x=None):
    # Generate training data
    train_y = train_x ** 3 + torch.randn(train_x.size()) * 0.1
    if test_x is not None:
        test_y = test_x ** 3
        return train_y, test_y
    else:
        return train_y


def DGP_complex_sin(train_x, test_x=None):
    '''
        x: [-0.6, 1]
    '''
    # Generate training data
    train_y = train_x * torch.sin(10 * train_x) + 0.1 * torch.sin(
        15 * train_x) + torch.randn(train_x.size()) * math.sqrt(0.01)
    if test_x is not None:
        test_y = test_x * torch.sin(10 * test_x) + 0.1 * torch.sin(15 * test_x)
        return train_y, test_y
    else:
        return train_y


def DGP_non_stationary(train_x, test_x=None):
    # Generate training data
    train_y = torch.sin((10 * math.pi) * train_x) / (2 * train_x) + \
        (train_x - 1) ** 4 + torch.randn(train_x.size()) * 0.1
    if test_x is not None:
        test_y = torch.sin((10 * math.pi) * test_x) / \
            (2 * test_x) + (test_x - 1) ** 4
        return train_y, test_y
    else:
        return train_y


def franke(train_x):
    X, Y = train_x[:, 0], train_x[:, 1]
    term1 = .75*torch.exp(-((9*X - 2).pow(2) + (9*Y - 2).pow(2))/4)
    term2 = .75*torch.exp(-((9*X + 1).pow(2))/49 - (9*Y + 1)/10)
    term3 = .5*torch.exp(-((9*X - 7).pow(2) + (9*Y - 3).pow(2))/4)
    term4 = .2*torch.exp(-(9*X - 4).pow(2) - (9*Y - 7).pow(2))

    f = term1 + term2 + term3 - term4

    return f


def load_airline_dataset():
    f = loadmat('./Data/airlinedata.mat')

    train_x, test_x = torch.tensor(f['xtrain'].astype(
        np.float)), torch.tensor(f['xtest'].astype(np.float))
    train_y, test_y = torch.tensor(f['ytrain'].astype(
        np.float)), torch.tensor(f['ytest'].astype(np.float))

    return train_x.squeeze(), test_x.squeeze(), train_y.squeeze(), test_y.squeeze()


def load_co2_dataset():
    f = loadmat('./Data/CO2data.mat')

    train_x, test_x = torch.tensor(f['xtrain'].astype(
        np.float)), torch.tensor(f['xtest'].astype(np.float))
    train_y, test_y = torch.tensor(f['ytrain'].astype(
        np.float)), torch.tensor(f['ytest'].astype(np.float))

    return train_x.squeeze(), test_x.squeeze(), train_y.squeeze(), test_y.squeeze()


def get_test_fcnt(n_type, n_pts=None, seed=1234, plot_on = True):
    torch.random.manual_seed(seed)
    n_grid = None
    if n_type == 1:  # 20
        if n_pts is None:
            n_pts = 10
        train_x = torch.linspace(-1, 1, n_pts)
        test_x = torch.linspace(-1, 1, 100)
        train_y, test_y = DGP_sin(train_x, test_x)
        train_x.squeeze()

    elif n_type == 2:  # 30
        if n_pts is None:
            n_pts = 10
        train_x = torch.linspace(-0.6, 1, n_pts)
        test_x = torch.linspace(-0.6, 1, 100)
        train_y, test_y = DGP_complex_sin(train_x, test_x)

    elif n_type == 3:  # 30
        if n_pts is None:
            n_pts = 10
        train_x = torch.linspace(-3, 3, n_pts)
        test_x = torch.linspace(-3, 3, 100)
        train_y1, test_y1 = DGP_sin(train_x, test_x)
        train_y2, test_y2 = DGP_cubic(train_x, test_x)
        train_y, test_y = train_y1 * 10 + train_y2, test_y1 * 10 + test_y2

    elif n_type == 4:  # 30
        if n_pts is None:
            n_pts = 10
        train_x = torch.linspace(0.5, 3.0, n_pts)
        test_x = torch.linspace(0.5, 3.0, 100)
        train_y, test_y = DGP_non_stationary(train_x, test_x)

    elif n_type == 5:  # 20
        if n_pts is None:
            n_pts = 20
        # Generate training data using LHS design
        import varstool.sampling.symlhs as symlhs
        train_x = torch.tensor(symlhs(
            sp=n_pts, params=2, seed=seed,
            criterion='maximin', iterations=50))
        train_y = franke(train_x)

        # Generate training data using fullfactorial design with level of 10 per each factors
        xv, yv = torch.meshgrid(torch.linspace(
            0, 1, 10), torch.linspace(0, 1, 10), indexing="ij")
        test_x = torch.cat((
            xv.contiguous().view(xv.numel(), 1),
            yv.contiguous().view(yv.numel(), 1)),
            dim=1)
        test_y = franke(test_x)
        n_grid = int(test_y.numel() ** 0.5)

    elif n_type == 6:
        train_x, test_x, train_y, test_y = load_airline_dataset()

    elif n_type == 7:
        train_x, test_x, train_y, test_y = load_co2_dataset()

    train_x, train_y, test_x, test_y = train_x.float(), train_y.float(), test_x.float(), test_y.float()

    # plotting
    if plot_on:
        plt.figure(figsize=(4, 3), dpi=200)
        if n_grid is None:
            plt.plot(train_x, train_y, 'b*', label='observation')
            plt.plot(test_x, test_y, 'w:', label='Target fcnt', linewidth=2)
            plt.grid(color='lightgray')
            plt.legend(fontsize=7)
            plt.show()
        else:

            plt.plot(train_x[:, 0], train_x[:, 1], 'b*', label='observation')
            CS = plt.contour(test_x[:, 0].reshape(n_grid, n_grid), test_x[:, 1].reshape(
                n_grid, n_grid), test_y.reshape(n_grid, n_grid), colors='w')
            plt.clabel(CS, inline=1, fontsize=10)
            plt.show()

    return train_x, train_y, test_x, test_y
