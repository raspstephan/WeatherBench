import pickle
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import sys
import pdb
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter

def in_notebook():
    """
    Returns ``True`` if the module is running in IPython kernel,
    ``False`` if in IPython shell or other Python shell.
    """
    return 'ipykernel' in sys.modules

if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def to_pickle(obj, fn):
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)
def read_pickle(fn):
    with open(fn, 'rb') as f:
        return pickle.load(f)

def plot_hist(h, ax=None, ylim=None, name='', train=True, valid=True, **kwargs):
    if ax is None: fig, ax = plt.subplots()
    if train: ax.plot(h['loss'], label=f'{name}', **kwargs)
    if valid: ax.plot(h['val_loss'], **kwargs)
    ax.legend()
    if ylim is not None: ax.set_ylim(ylim)


def plot_losses(path, exp_ids, plot_lrs=True, ylim=None, log=False):
    exp_ids = [str(exp_id) for exp_id in exp_ids]
    fig, ax1 = plt.subplots(2 if plot_lrs else 1, 1, figsize=(15, 15 if plot_lrs else 7))
    if plot_lrs: ax1, ax2 = ax1
    colors = sns.palettes.color_palette(n_colors=len(exp_ids))
    for exp_id, c, in zip(exp_ids, colors):
        fn = glob(f'{path}{exp_id}*.pkl')[0]
        name = fn.split('/')[-1].split('_history.pkl')[0]
        h = read_pickle(fn)
        plot_hist(h, ax1, name=name, valid=False, c=c, lw=2)
        plot_hist(h, ax1, name=name, train=False, c=c, ls='--', lw=2)


        if plot_lrs:
            ax2.plot(h['lr'], c=c)

    ax1.set_ylim(ylim)
    if log: ax1.set_yscale('log')


def plot_rmses(rmse, var, save_fn=None, ax=None, legend=False):
    # Color settings
    c_lri = '#ff7f00'
    c_lrd = '#ff7f00'
    c_cnni = '#e41a1c'
    c_cnnd = '#e41a1c'
    c_tigge = '#984ea3'
    c_t42 = '#4daf4a'
    c_t63 = '#377eb8'
    c_persistence = '0.2'
    c_climatology = '0.5'
    c_weekly_climatology = '0.7'

    if ax is None: fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    rmse[var + '_persistence'].plot(c=c_persistence, label='Persistence', lw=3, ax=ax)
    ax.axhline(rmse[var + '_climatology'], ls='--', c=c_climatology, label='Climatology', lw=3)
    ax.axhline(rmse[var + '_weekly_climatology'], ls='--', c=c_weekly_climatology, label='Weekly clim.', lw=3)
    rmse[var + '_t42'].plot(c=c_t42, label='IFS T42', lw=3, ax=ax)
    rmse[var + '_t63'][::2].plot(c=c_t63, label='IFS T63', lw=3, ax=ax)
    rmse[var + '_tigge'].plot(c=c_tigge, label='Operational', lw=3, ax=ax)
    #     rmse[var+'_lr_6h_iter'].plot(c=c_lri, label='LR (iterative)', lw=3, ax=ax)
    #     ax.scatter([3*24], [rmse[var+'_lr_3d']], c=c_lrd, s=100, label='LR (direct)', lw=2, edgecolors='k', zorder=10)
    #     ax.scatter([5*24], [rmse[var+'_lr_5d']], c=c_lrd, s=100, lw=2, edgecolors='k', zorder=10)
    #     rmse[var+'_cnn_6h_iter'].plot(c=c_cnni, label='CNN (iterative)', lw=3, ax=ax)
    #     ax.scatter([3*24], [rmse[var+'_cnn_3d']], c=c_cnnd, s=100, label='CNN (direct)', lw=2, edgecolors='k', zorder=10)
    #     ax.scatter([5*24], [rmse[var+'_cnn_5d']], c=c_cnnd, s=100, lw=2, edgecolors='k', zorder=10)

    if var == 'z':
        ax.set_ylim(0, 1200)
        ax.set_ylabel(r'Z500 RMSE [m$^2$ s$^{-2}$]')
        ax.set_title('a) Z500')
    elif var == 't':
        ax.set_ylim(0, 6)
        ax.set_ylabel(r'T850 RMSE [K]')
        ax.set_title('b) T850')

    if legend: ax.legend(loc=2, framealpha=1)
    ax.set_xlim(0, 122)
    ax.set_xticks(range(0, 121, 24))
    ax.set_xticklabels(range(6))
    ax.set_xlabel('Forecast time [days]')

    if not save_fn is None:
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
        fig.savefig(save_fn)

from src.clr import LRFinder
def find_lr(model, dg, **kwargs):
    lrf = LRFinder(dg.n_samples, dg.batch_size, verbose=0, **kwargs)
    model.fit(dg, epochs=1, callbacks=[lrf])
    return lrf

import matplotlib.ticker as ticker
def plot_lrf(lrf, xlim=None, ylim=None, log=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.plot(10**lrf.lrs, lrf.losses)
    plt.xlabel('lr'); plt.ylabel('loss')
    if log: plt.yscale('log')
    if ylim is not None: plt.ylim(ylim)
    if xlim is not None: plt.xlim(xlim)
    x_labels = ax.get_xticks()
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))

def pad_periodic(arr, p):
    arr = np.pad(arr, ((0, 0), (p, p), (0, 0)), mode='wrap')
    arr = np.pad(arr, ((p, p), (0, 0), (0, 0)), mode='edge')
    return arr

def compute_las(da, k, gauss_std=None, omit_idxs=[]):
    """Assume da = [lat, lon, level]"""
    da_out = da.copy().load()
    p = (k - 1) // 2
    arr = pad_periodic(da_out, p)
    arr = convolve(arr, np.ones((k, k, 1))/(k**2), mode='valid')
    if gauss_std is not None:
        arr = np.concatenate([gaussian_filter(arr[..., i], gauss_std) for i in range(arr.shape[-1])], -1)
    da_out[:] = arr
    return da_out