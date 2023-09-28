
import os, sys, pathlib
from pprint import pprint
import gc
import pickle
from importlib import reload
import logging, warnings
logging.basicConfig(level=logging.ERROR)

import pandas as pd
import numpy as np
import sklearn
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import scipy.linalg as linalg

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection
# %matplotlib inline
from monkey.defs import *
import pyaldata as pyal

'''
This code is referenced from https://github.com/BeNeuroLab/2022-preserved-dynamics, used for obtaining example raster 
graphs
'''

try:
    nbPath = pathlib.Path.cwd()
    RepoPath = nbPath.parent
    os.chdir(RepoPath)

    from tools import utilityTools as utility
    from tools import dataTools as dt
    import params
    defs = params.monkey_defs

    set_rc =  params.set_rc_params
    set_rc()
    # root = params.root
    root = pathlib.Path(nbPath/"data")
finally:
    os.chdir(nbPath)

def plot_colored_line(ax, x, y, colors=None):
    """
    based on this:
    https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html?highlight=multicolored_line
    """
    if colors is None:
        colors = utility.get_colors(x.shape[0])

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, colors=colors)

    lc.set_array(y)
    line = ax.add_collection(lc)
    ax.set_ylim([y.min(), y.max()])
    ax.set_xlim([x.min(), x.max()])

    return ax


def plot_3d_colored_line(ax, x, y, z, colors=None, **kwargs):
    """
    based on this:
    https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html?highlight=multicolored_line
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if colors is None:
        colors = utility.get_colors(x.shape[0])

    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = Line3DCollection(segments, colors=colors, **kwargs)

    lc.set_array(z)
    line = ax.add_collection(lc)
    ax.set_ylim([y.min(), y.max()])
    ax.set_xlim([x.min(), x.max()])
    ax.set_zlim([z.min(), z.max()])

    return ax

# raster_example = ('Chewie_CO_FF_2016-09-15.mat', 'Mihili_CO_CS_2015-05-11.mat', 'Jaco_CO_CS_2016-01-28.mat')
raster_example = ('Chewie_CO_FF_2016-09-15.mat', 'Mihili_CO_CS_2015-05-11.mat')
raster_example_df = []
for session in raster_example:
    path = root/session.split('_')[0]/session
    df = defs.prep_general(dt.load_pyal_data(path))
    df = pyal.restrict_to_interval(df, epoch_fun=defs.exec_epoch)
    raster_example_df.append(df)

min_units = min([df.MCx_rates[0].shape[1] for df in raster_example_df])

import matplotlib
matplotlib.use('TkAgg')

fig = plt.figure(figsize=(10,6))
gs1   =fig.add_gridspec(nrows=1, ncols=8, left=0.1, bottom=0.4, right=.48, top=.8)
gs2   =fig.add_gridspec(nrows=1, ncols=8, left=.52, bottom=0.4, right=.9, top=.8)
gs = [gs1,gs2]
gs_c = fig.add_gridspec(nrows=1, ncols=1, left=.945, bottom=0.5, right=.96, top=.7)
cax = fig.add_subplot(gs_c[:])

trial=12
axes = []
for i,df in enumerate(raster_example_df):
    data = []
    for tar in range(8):
        df_ = pyal.select_trials(df, df.target_id==tar)
        data.append(df_.MCx_rates[trial][:,:min_units])
    data = np.array(data)
    vmin = np.amin(data, axis= (0,1))
    vmax = np.amax(data, axis= (0,1))

    for j,tarData in enumerate(data):
        ax = fig.add_subplot(gs[i][j])
        axes.append(ax)
        tarData -= vmin
        tarData /= (vmax - vmin)
        ax.imshow(tarData.T, aspect='auto')
        ax.tick_params('both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

axes[0].set_ylabel(f'Units ($n={min_units}$)')
fig.colorbar(cm.ScalarMappable(),cax=cax, ticks=(0,1),drawedges=False)
cax.set_title('Normalised\nFR')
ax = utility.phantom_axes(fig.add_subplot(gs1[:]))
ax.set_title('Monkey1')
ax = utility.phantom_axes(fig.add_subplot(gs2[:]))
ax.set_title('Monkey2')

#========================
gs1   =fig.add_gridspec(nrows=1, ncols=8, left=0.1, bottom=0.32, right=.48, top=.39)
gs2   =fig.add_gridspec(nrows=1, ncols=8, left=.52, bottom=0.32, right=.9, top=.39)
gs = [gs1,gs2]

axes = []
for i,df in enumerate(raster_example_df):
    for tar in range(8):
        df_ = pyal.select_trials(df, df.target_id==tar)
        data = df_.vel[trial]
        ax = fig.add_subplot(gs[i][tar])
        axes.append(ax)
        ax.plot(data[:,0], label='$X$')
        ax.plot(data[:,1], label='$Y$')
        ax.tick_params('both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

axes[0].set_ylabel('Velocity',fontdict={'size': 8})
axes[-1].legend(frameon=False, loc=(1.5,0.04))
axes[0].set_xlabel('Time rel. movement onset',loc='left',fontdict={'size': 8})
fig.savefig('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/monkey-FR-example-1.pdf',
            format='pdf', bbox_inches='tight')
#fig.savefig(params.figPath / 'monkey-FR-example-1.pdf', format='pdf', bbox_inches='tight')

# AllData = dt.get_data_array(raster_example_df, area='M1', model=10)