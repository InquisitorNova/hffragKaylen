"""
*Filename: SecondaryVertexDeepSetTrainer
*Description: This jupyter notebook is an extension of the ProjectorTrainer, it trains the same
*residual deep sets neural network architecture as the projector trainer with the additional
*features as well. In addition it adds the secondary vertex displacement of the b_jets as an
*additional target for the network to converge to.
Date: 16/02/2023
Author: Kaylen Smith Darnbrook
"""
# Import relevant modules
import os
import numpy as np
import seaborn as sns
import wandb
#from wandb.keras import WandbCallback
from keras import callbacks
import keras
import DeepSetNeuralNetArchitecture2 as DSNNA
from DeepSetNeuralNetArchitecture import LogNormal_Loss_Function
from DeepSetNeuralNetArchitecture import Mean_Squared_Error
from HffragDeepSetsProjectionMultivariateAltered5 import DeepSetsProjection
from sklearn.feature_selection import mutual_info_regression
import keras.backend as k
import uproot
import awkward as ak
import sklearn as sk
from numpy.lib.recfunctions import structured_to_unstructured
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.utils.vis_utils import plot_model
from timeit import default_timer as timer
import matplotlib.pyplot as plt

#Format the style and format of the matplotlib plots
plt.style.use("default")
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rc('text',usetex = False)
plt.rc('font',family = 'Times New Roman')

particles_input_scaled = np.load("/home/physics/phujdj/DeepLearningParticlePhysics/particles_input_scaledA3.npy")
re_b_jets_input_scaled = np.load("/home/physics/phujdj/DeepLearningParticlePhysics/re_b_jets_input_scaledA3.npy")
bhads_targets = np.load("/home/physics/phujdj/DeepLearningParticlePhysics/bhad_targetsA3.npy")
onehot = np.load("/home/physics/phujdj/DeepLearningParticlePhysics/onehotA3.npy")

X1_train = particles_input_scaled[:10]
y_train = bhads_targets[:10]
print("X1_train is", X1_train.shape) 
print("0")
array = X1_train
b = np.zeros(array.shape)
print("1")
sorted_idx = np.argsort(-array, axis = 1, kind = 'stable')
print("2")
for i in range(array.shape[0]):
    for j in range(array.shape[1]):
        X1_train[i,j] = array[i, sorted_idx[i,j,7]]
    print("3")
from sklearn.feature_selection import mutual_info_regression
miscores=[]
print("hello")
'''for y in range(y_train.shape[-1]):
    miscores.append(mutual_info_regression(y_train, y_train[:,y]))
print(numpy.max(miscores))'''
summiscores = miscores
#summiscores=np.load("MI Scores Saliency.npy")
summiscores = []
print(np.shape(X1_train))
for y in range(y_train.shape[-1]):
    miscores=[]
    for x in range(X1_train.shape[-1]):
        miscores.append(mutual_info_regression(X1_train[:,:,x], y_train[:,y]))
        print(x)
    summiscores.append(np.average(miscores, axis=0))
    print(y)

np.save("/home/physics/phujdj/DeepLearningParticlePhysics/MIScoresSaliency", summiscores)


print(np.shape(miscores))
print(np.shape(summiscores))


np.save("/home/physics/phujdj/DeepLearningParticlePhysics/MIScoresSaliency+Jets", summiscores)

features = [r'$p_x^{track}$', r'$p_y^{track}$', r'$p_z^{track}$', r'$p_T^{track}$', r'$\eta^{track}$', r'$\phi^{track}', r'$z_0\sin{\theta}$', r'$s_{d0}$', r'$d_0$', r'$s_{d0}^{PV}$', r'$d_0^{PV}', r'$\log{\frac{p_T^{track}}{p_T^{jet}}}', r'$\Delta R$', r'$\log{\Delta R}$', ]
#targets = [r'$p_x^b$', r'$p_y^b$', r'$p_z^b$', r'$p_T^b$', r'$p^b$', r'$E^b$', r'$\frac{p_T^{B}}{p_T^{jet}}$', r'$\frac{\vec{p_{B}} \cdot \vec{p_{jet}}}{|\vec{p_{jet}}|^2}$', r'$\frac{|\vec{p_B} \times \vec{p_{jet}}|}{|\vec{p_{jet}}|}$']

fig, ax = plt.subplots()
im = ax.imshow(summiscores, cmap = 'hot', origin = 'lower', aspect = 'auto')
plt.colorbar(im)
ax.set(ylabel = 'Targets', xlabel = 'Features')
plt.suptitle("Feature Importance to Targets")
plt.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/SaliencyHeatmapTracks')
plt.tight_layout()
plt.close()
