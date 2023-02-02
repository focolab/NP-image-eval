import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import scipy.optimize
import os

def maha_dist(data, mu, sigma):

    data_mu = data-mu
    inv_sigma = np.linalg.inv(sigma)
    left_data = np.dot(data_mu, inv_sigma)
    mahal = np.square(np.dot(left_data, data_mu.T))
    
    return mahal
    
def get_ganglia(ganglia, xyz_mu, atlas_neurons, v1, v2):
    df_gangl = pd.read_csv(ganglia, index_col=0)
    df_atlas = pd.DataFrame(xyz_mu, columns = ['X mu','Y mu','Z mu'])

    # find the LR paired neurons and assign neuron_class
    all_neurons = atlas_neurons
    neuron_class, is_LR, is_L, is_R = [], [], [], []
    for i in range(len(atlas_neurons)):
        ID = atlas_neurons[i]
        if ID[-1] in ['L', 'R'] and ID[:-1]+'L' in all_neurons and ID[:-1]+'R' in all_neurons:        
            neuron_class.append(ID[:-1])
            is_LR.append(1)
            if ID[-1] == 'L':
                is_L.append(1)
                is_R.append(0)
            if ID[-1] == 'R':
                is_R.append(1)
                is_L.append(0)
        else:
            neuron_class.append(ID)
            is_LR.append(0)
            is_L.append(0)
            is_R.append(0)

    df_atlas['neuron_class'] = neuron_class
    df_atlas['is_LR'] = is_LR
    df_atlas['is_L'] = is_L
    df_atlas['is_R'] = is_R

    # add ganglion column
    gang_dict = dict(zip(df_gangl['neuron_class'].values, df_gangl['ganglion'].values))
    df_atlas['ganglion'] = [gang_dict.get(k, 'other') for k in df_atlas['neuron_class']]  

    df_xyz_atlas = convert_coordinates(df_atlas, v1, v2, [0,0,0])

    return df_xyz_atlas

def get_atlas(matfile, body='head'):
    
    eng = matlab.engine.start_matlab()
    atlas_file = eng.load(matfile)
    atlas = eng.getfield(atlas_file, 'atlas')
    body = eng.getfield(atlas, body)
    model = eng.getfield(body, 'model')
    mu = eng.getfield(model, 'mu')
    sigma = eng.getfield(model, 'sigma')
    neurons = eng.getfield(body, 'N')
    eng.quit()

    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    xyz_mu = mu[:,0:3]
    rgb_mu = mu[:,3:7]

    xyz_sigma = sigma[0:3, 0:3, :]
    rgb_sigma = sigma[3:7, 3:7, :]

    atlas_neurons = np.asarray(neurons)

    return xyz_mu, xyz_sigma, rgb_mu, rgb_sigma, atlas_neurons

def process_atlas(atlas_file = 'NeuroPAL_data_zenodo/atlas_xx_rgb.mat', ganglia = 'NeuroPAL_data_zenodo/neuron_ganglia.csv'):

    xyz_mu, xyz_sigma, rgb_mu, rgb_sigma, atlas_neurons = get_atlas(atlas_file)

    # basis vectors for new coord system (hand positioned by trial/errror)
    v1 = np.asarray([[-40, 0, 0], [80, 0, -8]])
    v2 = np.asarray([[-40, 0, 0], [-40.8, 0, -12]])

    df_xyz_atlas = get_ganglia(ganglia, xyz_mu, atlas_neurons, v1, v2)

    #TODO: return dictionary with Neuron name as key, mu, sigmas as 

    return df_xyz_atlas, xyz_sigma, rgb_mu, rgb_sigma, atlas_neurons


if __name__ == '__main__':
    # run initially on 2022-02-12-w01-NP1
    file = '2022-02-12-w01-NP1'

    # open file and get neurons
    autID = pd.read_csv('NP_trunc/'+file+'/autoID_output_'+file+'.csv')
    GTID = pd.read_csv('NP_trunc/'+file+'/blobs.csv')

    autID = autID.fillna('')
    autID.insert(0, 'ID', GTID['ID'])

    # process atlas to get mu and sigma
    xyz_mu, xyz_sigma, rgb_mu, rgb_sigma = process_atlas()

    for index, row in autID.iterrows():
        autoID = row['most_likely_ID']
        accuracy = -1
        if not row['ID'].isnull():
            if row['most_likely_ID']==row['ID']:
                accuracy =1
            else:
                accuracy =0

        xyz = 
        rgb = 

        xyz_dist = ma
        

        if index==0:




    # calculate Maha XYZ, RGB maha distance for each neuron

    # plot both points for each neuron, indicate if assign was accurate or not
