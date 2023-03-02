import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import process.file as fil
import utils.utils as uti

def summarize(folder, atlas, type='FOCO'):
    alignvals = []
    accvals = []
    names = []
    corr_dict = {}
    ID_dict = {}

    folders = os.listdir('data/'+folder)

    fig, ax = plt.subplot_mosaic([["A","B","C"],["D","D","D"]])

    for f in folders:

        if not os.path.isdir('data/'+folder + '/'+f):
            continue 
        # want to plot accuracy vs alignment cost (both xyz only and xyzrgb)
        
       
        if type == 'FOCO':
            df_data = fil.proc_FOCO('data/'+folder + '/'+f)

        elif type == 'chaud':
            df_data = fil.proc_Chaud('data/'+folder+'/'+f)


        cost_xyz, cost_rgb = uti.calc_costs(atlas.df, atlas.sigma, df_data)

        IDd, correctID, correctfirstsecond, correct_df, correcttop2 = uti.check_accuracy(df_data)

        corrIDs = np.asarray(correct_df['ID'])
        for txt in corrIDs:
            if txt not in corr_dict.keys():
                corr_dict[txt] = 1
            else:
                corr_dict[txt] += 1

        IDs = np.asarray(df_data['ID'])
        for txt in IDs:
            if txt not in ID_dict.keys():
                ID_dict[txt] = 1
            else:
                ID_dict[txt] +=1

        df_neurons = check_neurons(atlas.neur_dict, df_data, plot=False)


        alignvals.append([cost_xyz, cost_rgb, correctID])
        accvals.append([IDd, correctID, correctfirstsecond])
        names.append(f)

        unID_df = df_neurons[df_neurons['acc']=='gray']
        corr_df = df_neurons[df_neurons['acc']=='green']
        incorr_df = df_neurons[df_neurons['acc']=='red'] 

        ax["A"].scatter(np.asarray(unID_df['xyz_cost']), np.asarray(unID_df['rgb_cost']), c='gray', label='UnIDd')
        ax["B"].scatter(np.asarray(corr_df['xyz_cost']), np.asarray(corr_df['rgb_cost']), c='green', label='Correctly IDd')
        ax["C"].scatter(np.asarray(incorr_df['xyz_cost']), np.asarray(incorr_df['rgb_cost']), c='red', label= 'Incorrectly IDd')

    
    for key in corr_dict.keys():
        corr_dict[key] = corr_dict[key]/ID_dict[key]

    alignvals = np.asarray(alignvals)
    accvals = np.asarray(accvals)


    ax["D"].scatter(alignvals[:,0], alignvals[:,2], label = 'xyz alignment cost')
    ax["D"].scatter(alignvals[:,1], alignvals[:,2], label = 'rgb alignment cost')
    ax["D"].scatter(np.mean(alignvals[:,0]), np.mean(alignvals[:,2]), label = 'average xyz alignment and accuracy')
    ax["D"].scatter(np.mean(alignvals[:,1]), np.mean(alignvals[:,2]), label = 'average rgb alignment and accuracy')
    ax["D"].legend()
    ax["D"].set_xlabel('Alignment cost - mahalanobis distance')
    ax["D"].set_ylabel('Assignment accuracy')
    ax["A"].set_ylabel('RGB alignment cost')
    ax["B"].set_xlabel('XYZ alignment cost')
    ax["A"].set_title('unIDd neurons')
    ax["B"].set_title('correctly labeled neurons')
    ax["C"].set_title('incorrectly labeled neurons')
    ax["D"].set_title(folder)

    for i, txt in enumerate(names):
        ax["D"].annotate(txt, (alignvals[i,0], alignvals[i,2]), size=6)

    fig.tight_layout()
    plt.show()

def check_neurons(neurons, df_data, plot=True, name=None):
    '''
    Want bar chart to plot each neurons xyz cost, rgb cost and whether or not
    it was accurately labeled or not
    '''

    df_neurons = pd.DataFrame()

    Ids = []
    accs = []
    cost_xyzs = []
    cost_rgbs = []

    for i, row in df_data.iterrows():
        if pd.isnull(row['autoID_1']):
            continue
        ID = row['autoID_1']
        if pd.isnull(row['ID']):
            acc = 'gray'
        elif row['ID'] == row['autoID_1']:
            acc = 'green'
        else: 
            acc = 'red'

        cost_xyz = uti.maha_dist(np.asarray(row[['X', 'Y', 'Z']]), neurons[ID]['xyz_mu'], neurons[ID]['xyz_sigma'])
        cost_rgb = uti.maha_dist(np.asarray(row[['R', 'G', 'B']]),neurons[ID]['rgb_mu'], neurons[ID]['rgb_sigma'])

        df_neurons = df_neurons.append({'ID':ID, 'acc':acc, 'xyz_cost': cost_xyz, 'rgb_cost':cost_rgb}, ignore_index=True)


        Ids.append(ID)
        accs.append(acc)
        cost_xyzs.append(cost_xyz)
        cost_rgbs.append(cost_rgb)

    if plot:

        fig, ax = plt.subplots()
        ax.scatter(np.asarray(df_neurons['xyz_cost']), np.asarray(df_neurons['rgb_cost']), c=np.asarray(df_neurons['acc']))

        #for i, row in df_neurons.iterrows():
        #    ax.annotate(row['ID'], (row['xyz_cost'], row['rgb_cost']), size=8)


        ax.set_ylabel('RGB alignment cost')
        ax.set_xlabel('XYZ alignment cost')
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 1000)
        if name:
            ax.set_title('Alignment cost per neuron: '+name)
        else:
            ax.set_title('Alignment cost per neuron')
        ax.legend()

        plt.show()

    return df_neurons

def comp_acc():
    '''
    Currently set up to compare cropped, color_set and normal approaches
    '''

    df_accs = pd.DataFrame()

    for folder in os.listdir('data/NP_FOCO_cropped'):
        if not folder[0:2] == '20':
            continue 

        df_norm = fil.proc_FOCO('data/NP_FOCO_cropped/'+folder)
        df_cropped = fil.proc_FOCO('data/NP_FOCO_hist/'+folder)
        df_color_set = fil.proc_FOCO('data/NP_FOCO_set_color/'+folder)

        per_ID, per_correct, per_top2, correctID, correcttop2 = uti.check_accuracy(df_norm)
        per_ID_crop, per_correct_crop, per_top2_crop, correctID_crop, correcttop2_crop = uti.check_accuracy(df_cropped)
        per_ID_color, per_correct_color, per_top2_color, correctID_color, correcttop2_color = uti.check_accuracy(df_color_set)

        df_accs = df_accs.append({'Name':folder[:-4], 'Percent IDd':per_ID, 'Percent correct':per_correct, 'Percent correct cropped':per_correct_crop, 'Percent correct color set':per_correct_color}, ignore_index=True)

    fig, ax = plt.subplots(2,1)

    labels = np.asarray(df_accs['Name'])
    per_IDs = np.asarray(df_accs['Percent IDd'])
    per_corrects = np.asarray(df_accs['Percent correct'])
    per_corr_crops = np.asarray(df_accs['Percent correct cropped'])
    per_corr_colors = np.asarray(df_accs['Percent correct color set'])

    x = np.arange(len(labels))
    width = 0.6

    rects1 = ax[0].bar(x-width/3, per_corrects, width = width/3, label ='Percent correct original data')
    rects2 = ax[0].bar(x, per_corr_crops, width =width/3, label = 'Percent correct cropped data')
    rects3 = ax[0].bar(x+width/3, per_corr_colors, width =width/3, label= 'Percent correct color set data')
    ax[0].set_ylabel('Percent accuracy')
    ax[0].set_title('Comparing accuracy from different data processing approaches')
    ax[0].legend()

    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)
    ax[0].tick_params(labelrotation=45)
    ax[0].legend()

    ax[1].bar(labels, per_IDs, width =0.4, label='Percent IDd', color='gray')
    ax[1].legend()
    ax[1].tick_params(labelrotation=45)

    plt.tight_layout()

    plt.show()