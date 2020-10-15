from PIL import Image
import csv
from scipy.signal import convolve2d
import numpy as np
from tqdm import tqdm

def getEpScenValbyRec(filename):

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        numExamples = 0
        epi_scen = []
        heights = []
        for row in reader:
            isValid = row['Val'] #V or I are the first element of the list thisLine
            #valid_user = int(row['VehicleArrayID'])
            if isValid == 'V': #check if the channel is valid
                numExamples = numExamples + 1
                epi_scen.append([int(row['EpisodeID']),int(row['SceneID'])])
                heights.append(row['z'])
            lastEpisode = int(row['EpisodeID'])

    return numExamples, lastEpisode, epi_scen, heights


def getCoord(filename, limitEp):

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        coordinates_train = []
        coordinates_test = []

        for row in reader:
            isValid = row['Val'] #V or I are the first element of the list thisLine
            if isValid == 'V': #check if the channel is valid
                if int(row['EpisodeID']) <= limitEp:
                    coordinates_train.append([float(row['x']),float(row['y'])])
                if int(row['EpisodeID']) > limitEp:
                    coordinates_test.append([float(row['x']),float(row['y'])])

    return coordinates_train, coordinates_test

def save_img(sample,name):
    #############################################
    image_to_save = np.zeros((sample.shape[0],sample.shape[1],3),dtype=np.float32)
    for r in range(sample.shape[0]):
        for c in range(sample.shape[1]):
            if sample[r,c] == 0:
                image_to_save[r,c,:] = (255,255,255)
            elif sample[r,c] == 1:
                image_to_save[r,c,:] = (255,0,0)
            elif sample[r,c] ==2:
                image_to_save[r,c,:] = (255,128,0)
            elif sample[r,c] ==3:
                image_to_save[r,c,:] = (51,153,255)

    image_to_save= image_to_save.astype('uint8')

    img = Image.fromarray(image_to_save,mode='RGB')
    img.save(name+'.png')
    #############################################




csv_file = '/home/batool/beam_selection/image/CoordVehiclesRxPerScene_s009.csv'

nSamples, lastEpisode, epi_scen_list, heights = getEpScenValbyRec(csv_file)
Dict = {'1.59':2,'4.3':3,'3.2':1}
inputs = np.zeros([nSamples,101,185])
print(nSamples,epi_scen_list)
limit = 0

for samp in tqdm(range(0,nSamples)):
    epi_scen = epi_scen_list[samp]
    imgURL = '/home/batool/beam_selection/image/npys/'+'{:0>1}'.format(epi_scen[0])+'.npy'
    img = np.load(imgURL)
    img[img!=Dict[heights[samp]]] = 0
    inputs[samp,:] = img
    print(inputs.shape)

    input_test = inputs[limit:]
    #input_train =  inputs[:limit]

np.savez('img_input_test.npz',inputs=input_test)

"""
print('saving train set')
np.savez('img_input_train.npz',inputs=input_train)
print('saving validation set')
np.savez('img_input_validation.npz',inputs=input_validation)
"""
