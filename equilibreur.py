# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 03:51:47 2020

@author: sangbe SIDIBE
"""

import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KernelDensity

# =============================================================================
def load_data(filename):
    """
            Used to load data from a tdms file.

        Parameters
        ----------
        filename : String
            The path for the "tdms" file we want to load.

        Returns
        -------
        list
            A list which contains data from Sensors [1, 4].

        """
    tdms_file = TdmsFile.read(filename)
    group = tdms_file['Untitled']
    if len(group) == 4:
        channel1 = group['Untitled']
        channel2 = group['Untitled 1']
        channel3 = group['Untitled 2']
        channel4 = group['Untitled 3']
    else:
        channel1 = group['Untitled 1']
        channel2 = group['Untitled 2']
        channel3 = group['Untitled 3']
        channel4 = group['Untitled 4']
    # -- end if
    dataC1 = channel1[:]
    dataC2 = channel2[:]
    dataC3 = channel3[:]
    dataC4 = channel4[:]
    return [dataC1, dataC2, dataC3, dataC4]


# =============================================================================
def create_matrix_from_9_area_data(folder_path, mass):
    area_file = "Mesure_Z{}_{}kg.tdms"
    matrix = []
    mes_vide = load_data("Matrix\Mesure_a_vide.tdms")
    
    for i in range(9):
        d = load_data(folder_path + '\\' + area_file.format(i+1, mass))
        variation = np.asarray(d) - np.asarray(mes_vide)
        matrix.append(np.asarray(variation))
    return matrix


# =============================================================================
def compute_variation(filename1, filename2):
    data = load_data('Data\{}'.format(filename1))
    data_without_loads = load_data('Data\{}'.format(filename2))
    return (np.asarray(data) - np.asarray(data_without_loads))


# =============================================================================
def compute_euclidean_distance_from_matrix(matrix, data):
    """
    For each aera in matrix, we compute its Euclidean Distance from data loaded

    Parameters
    ----------
    matrix : List
        A list of 9 arrays.
    data : Array of float64 (4, nbData)
        Data from which we want to process.

    Returns
    -------
    List of 9 arrays 
        Array which contains the result of the Eucliden Distance's computation.

    """
    def compute_euclidean_distance(matrix, dataC1, dataC2, dataC3, dataC4):
        result = []
        for zone in matrix:    
            tmp1 = np.sqrt(np.dot(dataC1, dataC1) - 2 * np.dot(dataC1, zone[0]) + np.dot(zone[0], zone[0]))
            tmp2 = np.sqrt(np.dot(dataC2, dataC2) - 2 * np.dot(dataC2, zone[1]) + np.dot(zone[1], zone[1]))
            tmp3 = np.sqrt(np.dot(dataC3, dataC3) - 2 * np.dot(dataC3, zone[2]) + np.dot(zone[2], zone[2]))
            tmp4 = np.sqrt(np.dot(dataC4, dataC4) - 2 * np.dot(dataC4, zone[3]) + np.dot(zone[3], zone[3]))
            result.append(np.asarray([tmp1, tmp2, tmp3, tmp4]))
        # -- end for
        return result
    # -- end compute_euclidean_distance

    tab_dist = compute_euclidean_distance(matrix, data[0], data[1], data[2], data[3])
    return tab_dist
  

# =============================================================================
def plot_deformation(data, area=None, filename=None):
    """
    Used to visualize data by box plot and violin plot.

    Parameters
    ----------
    data : List of data we want to visualize
    area : List of position we want to visualize
           The default is None.
    filename: File in which data is from

    Returns
    -------
    None.

    """    
    my_dict = {}
    if area == None:
        for i in range(len(data)):
                name = 'Zone{}'.format(i+1)
                my_dict[name] = data[i]
    else:
        for z in area:
            name = 'Zone{}'.format(z)
            my_dict[name] = data[z-1]
        
    # Create the DataFrame from my_dict
    df = pd.DataFrame(my_dict)

    plt.figure()
    if len(data) == 9:
        plt.xlabel("Zones", size=12)
    else:
        plt.xlabel("Capteurs", size=15)
    plt.ylabel("Distances Euclidiennes", size=12)
    
    if filename == None:
        plt.title("Pression subie par zones les plus impactées")
    else:
        plt.title(("Box Plot: " + filename))
    
    sns.boxplot(data=df)
    plt.show()
    
    sns.violinplot(data=df)
    if filename == None:
        plt.title("repartition de la pression subie par centre de zone")
    else:
        plt.title(("Violin Plot: " + filename))
    plt.show()

# =============================================================================
def compute_position(data, draw=None):
    """    
        For each area in data, we compute its kernel density estimation (kde).
        And we create a matrix (3,3) from kde computed for each area.
        Then we sort the area from their kde and we store the 4 min area in
        the parameter area_numbrers.
        Finally, from area_numbers we set to 0 the area not selected in matrix.
        Parameters
        ----------
        data : List of 9 arrays which contains the Euclidean Distances computed.
        area_number: will be filled with the 4 most impacted areas
    
        Returns
        -------
        mat_pos : A matrix where only selected areas are not set to 0.

    """
    my_dict = {}
    my_dict2 = {}
    matrix = []
    tab = []
    ind = 0
    k = 0
    for area in data:
        name = 'Zone{}'.format(ind+1)
        kde = KernelDensity(kernel='gaussian', bandwidth=0.75).\
            fit(np.asarray(area).reshape(-1, 1))
        my_dict[name] = kde.score_samples(np.asarray(area).reshape(-1, 1)).sum()
        tab.append(kde.score_samples(np.asarray(area).reshape(-1, 1)).sum())
        
        my_dict2[name] = np.max(area) - np.min(area)
        
        if k == 2:
            matrix.append(np.asarray(tab))
            tab = []
            k = 0
        else:
            k += 1
        # -- end if
        ind += 1
    # -- end for     
    matrix = np.array(matrix)
    
    # Sort values
    result = dict(sorted(my_dict.items(),
                          key = lambda  item: item[1],
                          reverse = False))
    
    # Get the last 4 values sorted by ASC
    zones = {}
    last_pos = ""
    area_numbers = []
    for x in list(reversed(list(result)))[0:4]:
        zones[x] = result[x]
        last_pos = x
        area_numbers.append(int(x[-1]))
    # -- end for
    print("------ Zones 1 ---------")
    print(zones)
    
    
    # Plot deformation from area numbers
    plot_deformation(data, area_numbers)
    
    # Erase Area from Matrix
    # If matrix[k] > zones[last_pos] then matrix[k] = 0
    ind = 0    
    while ind < 3: 
        last_pos = list(zones)[-1-ind]
        matrix = np.where(matrix < zones[last_pos], 0, matrix)
        print(matrix)
    
        # Add column and line
        col_sum = np.sum(matrix, axis=0)
        print("\nCol_Sum: ", col_sum)
        line_sum = np.sum(matrix, axis=1)
        print("Line_Sum :", line_sum, "\n")
        if (0 in col_sum) and (0 in line_sum):
            ind = 100
        else:
            ind += 1
    # -- End while
    return matrix

# =============================================================================
def draw_real_position(position):
    """
        Used to draw the force plateform and different areas.
        Parameters
        ----------
        postion : List which contains position of the gravity center
        
        Returns
        -------
        None.

    """
    realX = np.array([-6.9, 0, 7.2, -6.9, 0, 7, -7, 0, 7])
    realY = np.array([6.4, 6.5, 6.6, 0, 0, 0, -6.8, -6.5, -6.2])
    sensorX = np.array([13.3, 10.7, -11, -13.8])
    sensorY = np.array([13.3, -10.1, -10, 13])
    
    tmp = []
    X = []
    Y = []
    pos = position.reshape(1, -1)
    pos = pos[0]
    for i in range(len(pos)):
        if pos[i] != 0:
           tmp.append(pos[i])
           X.append(realX[i])
           Y.append(realY[i])
    X = np.mean(np.asarray(X))
    Y = np.mean(np.asarray(Y))
    
    plt.figure()
    fig, ax = plt.subplots()
    # Plot Area
    ax.scatter(realX, realY, color='blue', s=80)
    # Plot Sensor
    ax.scatter(sensorX, sensorY, color='black', s=80)
    for i in range(len(realX)):
        ax.annotate("Z{}".format(i+1), (realX[i], realY[i]))
    
    for i in range(len(sensorX)):
        ax.annotate("C{}".format(i+1), (sensorX[i], sensorY[i]))
        
    # Draw the balance
    plt.plot([sensorX[0], sensorX[1]], [sensorY[0], sensorY[1]], color='black')
    plt.plot([sensorX[0], sensorX[3]], [sensorY[0], sensorY[3]], color='black')
    plt.plot([sensorX[1], sensorX[2]], [sensorY[1], sensorY[2]], color='black')
    plt.plot([sensorX[3], sensorX[2]], [sensorY[3], sensorY[2]], color='black')
    
    # Draw grid    
    gridX = [realX[0]/2 + realX[0], 
             realX[2]/2 + realX[2], 
             realX[2] + realX[2]/2,
             realX[0] + realX[0]/2,
             realX[0]/2,
             realX[2]/2,
             realX[6]/2,
             realX[8]/2,
             realX[3] + realX[3]/2,
             realX[6] + realX[6]/2,
             realX[5] + realX[5]/2,
             realX[8] + realX[8]/2
             ]
    gridY = [realY[2]/2 + realY[2], 
             realY[6]/2 + realY[6], 
             realY[2] + realY[2]/2,
             realY[6] + realY[6]/2,
             realY[1] + realY[1]/2,
             realY[2] + realY[2]/2,
             realY[7] + realY[7]/2,
             realY[6] + realY[6]/2,
             realY[0]/2,
             realY[6]/2,
             realY[2]/2,
             realY[8]/2
             ]
    
    plt.plot([gridX[0], gridX[2]], [gridY[0], gridY[2]], color='g')
    plt.plot([gridX[0], gridX[3]], [gridY[0], gridY[3]], color='g')
    plt.plot([gridX[1], gridX[3]], [gridY[1], gridY[3]], color='g')
    plt.plot([gridX[1], gridX[2]], [gridY[1], gridY[2]], color='g')
    plt.plot([gridX[4], gridX[6]], [gridY[4], gridY[6]], color='g')
    plt.plot([gridX[5], gridX[7]], [gridY[5], gridY[7]], color='g')
    plt.plot([gridX[9], gridX[11]], [gridY[9], gridY[11]], color='g')
    plt.plot([gridX[8], gridX[10]], [gridY[8], gridY[10]], color='g')
    
    # Draw the position
    # ax.scatter(X, Y, color='red', s=80)
    # plt.plot([0, X], [0, Y], color='red')
    plt.xticks([])
    plt.yticks([])
    plt.title("Deplacement du centre de gravité")
    plt.show()
    
    
    
# =============================================================================
#
#           TEST FUNCTIONS
#
# =============================================================================
#  ________________                  ___________________
# |      |___|     |                | (C4)  |___|  (C1) |
# | [Z1] [Z2] [Z3] |                |                   |
# |                |                |                   |  
# | [Z4] [Z5] [Z6] |                |                   |
# |                |                |                   |
# | [Z7] [Z8] [Z9] |                |                   |
# \                /                \ (C3)         (C2) /
#  \______________/                  \_________________/
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

fd_empty_mes = 'Mesure_a_vide.tdms'  # Data from sensors without load
filename = 'Mesure_Z8_80kg.tdms'  # Data we want to process
filename2 = 'Mesure_Z2&Z3&Z5&Z6_60kg.tdms' # Data we want to process
filename3 = 'Mesure_Z6&Z9_60kg.tdms' # Data we want to process

# Create matrix with 9 classes from 60kg et 80kg
mat60 = create_matrix_from_9_area_data('Matrix', 60)
mat80 = create_matrix_from_9_area_data('Matrix', 80)
matrix = np.mean(np.asarray([mat60, mat80]), axis=0)

# Load data and compute variation
data = compute_variation(filename3, fd_empty_mes)

# Compute euclidean distance between data and matrix
tab_dist = compute_euclidean_distance_from_matrix(matrix, data)

# Compute Position
tab_pos = compute_position(tab_dist)

# Draw Position
draw_real_position(tab_pos)
