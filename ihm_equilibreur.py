# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 09:08:58 2020

@author: sangbe SIDIBE
"""
"""
try:
    from tkinter import *
    from tkinter.ttk import *
except:
    from Tkinter import *
    from ttk import *
"""
import tkinter as tk
from tkinter import *
import pandas as pd
from nptdms import TdmsFile
import seaborn as sns
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



class MyApp(Frame):
    """Basic Frame for the application"""

    def __init__(self, parent=None):
        """ Used to create the interface """
        self.parent = parent
        self.data = []
        self.area_number = []
        self.tab_dist = []
        self.btn_enabled = False
        self.fig_boxplot = None
        self.fig_violinplot = None
        self.fig_realpos = None
        self.ax = None
        Frame.__init__(self)
        
        self.main = self.master
        self.main.geometry('800x600+200+100')
        self.main.title("L'EQUILIBREUR")
        
        f = Frame(self.main, bg="cyan")
        f.pack(fill=tk.BOTH)
        
        b = Button(f, text="Charger Données", command=self.select_data, fg="blue")
        b.pack(side=tk.TOP)
        
        # self.txt_filename = Entry(f)
        self.txt_filename = Label(f, text='')
        self.txt_filename.pack(side=tk.TOP, fill=tk.BOTH)
        
        b = tk.Button(f, text="Effacer Ecran", command=self.destroy_frame, fg="red")
        b.pack(side=tk.RIGHT, fill=tk.BOTH)
        
        b = tk.Button(f, text="Traiter Données", command=self.process_data, fg="green")
        b.pack(side=tk.LEFT, fill=tk.BOTH)
        
        
        self.frame_fig = Frame(self.main, bg="white")
        self.frame_fig.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        # pt = make_table(self.frame_fig)
        
        btn_frame = Frame(self.main) 
        btn_frame.pack(side=BOTTOM)
        
        b = Button(btn_frame,text='Pression subie par Zones', command=self.draw_boxplot)
        b.pack(side=tk.LEFT,fill=tk.BOTH,)
        b = Button(btn_frame,text='Déformation Centre de Zones', command=self.draw_violinplot)
        b.pack(side=tk.LEFT,fill=tk.BOTH,)
        b = Button(btn_frame,text='Déplacement Centre Gravité', command=self.draw_real_position)
        b.pack(side=tk.LEFT,fill=tk.BOTH,)
        b = Button(btn_frame,text='Sauvegarder Résultats', bg="blue", fg="white", command=self.save_result)
        b.pack(side=tk.LEFT,fill=tk.BOTH,)
        return

    
    # ----------------------
    def load_data(self, filename):
        """
            Used to load data from a tdms file.

        Parameters
        ----------
        filename : String
            The path for the file we want to load.

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
    
    
    # ----------------------
    def save_result(self):
        """
            Use to save the deformation in the computer or somewhere else.
            Results are saved as image in png format.
            
        Returns
        -------
        None.

        """
        if self.btn_enabled and len(self.tab_dist) > 0 and len(self.data) > 0:
            # self.fig.savefig("Test.png")
            filename =  tk.filedialog.asksaveasfilename(initialdir = "/",
                                                     title = "Enregistrer Résultats",
                                                     filetypes = (("png files","*.png"),
                                                                  ("png files","*.png")))
            words = filename.split('/')
            path = '/'.join(words[:-1])
            
            box_name = path + '/' + words[-1] + '_01.png'
            violin_name = path + '/' + words[-1] + '_02.png'
            pos_name = path + '/' + words[-1] + '_03.png'
            
            self.fig_boxplot.savefig(box_name)
            self.fig_violinplot.savefig(violin_name)
            self.fig_realpos.savefig(pos_name)
            
            
    # ----------------------
    def select_data(self):
        """
            Used to load a file from a device (Computer, Server, etc...)
            Only tdms file will be allowed.
            When, the path of the file is get, the data are loaded through
            the function load_data.

        Returns
        -------
        None.

        """
        self.txt_filename['text'] = ''
        self.data = []
        self.mat_pos = []
        self.area_number = []
        self.tab_dist = []
        
        filename = tk.filedialog.askopenfilename(initialdir = "/",
                                              title = "Select file",
                                              filetypes = (("TDMS files","*.tdms"),
                                                           ("TDMS files","*.tdms")))
        # self.txt_filename.insert(tk.END, str(filename))
        self.txt_filename['text'] = str(filename)
        
        # Load the data
        self.data = self.load_data(filename)
        self.btn_enabled = False
    
    
    # ----------------------
    def process_data(self):
        """
            Used to compute the displacement of the center of gravity.            

        Returns
        -------
        None.

        """
        if len(self.txt_filename['text']) > 0 and len(self.data) > 0:
            self.mat_pos = []
            self.area_number = []
            self.tab_dist = []
            self.destroy_frame()
            fd_empty_mes = 'Mesure_a_vide.tdms'  # Data from sensors without load
            
            # Create matrix with 9 classes from 60kg et 80kg
            mat60 = create_matrix_from_9_area_data('Matrix', 60)
            mat80 = create_matrix_from_9_area_data('Matrix', 80)
            matrix = np.mean(np.asarray([mat60, mat80]), axis=0)
            
            # Load data and compute variation
            # self.data = compute_variation(self.txt_filename.get(), fd_empty_mes)
            self.data = compute_variation(self.txt_filename['text'], fd_empty_mes)
    
            # Compute euclidean distance between data and matrix
            self.tab_dist = compute_euclidean_distance_from_matrix(matrix, self.data)
            
            # Compute Position
            self.mat_pos = compute_position(self.tab_dist, self.area_number)
            self.btn_enabled = True
    
    
    # ----------------------
    def destroy_frame(self):
        """
            Used to destroy a drawing.

        Returns
        -------
        None.

        """
        self.frame_fig.destroy()
        self.frame_fig = tk.Frame(self.main, bg="white")
        self.frame_fig.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    
    # ----------------------
    def draw_boxplot(self, area=None, filename=None):
        """
            Used to draw the deformation of areas using interquartiles method.

        Parameters
        ----------
        area : List, optional
            Specific areas you want to draw. The default is None.
        filename : String, optional
            A title for the plot. The default is None.

        Returns
        -------
        None.

        """
        if self.btn_enabled and len(self.tab_dist) > 0 and len(self.data) > 0:
            my_dict = {}
            for z in self.area_number:
                name = 'Zone{}'.format(z)
                my_dict[name] = self.tab_dist[z-1]
                
            # Create the DataFrame from my_dict
            df = pd.DataFrame(my_dict)
            
            self.destroy_frame()
            
            self.fig_boxplot, ax = plt.subplots()
            plt.xlabel("Zones", size=12)
                
            if filename == None:
                plt.title("Pression subie par zones les plus impactées")
            else:
                plt.title(("Box Plot: " + filename))
            
            sns.boxplot(data=df)
            canvas = FigureCanvasTkAgg(self.fig_boxplot, self.frame_fig)
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            
    
    # ----------------------
    def draw_violinplot(self, filename=None):
        """
            Used to draw the deformation of areas using interquatiles and 
            kernel density distribution combined method.

        Parameters
        ----------
        filename : String, optional
            A title for the plot. The default is None.

        Returns
        -------
        None.

        """
        if self.btn_enabled and len(self.tab_dist) > 0 and len(self.data) > 0:
            my_dict = {}
            for z in self.area_number:
                name = 'Zone{}'.format(z)
                my_dict[name] = self.tab_dist[z-1]
                
            # Create the DataFrame from my_dict
            df = pd.DataFrame(my_dict)
            
            self.destroy_frame()
            
            self.fig_violinplot, ax = plt.subplots()
            plt.xlabel("Zones", size=12)
                
            if filename == None:
                plt.title("Pression subie par les centres de zones les plus impactées")
            else:
                plt.title(("Box Plot: " + filename))
            
            sns.violinplot(data=df)
            canvas = FigureCanvasTkAgg(self.fig_violinplot, self.frame_fig)
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            
    
    # ----------------------
    def draw_real_position(self):
        """
            Used to draw the balance and different areas.

        Returns
        -------
        None.

        """
        if self.btn_enabled and len(self.tab_dist) > 0 and len(self.data) > 0:
            realX = np.array([-6.9, 0, 7.2, -6.9, 0, 7, -7, 0, 7])
            realY = np.array([6.4, 6.5, 6.6, 0, 0, 0, -6.8, -6.5, -6.2])
            sensorX = np.array([13.3, 10.7, -11, -13.8])
            sensorY = np.array([13.3, -10.1, -10, 13])
            
            tmp = []
            X = []
            Y = []
            pos = self.mat_pos.reshape(1, -1)
            pos = pos[0]
            for i in range(len(pos)):
                if pos[i] != 0:
                   tmp.append(pos[i])
                   X.append(realX[i])
                   Y.append(realY[i])
            X = np.mean(np.asarray(X))
            Y = np.mean(np.asarray(Y))
            
            self.destroy_frame()
            
            self.fig_realpos, ax = plt.subplots()
            
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
            ax.scatter(X, Y, color='red', s=80)
            plt.plot([0, X], [0, Y], color='red')
            plt.xticks([])
            plt.yticks([])
            plt.title("Deplacement du centre de gravité")
            canvas = FigureCanvasTkAgg(self.fig_realpos, self.frame_fig)
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

# -- End Class MyApp


# =============================================================================
def load_data(filename):
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
    data = load_data(filename1)
    data_without_loads = load_data('Matrix\{}'.format(filename2))
    if len(data[0]) < len(data_without_loads[0]):
        data_without_loads[0] = data_without_loads[0][:len(data[0])]
        data_without_loads[1] = data_without_loads[1][:len(data[1])]
        data_without_loads[2] = data_without_loads[2][:len(data[2])]
        data_without_loads[3] = data_without_loads[3][:len(data[3])]
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
            tmp1 = np.sqrt(np.dot(dataC1, dataC1) - 
                           2 * np.dot(dataC1, zone[0][:len(dataC1)]) + 
                           np.dot(zone[0][:len(dataC1)], zone[0][:len(dataC1)]))
            tmp2 = np.sqrt(np.dot(dataC2, dataC2) - 
                           2 * np.dot(dataC2, zone[1][:len(dataC2)]) + 
                           np.dot(zone[1][:len(dataC2)], zone[1][:len(dataC2)]))
            tmp3 = np.sqrt(np.dot(dataC3, dataC3) - 
                           2 * np.dot(dataC3, zone[2][:len(dataC3)]) + 
                           np.dot(zone[2][:len(dataC3)], zone[2][:len(dataC3)]))
            tmp4 = np.sqrt(np.dot(dataC4, dataC4) - 
                           2 * np.dot(dataC4, zone[3][:len(dataC4)]) + 
                           np.dot(zone[3][:len(dataC4)], zone[3][:len(dataC4)]))
            result.append(np.asarray([tmp1, tmp2, tmp3, tmp4]))
        # -- end for
        return result
    # -- end compute_euclidean_distance

    tab_dist = compute_euclidean_distance(matrix, data[0], data[1], data[2], data[3])
    return tab_dist


# =============================================================================
def compute_position(data, area_number):
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
    for x in list(reversed(list(result)))[0:4]:
        zones[x] = result[x]
        last_pos = x
        area_number.append(int(x[-1]))
    # -- end for
    
    # Erase Area from Matrix
    # If matrix[k] > zones[last_pos] then matrix[k] = 0
    ind = 0    
    while ind < 3: 
        last_pos = list(zones)[-1-ind]
        matrix = np.where(matrix < zones[last_pos], 0, matrix)
    
        # Add column and line
        col_sum = np.sum(matrix, axis=0)
        line_sum = np.sum(matrix, axis=1)
        if (0 in col_sum) and (0 in line_sum):
            ind = 100
        else:
            ind += 1
    # -- End while
    return matrix


# =============================================================================
#               MAIN
# =============================================================================
app = MyApp()
app.mainloop()
