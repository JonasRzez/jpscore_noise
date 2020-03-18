import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math as m
import os
from multiprocessing import Pool

def max_frame_fetch(traj):
    return  traj['FR'].max()


def lattice_data(traj):
    max_id = traj['#ID'].max()
    max_frame = traj['FR'].max()
    data_x_new = []
    data_y_new = []
    data_id_new = []
    data_frames_new = []
    for id in np.arange(1, max_id + 1):
        x_i = np.array(traj[traj['#ID'] == id]['X'])
        x_f = np.array(traj[traj['#ID'] == id]['FR'])
        y_i = np.array(traj[traj['#ID'] == id]['Y'])
        if x_i.shape[0] < max_frame:
            diff = max_frame - x_i.shape[0]
            x_nan = [np.nan for i in np.arange(0, diff)]
            x_i = np.append(x_i, x_nan)
            y_i = np.append(y_i, x_nan)
            x_f = np.append(x_f, np.arange(x_f[-1] + 1, x_f[-1] + diff + 1))
            x_id = [id for i in np.arange(0, x_i.shape[0])]
        else:
            x_id = np.array(traj[traj['#ID'] == id]['#ID'])  # deletes the last frame of the person with maximal frames saved to unify the length of all frames
            x_id = x_id[0:-1]
            x_i = x_i[0:-1]
            x_f = x_f[0:-1]
            y_i = y_i[0:-1]
        data_x_new = np.append(data_x_new, x_i)
        data_id_new = np.append(data_id_new, x_id)
        data_frames_new = np.append(data_frames_new, x_f)
        data_y_new = np.append(data_y_new, y_i)
        trajectory_dic = {'id': data_id_new, 'frame': data_frames_new, 'x': data_x_new, 'y': data_y_new}
        traj_frame = pd.DataFrame(trajectory_dic)
        x_dic = {}
        y_dic = {}
    for i in np.arange(1, max_id):
        x_col = np.array(traj_frame[traj_frame['id'] == i]['x'])
        y_col = np.array(traj_frame[traj_frame['id'] == i]['y'])
        x_dic[i] = x_col
        y_dic[i] = y_col
    traj_x_frame = pd.DataFrame(x_dic)
    traj_y_frame = pd.DataFrame(y_dic)
    return traj_x_frame, traj_y_frame

def lattice_data_simplified(traj):
    max_id = traj['#ID'].max()
    max_frame = traj['FR'].max()
    data_x_new = []
    data_y_new = []
    data_id_new = []
    data_frames_new = []
    for id in np.arange(1, max_id + 1):
        x_i = np.array(traj[traj['#ID'] == id]['X'])
        y_i = np.array(traj[traj['#ID'] == id]['Y'])
        data_x_new = np.append(data_x_new, x_i)
        data_y_new = np.append(data_y_new, y_i)
    return np.array(data_x_new),np.array(data_y_new)

def densty1d(delta_x, a):
    return np.array(list(map(lambda x: 1 / (m.sqrt(m.pi) * a) * m.e ** (-x ** 2 / a ** 2), delta_x)))


def normal(lattice_x, lattice_y, x_array, y_array, a):
    x_dens = np.array(
        [lattice_x - x for x in x_array])  # calculate the distant of lattice pedestrians to the measuring lattice
    y_dens = np.array([lattice_y - y for y in y_array])

    rho_matrix_x = np.array([densty1d(delta_x, a) for delta_x in x_dens])  # density matrix is calculated
    rho_matrix_y = np.array([densty1d(delta_y, a) for delta_y in y_dens])
    rho_matrix = np.matmul(rho_matrix_x, np.transpose(rho_matrix_y))
    return rho_matrix.mean()

path = "../../../files/evacuation_files/trajectories/"
sl = "/"
folder_frame = pd.read_csv("folder_list.csv")
folder_list = np.array(folder_frame['ini_folder'])
#b = np.arange(0.6,3.0,0.2)
variables = pd.read_csv("variables_list.csv")
print(folder_list)
folder = "trajectories/"
N_runs = variables['N_runs'][0]
fps = variables['fps'][0]
b_frame = pd.read_csv("b_list.csv")
b = np.array(b_frame['b'])
for i in range(N_runs):
    location = ["evac_traj_" + str(round(2 * bi,2))[0] + str(round(2*bi,2))[-1] + "_"+ str(i) +".txt" for bi in b]

    for loc,run_folder in zip(location,folder_list):
        file = open(folder  + run_folder + sl +loc, 'r')
        line_count = 0
        new_file = open(folder  + run_folder+ sl + "new_" + loc ,'w')
        print(loc)
        for line in file:
            line_count+= 1
            if line_count > 10:
                new_file.write(line)
        #os.system('rm ' + folder + loc)

        new_file.close()
        
path = "../../../files/evacuation_files/trajectories/"

density_mean = []
density = []
for bi, ini in zip(b,folder_list):
    location = "new_evac_traj_" + str(round(2 * bi,2))[0] + str(round(2*bi,2))[-1]
    density_runs = []
    max_frame = 0
    for i in range(N_runs):
        loc = folder + ini + sl + location + "_" + str(i) + ".txt"
        traj = pd.read_csv(loc, sep="\s+", header=0)
        frame = max_frame_fetch(traj)
        if frame > max_frame:
            max_frame = frame
    
    density_run = []
    for i in range(N_runs):
        loc = folder + ini + sl + location + "_" + str(i) + ".txt"
        print(loc)

        traj = pd.read_csv(loc, sep="\s+", header=0)
        l_x, l_y = lattice_data(traj)

        fwhm = 0.15
        a = fwhm * m.sqrt(2) / (2 * m.sqrt(2 * m.log(2)))
        lattice_x = np.array(l_x)
        lattice_y = np.array(l_y)
        x_array = np.linspace(-0.5,0.5,50)
        y_array = np.linspace(0.5,1.5,50)
        print("*****************<calc density>*****************")
        lat_x_no_nan = []
        lat_y_no_nan = []

        for lat_x,lat_y in zip(lattice_x,lattice_y):
            l_x_no_nan = [x  for x in lat_x if np.isnan(x) == False]
            l_y_no_nan = [y  for y in lat_y if np.isnan(y) == False]
            lat_x_no_nan.append(l_x_no_nan)
            lat_y_no_nan.append(l_y_no_nan)

        #dens_run = normal(lat_x_no_nan,lat_y_no_nan,x_array,y_array,a)
        print("    *****************<pool>*********************")
        pool = Pool()
        g_pool = np.array([pool.apply_async(normal, args=(l_x_no_nan,l_y_no_nan,x_array,y_array,a)) for l_x_no_nan, l_y_no_nan in zip(lat_x_no_nan, lat_y_no_nan)])
        density_run = [p.get() for p in g_pool]
        pool.close()
        print("    *****************</pool>********************")
        print("*****************</calc density>****************")
        
        dens_run_shape = np.array(density_run).shape[0]
        #print(dens_run_shape.shape[0],max_frame)
        print(np.array(density_run).shape)
        

        if dens_run_shape < max_frame:
            diff = int(max_frame-dens_run_shape)
            #print(diff)
            nod_list = [0 for i in np.arange(0,diff)]
            density_run = np.append(density_run,nod_list)
        #print(np.array(density_run).shape,max_frame)
        density_runs.append(np.array(density_run))
        #plt.plot(density_run)
        
    
    
    #plt.show()
    mean_runs_dens = np.array(density_runs).mean(axis=0)
    std_runs_dens = np.array(density_runs).std(axis=0)
    density.append(mean_runs_dens)
    
    print("density shape = ", np.array(density).shape)
            #time = np.arange(0,59.2,0.1)
    density_sat_state = np.array(mean_runs_dens[int(fps * 5):int(fps*10)])
    print(density_sat_state.shape)
    density_mean.append(density_sat_state.mean())

print(std_runs_dens)
t = np.arange(0,np.array(density[0]).shape[0])

print(std_runs_dens)
t = np.arange(0,np.array(density[0]).shape[0])
dens_map = {}

for d in density:
    plt.plot(d)
    dens_map[str(bi)] = d

dens_frame = pd.DataFrame(dens_map)
dens_frame.to_csv('dens_frame.csv')
plt.show()

den = []
std = []
den2 = []
print(np.array(density)[0].mean())
fig, ax = plt.subplots(1, figsize=(8, 6))

for d in np.array(density):
    den.append(d[fps * 5: fps*10].mean())
    std.append(d[fps * 5: fps*10].std())
    #den2.append(d2[25 * 5: 25*10].mean())
den1 = []
for d in den:
    den1.append(1/d)
    
plt.scatter(2*b,den1)
#plt.scatter([1.2,2.3,3.4,4.5,5.6],[2.,3.7,4.2,4.8,5], label = "approx ben sim")
b22 = [1.2,2.3,3.4,4.5,4.5,5.6]
#plt.scatter(b22,[2.2,3.3,4.3,3.8,5.1,4.3], label = "approx jules results")
ax.legend(loc="upper left", title="density", frameon=False)

#plt.scatter(2*b,den2)
plt.yscale('log')
plt.xscale('log')

plt.show()
