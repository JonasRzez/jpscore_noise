#!/usr/bin/env python
import os
import numpy as np
import itertools
from jinja2 import Environment, FileSystemLoader
import pandas as pd
import random as rand
PATH = os.path.dirname(os.path.abspath("evac_geo_temp.xml"))
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(PATH),
    trim_blocks=False)

def render_template(template_filename, context):
    return TEMPLATE_ENVIRONMENT.get_template(template_filename).render(context)

def Product(variables):
    return list(itertools.product(*variables))

def create_inifile(geo_name,geo_name_ini,b_list,ini_list,location,folder,r,l_list,seed_list,stepsize,fps,N_ped):
    
   # location = [path + "evac_traj_" + str(2 * bi)[0] + str(2*bi)[-1] + ".txt" for bi in b_list]
    print(geo_name,location)
    for b,fname,l in zip(b_list, geo_name,l_list):
        context= {'b': b,'l':l , 'll': 1 * l}
        fname = folder + fname
        print("output: ",  fname)
        with open(fname,'w') as f:
            xml = render_template('evac_geo_temp.xml', context)
            f.write(xml)
    
    for geo,loc,fname_ini,b,l,seed in zip(geo_name_ini,location,ini_list,b_list,l_list,seed_list):
        ini_context = {'geo':geo,'location':loc,'b':b, 'r':r,'l':l, 'll':  1 * l,'seed':seed,'stepsize':stepsize,'fps':fps,'N_ped':N_ped}
        #fname_ini = "ini_" + geo[-6:-4] + ".xml"
        fname_ini = folder + fname_ini
        print("output: ", fname_ini)
        with open(fname_ini, 'w') as f:
            xml_ini = render_template('evac_ini_temp.xml', ini_context)
            f.write(xml_ini)
            
def b_data_name(b,dig):
    b = round(2 * b,dig)
    str_b = ''
    for let in str(b):
        if (let in '.'):
            str_b += '_'
        else:
            str_b += let
    return str_b

def main():
    dig = 3 #for the rounding of digits


    b = np.array([1.2,2.3,3.4,4.5,5.6])
    #b = np.array([5.6 for i in range(5)])
    #b = np.array([5.6])
    """
    b1 = [round(b,dig) for b in b1]
    b2 = np.arange(3  ,4,0.1)
    b2 = [round(b,dig) for b in b2]
    b3 = np.arange(4  ,8,0.5)
    b3 = [round(b,dig) for b in b3]
    b4 = np.arange(8  ,10,1)
    b4 = [round(b,dig) for b in b4]
    """
    b = np.array([1.2])

    b = b/2
    
    rho_ini_list = [4,2.4,3.5,2.5,2.5]
    r = 0.17
    fps = 16
    stepsize = 0.05
    pi = 3.14159
    N_ped = 50
    N_runs = 1
    #l_list = [round(N_ped * pi * r*r/(2*bi * rho_ini),2) for bi in b]
    l_list = [round(N_ped/(rho_ini*bi),2) for bi,rho_ini in zip(b,rho_ini_list)]
    #l_list = [3 for bi,rho_ini in zip(b,rho_ini_list)]
    print(str(round(2 * b[0],2)))
    ini_folder_list = []
    
    
    path = "/trajectories/"
    traj_folder = "trajectories/"
    data_folder = "ini_lm_" + str(N_ped)
    print_level = "--log-level [debug, info, warning, error, off]"

    os.system("mkdir " + traj_folder + data_folder)
    path = path + "/" + data_folder + "/"
    for bi in b:
        ini_folder_name = "ini_"+ b_data_name(bi,dig) + "_lm_" + str(N_ped)
        os.system("mkdir " + traj_folder + "/" + data_folder + "/" + ini_folder_name)
        ini_folder_list.append(ini_folder_name)
    geo_name = [data_folder + "/" + ini_folder + "/" + "geo_" + b_data_name(bi,dig) + ".xml" for bi, ini_folder in zip(b,ini_folder_list)]
    geo_name_ini = ["geo_" + b_data_name(bi,dig) + ".xml" for bi in b]

    ini_list = [ data_folder + "/" +ini_folder + "/" + "ini_" + b_data_name(bi,dig) + ".xml" for bi, ini_folder in zip(b,ini_folder_list)]
    

    print(ini_list)
    
    variables_df = pd.DataFrame({'r':[r],'fps':[fps],'N_runs':[N_runs],'N_ped':[N_ped]})
    variables_df.to_csv("variables_list.csv")
    b_df = pd.DataFrame({'b':b})
    b_df.to_csv("b_list.csv")
    folder_path_list = [data_folder + "/" + ini for ini in ini_folder_list]
    folder_df = pd.DataFrame({'ini_folder':folder_path_list})
    folder_df.to_csv("folder_list.csv")
    
   
    for i in range(N_runs):
        location = ["evac_traj_" + b_data_name(bi,dig) +"_" +str(i)+ ".txt" for bi,folder_name in zip(b,ini_folder_list)]
        seed_list = [int(rand.uniform(0,1485583981)) for bi in b]

        create_inifile(geo_name,geo_name_ini,b,ini_list,location,traj_folder,r,l_list,seed_list,stepsize,fps,N_ped)
        os.system("pwd")
        os.chdir("../../build/bin")
        
        for ini, ini_folder in zip(ini_list,ini_folder_list):
            jps = "./jpscore ../../files/evacuation_files/"+ traj_folder + ini
            print(jps)
            os.system(jps)
        os.chdir("../../files/evacuation_files")
        os.system("pwd")


    b_width = 0
    b_loc_count = 0
    for bi in 2*b:
        
        if round(bi,dig) == 2.0:
            b_width = b_loc_count
        b_loc_count += 1
    os.chdir("trajectories/" + data_folder +"/"+ ini_folder_list[b_width] + "/")
    os.system("pwd")
    vis_loc = "evac_traj_" + b_data_name(b[b_width],dig) + "_0.txt"
    print("jpsvis "+ vis_loc)

    print("vis location = ", vis_loc)
    os.system("jpsvis "+ vis_loc)
    #for geo, ini in zip(geo_name,ini_list):
        #os.system("rm " + geo)
        #print("removed "  + geo)
        #os.system("rm " + ini)
        #print("removed " + ini)
if __name__ == "__main__":
    main()


