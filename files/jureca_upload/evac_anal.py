import os

os.system("python folder_ini.py")
os.system("python run_jupedsim.py")
#os.system("python run_jupedsim.py")
#os.system("python trajectory_modify.py")
#os.system("python trajectory_vornoi.py")
os.system("python density_map.py & python waiting_time_err.py")
os.system("python waiting_time_err.py")
os.system("wait")
print("!!!!!!!!!!!!Script finished!!!!!!!!!!!!!!!")
