import time
import pickle
import copy
import gc
import numpy as np

from utils.options import args_parser
from utils.sumo_utils import read_trajectoryInfo_timeindex


args = args_parser()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
import mitsuba as mi
try:
    import sionna.rt
except ImportError as e:
    import os
    os.system("pip install sionna-rt")
    import sionna.rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, ITURadioMaterial, SceneObject, subcarrier_frequencies


# 设置参数
args.trajectoryInfo_path = './sumo_data/trajectory_Lbd0.10.csv'
start_time=args.sionna_start_time
end_time=args.sionna_end_time
frequency = args.freq # 28e9 or 5.9e9
pattern = args.antenna_pattern # "iso" or "tr38901"
N_t_H = args.N_t_H
N_t_V = args.N_t_V
N_r_H = args.N_r_H
N_r_V = args.N_r_V
h_car = args.h_car
h_rx = args.h_rx
h_tx = args.h_tx

trajectoryInfo = read_trajectoryInfo_timeindex(
    args,
    start_time=start_time,
    end_time=end_time,
    display_intervel=0.05,
)
channel_gains_list = []

car_material = ITURadioMaterial("car-material",
                                "metal",
                                thickness=0.01,
                                color=(0.8, 0.1, 0.1))
#for scene_time in 500+0.1*np.array(list(range(10))):
scene = load_scene('scene_from_sionna.xml',merge_shapes=False)

Road_horizontal1 = scene.get('Road_horizontal1')
Road_horizontal1.radio_material = 'itu_concrete'
Road_horizontal2 = scene.get('Road_horizontal2')
Road_horizontal2.radio_material = 'itu_concrete'
Road_vertical1 = scene.get('Road_vertical1')
Road_vertical1.radio_material = 'itu_concrete'
Road_vertical2 = scene.get('Road_vertical2')
Road_vertical2.radio_material = 'itu_concrete'

# Configure antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=N_t_H,
                            num_cols=N_t_V,
                            vertical_spacing=0.5,
                            horizontal_spacing=0.5,
                            pattern=pattern,
                            polarization="V")

# Configure antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=N_r_H,
                            num_cols=N_r_V,
                            vertical_spacing=0.5,
                            horizontal_spacing=0.5,
                            pattern=pattern,
                            polarization="V")

# Create transmitter
tx1 = Transmitter(name="tx_1",
                position=[300,300,h_tx])
tx2 = Transmitter(name="tx_2",
                position=[-300,300,h_tx])
tx3 = Transmitter(name="tx_3",
                position=[300,-300,h_tx])
tx4 = Transmitter(name="tx_4",
                position=[-300,-300,h_tx])
tx1.look_at(mi.Point3f(0,0,0))
tx2.look_at(mi.Point3f(0,0,0))
tx3.look_at(mi.Point3f(0,0,0))
tx4.look_at(mi.Point3f(0,0,0))

# Add transmitter instance to scene
scene.add(tx1)
scene.add(tx2)
scene.add(tx3)
scene.add(tx4)

max_car_num = max([len(v) for v in trajectoryInfo.values()])
sim_cars = [SceneObject(fname=sionna.rt.scene.low_poly_car, # Simple mesh of a car
                    name=f"car_{i+1}",
                    radio_material=car_material)
        for i in range(max_car_num)]

for scene_time in trajectoryInfo.keys():
    car_positions = []
    car_velocities = []
    rx_positions = []
    rx_velocities = []
    
    num_cars = len(trajectoryInfo[scene_time])
    scene.edit(add=sim_cars[:num_cars])
    
    for i,veh in enumerate(trajectoryInfo[scene_time].values()):
        x, y = veh['pos']
        v, angle = veh['v'], veh['angle']
        v_x, v_y = v*np.cos(angle), v*np.sin(angle)
        
        sim_cars[i].position = mi.Point3f(x.item(), y.item(), h_car)
        sim_cars[i].velocity = mi.Point3f(v_x.item(), v_y.item(), 0)
        sim_cars[i].orientation = mi.Point3f((angle/180-0.5)*np.pi, 0, 0)
        car_positions.append(sim_cars[i].position)
        car_velocities.append(sim_cars[i].velocity)
        rx_positions.append([x.item(),y.item(),h_rx])
        rx_velocities.append(sim_cars[i].velocity)
        
        # Create a receiver
        rx = Receiver(name=f"rx_{i+1}",
                    position=rx_positions[i],
                    orientation=[(angle/180-0.5)*np.pi,0,0],
                    display_radius=0.5,)
        # Add receiver instance to scene
        scene.add(rx)

    scene.frequency = frequency # in Hz; implicitly updates RadioMaterials
    scene.synthetic_array = True # If set to False, ray tracing will be done per antenna element (slower for large arrays)
    
    # Compute propagation paths
    p_solver = PathSolver()
    paths = p_solver(scene, max_depth=5)
    
    # OFDM system parameters
    num_subcarriers = 1 # 1024
    subcarrier_spacing=30e3

    # Compute frequencies of subcarriers relative to the carrier frequency
    frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)
    h_freq = paths.cfr(frequencies=frequencies,
                   normalize=False,
                   normalize_delays=True,
                   out_type="numpy")
    
    channel_gains = copy.deepcopy(h_freq[:,:,:,:,0,0])
    channel_gains_list.append(channel_gains)
    
    for i,k in enumerate(trajectoryInfo[scene_time].keys()):
        trajectoryInfo[scene_time][k]['h'] = channel_gains[i,:,:]

    del h_freq
    del paths
    del p_solver
    scene.edit(remove=sim_cars[:num_cars])
    for i,v in enumerate(trajectoryInfo[scene_time].values()):
        scene.remove(f"rx_{i+1}")
    gc.collect()
    
# 保存文件
os.makedirs(args.sionna_result_tmp_dir, exist_ok=True)
with open(os.path.join(args.sionna_result_tmp_dir,
    f"time({start_time:.1f},{end_time:.1f})_tx({N_t_H},{N_t_V})_rx({N_r_H},{N_r_V})_freq{frequency:.1e}.pkl"), "wb") as tf:
    pickle.dump(trajectoryInfo,tf)