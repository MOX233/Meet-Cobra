import os # Configure which GPU
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sionna
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

# Set random seed for reproducibility
sionna.config.seed = 18

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

# For link-level simulations
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement

from utils.options import args_parser
from utils.sumo_utils import read_trajectoryInfo_timeindex

# 设置参数
N_t = 64  # 发射天线数
N_r = 1   # 接收天线数

# 生成 DFT 码本
def generate_dft_codebook(N_t):
    # 创建 DFT 矩阵
    # 这就是 N_t 个基向量的矩阵，包含 N_t 列，每列是一个 DFT 基向量
    dft_matrix = np.exp(-1j * 2 * np.pi * np.outer(np.arange(N_t), np.arange(N_t)) / N_t)
    return dft_matrix

args = args_parser()
args.slots_per_frame = 100
args.beta_macro = 3
args.beta_micro = 4
args.bias_macro = -60
args.bias_micro = -60
args.shd_sigma_macro = 0
args.shd_sigma_micro = 0
args.num_RB_macro = 100
args.num_RB_micro = 100
args.RB_intervel_macro = 0.18 * 1e6
args.RB_intervel_micro = 1.8 * 1e6
args.p_macro = 1
args.p_micro = 0.1
args.data_rate = 0.7 * 1e6
args.trajectoryInfo_path = '/home/ubuntu/niulab/Sionna/sumo_result/trajectory_Lbd0.10.csv'
start_time=500
end_time=500.5
# pattern = "tr38901"
pattern = "iso"
h_rx = 10
h_tx = 10

trajectoryInfo = read_trajectoryInfo_timeindex(
    args,
    start_time=start_time,
    end_time=end_time,
    display_intervel=0.05,
)
channel_gains_list = []
#for scene_time in 500+0.1*np.array(list(range(10))):
for scene_time in trajectoryInfo.keys():
    _time = time.time()
    #scene = sionna.rt.load_scene('scene_self_defined_v2.xml')
    scene = sionna.rt.load_scene('scene_NoItem.xml')
    car_positions = []
    car_velocities = []
    rx_positions = []
    
    for i,(k,v) in enumerate(trajectoryInfo[scene_time].items()):
        # 强行修改车辆位置
        trajectoryInfo[scene_time][k]['pos'] = np.random.uniform(-100,100,2)
        
    for i,v in enumerate(trajectoryInfo[scene_time].values()):
        car = scene.get(f'Car.{i+1}')
        x, y = v['pos']
        v, angle = v['v'], v['angle']
        v_x, v_y = v*np.cos(angle), v*np.sin(angle)
        
        rx_positions.append([x,y,h_rx])
        

    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=64,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern=pattern,
                                polarization="V")

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern=pattern,
                                polarization="V")

    # Create transmitter
    tx1 = Transmitter(name="tx_1",
                    position=[270,270,h_tx])
    tx2 = Transmitter(name="tx_2",
                    position=[-270,270,h_tx])
    tx3 = Transmitter(name="tx_3",
                    position=[270,-270,h_tx])
    tx4 = Transmitter(name="tx_4",
                    position=[-270,-270,h_tx])
    tx1.look_at([0,0,0])
    tx2.look_at([0,0,0])
    tx3.look_at([0,0,0])
    tx4.look_at([0,0,0])
    
    # Add transmitter instance to scene
    scene.add(tx1)
    scene.add(tx2)
    scene.add(tx3)
    scene.add(tx4)

    """rx0 = Receiver(name=f"rx_0",
                    position=[0,0,h_rx],
                    orientation=[0,0,0])
        
        # Add receiver instance to scene
    scene.add(rx0)"""
        
    # Create a receiver
    for i,v in enumerate(trajectoryInfo[scene_time].values()):
        rx = Receiver(name=f"rx_{i+1}",
                    position=rx_positions[i],
                    orientation=[0,0,0])
        
        # Add receiver instance to scene
        scene.add(rx)

    # tx.look_at(rx) # Transmitter points towards receiver

    scene.frequency = 5.9e9 # in Hz; implicitly updates RadioMaterials

    scene.synthetic_array = True # If set to False, ray tracing will be done per antenna element (slower for large arrays)

    # Compute propagation paths
    paths = scene.compute_paths(max_depth=5,
                                num_samples=1e6)  # Number of rays shot into directions defined
                                                # by a Fibonacci sphere , too few rays can
                                                # lead to missing paths

    # Show the coordinates of the starting points of all rays.
    # These coincide with the location of the transmitters.
    # print("Source coordinates: ", paths.sources.numpy())
    # print("Transmitter coordinates: ", list(scene.transmitters.values())[0].position.numpy())

    # Show the coordinates of the endpoints of all rays.
    # These coincide with the location of the receivers.
    # print("Target coordinates: ",paths.targets.numpy())
    # print("Receiver coordinates: ",list(scene.receivers.values())[0].position.numpy())

    # Show the types of all paths:
    # 0 - LoS, 1 - Reflected, 2 - Diffracted, 3 - Scattered
    # Note that Diffraction and scattering are turned off by default.
    # print("Path types: ", paths.types.numpy())

    # We can now access for every path the channel coefficient, the propagation delay,
    # as well as the angles of departure and arrival, respectively (zenith and azimuth).

    # Let us inspect a specific path in detail
    # path_idx = 0 # Try out other values in the range [0, 13]
    # For a detailed overview of the dimensions of all properties, have a look at the API documentation
    # print(f"\n--- Detailed results for path {path_idx} ---")
    # print(f"Channel coefficient: {paths.a[0,0,0,0,0,path_idx, 0].numpy()}")
    # print(f"Propagation delay: {paths.tau[0,0,0,path_idx].numpy()*1e6:.5f} us")
    # print(f"Zenith angle of departure: {paths.theta_t[0,0,0,path_idx]:.4f} rad")
    # print(f"Azimuth angle of departure: {paths.phi_t[0,0,0,path_idx]:.4f} rad")
    # print(f"Zenith angle of arrival: {paths.theta_r[0,0,0,path_idx]:.4f} rad")
    # print(f"Azimuth angle of arrival: {paths.phi_r[0,0,0,path_idx]:.4f} rad")

    paths_dict = paths.to_dict()
    
    # import ipdb;ipdb.set_trace()
    
    [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps] = paths_dict['a'].shape
    # print('batch_size=', batch_size)
    # print('num_rx=', num_rx)
    # print('num_rx_ant=',num_rx_ant)
    # print('num_tx=', num_tx)
    # print('num_tx_ant=', num_tx_ant)
    # print('max_num_paths=', max_num_paths)
    # print('num_time_steps=', num_time_steps)

    from sionna.ofdm import ResourceGrid
    rg = ResourceGrid(num_ofdm_symbols=1024,
                    fft_size=1024,
                    subcarrier_spacing=30e3)

    # delay_resolution = rg.ofdm_symbol_duration/rg.fft_size
    # print("Delay   resolution (ns): ", int(delay_resolution/1e-9))

    # doppler_resolution = rg.subcarrier_spacing/rg.num_ofdm_symbols
    # print("Doppler resolution (Hz): ", int(doppler_resolution))

    a = paths_dict['a']
    tau = paths_dict['tau']
    frequencies = tf.ones([1]) * 5.9e9
    # frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
    channel_gains = cir_to_ofdm_channel(frequencies,a,tau)[0,:,0,:,:,0,0]
    # channel_gains = cir_to_ofdm_channel(frequencies,a,tau)
    channel_gains_list.append(channel_gains)
    
    for i,k in enumerate(trajectoryInfo[scene_time].keys()):
        trajectoryInfo[scene_time][k]['h'] = channel_gains[i,:,:].numpy()

    del scene
    print('loaded scene_time=',scene_time, ' process time=',time.time()-_time, ' s')
# 保存文件
with open(f"./sionna_result/trajectoryInfo_{start_time}_{end_time}.pkl", "wb") as tf:
    pickle.dump(trajectoryInfo,tf)

# import ipdb;ipdb.set_trace()


# new_orientation = (np.pi, 0,  0)
# car.orientation = type(car.orientation)(new_orientation, device=car.orientation.device)