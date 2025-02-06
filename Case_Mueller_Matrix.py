from tfsolver import calc_Mueller_matrix
import scipy.io as sio
import torch
import time
from torch.utils.benchmark import Timer

device = 'cpu' # Define the device to run the simulation

wavelenths = torch.linspace(200, 1600, 1401, dtype=torch.cfloat, device=device) * 1e-7 # Define the spectral range, unit in cm
num_wavelengths = wavelenths.shape[0]

Theta = torch.linspace(65, 75, 2, dtype=torch.cfloat, device=device) * torch.pi / 180 # Define the incident angle, unit in rad

mode = 'r'  # 'r' for reflectance

num_stacks = 1    # number of stacks
num_layers = 1  # number of layers

# define material properties
Ni = torch.ones(num_wavelengths, dtype=torch.cfloat, device=device) # set the complex refractive index of the Air

data = sio.loadmat('material_data_MM_silica.mat')
Nt = data['n'] + 1j * data['k']
Nt = torch.tensor(Nt, dtype=torch.cfloat, device=device).view(-1) # set the complex refractive index of the SiO2

# create N
N = torch.zeros((num_stacks, num_layers, num_wavelengths, 3, 3), dtype=torch.cfloat, device=device)
# set the refractive index of the anisotropic layer
data = sio.loadmat('material_data_MM.mat')
Nxx = data['n_x'] + 1j * data['k_x']
Nyy = data['n_y'] + 1j * data['k_y']
Nzz = data['n_z'] + 1j * data['k_z']
N[:, 0, :, 0, 0] = torch.tensor(Nxx, dtype=torch.cfloat, device=device)
N[:, 0, :, 1, 1] = torch.tensor(Nyy, dtype=torch.cfloat, device=device)
N[:, 0, :, 2, 2] = torch.tensor(Nzz, dtype=torch.cfloat, device=device)

# create D
D = torch.empty((num_stacks, num_layers), dtype=torch.cfloat, device=device)
D[:, 0] = 100 * 1e-7

# create Euler_angles
Euler_angles = torch.tensor([[[30, 40, 0]]], dtype=torch.cfloat, device=device) * torch.pi / 180 # Define the euler angle, unit in rad

start_time = time.time()
MM = calc_Mueller_matrix(N, Euler_angles, D, Ni, Nt, Theta, wavelenths, device='cpu', mode='r')
print("Execution time: %s seconds." % (str(time.time() - start_time)))
# Execution time: 0.013962507247924805 seconds.

timer =Timer(stmt='calc_Mueller_matrix(N, Euler_angles, D, Ni, Nt, Theta, wavelenths, device="cpu", mode="r")',
             globals={"N": N, "Euler_angles": Euler_angles, "D": D, "Ni": Ni, "Nt": Nt, "Theta": Theta,
                      "wavelenths": wavelenths, "calc_Mueller_matrix": calc_Mueller_matrix}, num_threads=1)
time_cost = timer.timeit(number=100)
print(time_cost)

# Mueller matrix with incident angle 65 degree
MM65 = MM[:,0,:]
# Mueller matrix with incident angle 75 degree
MM75 = MM[:,1,:]

sio.savemat('Case_MM_calculation.mat', {'MM65': MM65.squeeze().reshape(-1,16).transpose(1,0).cpu().numpy(),
                                        'MM75': MM75.squeeze().reshape(-1,16).transpose(1,0).cpu().numpy()})
