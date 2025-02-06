from tfsolver import calc_intensity
import scipy.io as sio
import torch
import time

device = 'cpu' # Define the device to run the simulation

wavelenths = torch.linspace(900, 1300, 401, dtype=torch.cfloat, device=device) * 1e-7 # Define the spectral range, unit in cm
num_wavelengths = wavelenths.shape[0]

Theta = torch.tensor([0.], dtype=torch.cfloat, device=device) # Define the incident angle, unit in rad

mode = 'r'  # 'r' for reflectance

thickness_list = torch.linspace(100, 400, 61, dtype=torch.cfloat, device=device) * 1e-7 # investigated thickness range
num_stacks = thickness_list.shape[0]     # number of stacks
num_layers = 2  # number of layers

# create D
D = torch.empty((num_stacks, num_layers), dtype=torch.cfloat, device=device)
D[:, 0] = thickness_list    # set the thickness of the GeSe layer
D[:, 1] = 200 * 1e-7  # set the thickness of the SiO2 layer

# define material properties
Ni = torch.ones(num_wavelengths, dtype=torch.cfloat, device=device) # set the complex refractive index of the Air
data = sio.loadmat('material_data_Si.mat')
Nt = data['n'] + 1j * data['k']
Nt = torch.tensor(Nt, dtype=torch.cfloat, device=device).view(-1) # set the complex refractive index of the Si

# create N
N = torch.zeros((num_stacks, num_layers, num_wavelengths, 3, 3), dtype=torch.cfloat, device=device)
# set the complex refractive index of the GeSe layer
data = sio.loadmat('material_data_GeSe.mat')
Nxx = data['n_zz'] + 1j * data['k_zz']
Nyy = data['n_ac'] + 1j * data['k_ac']
Nzz = data['n_lay']
N[:, 0, :, 0, 0] = torch.tensor(Nxx, dtype=torch.cfloat, device=device)
N[:, 0, :, 1, 1] = torch.tensor(Nyy, dtype=torch.cfloat, device=device)
N[:, 0, :, 2, 2] = torch.tensor(Nzz, dtype=torch.cfloat, device=device)
# set the complex refractive index of the SiO2 layer
data = sio.loadmat('material_data_SiO2.mat')
N_SiO2 = data['n']
N[:, 1, :, 0, 0] = torch.tensor(N_SiO2, dtype=torch.cfloat, device=device)
N[:, 1, :, 1, 1] = torch.tensor(N_SiO2, dtype=torch.cfloat, device=device)
N[:, 1, :, 2, 2] = torch.tensor(N_SiO2, dtype=torch.cfloat, device=device)

# create Euler_angles
Euler_angles = torch.zeros((num_stacks, num_layers, 3), dtype=torch.float, device=device)

start_time = time.time()
# p-polarized light
Jones_vector = torch.tensor([1., 0.], dtype=torch.cfloat, device=device)
O = calc_intensity(N, Euler_angles, D, Ni, Nt, Theta, wavelenths, Jones_vector, device='cpu', mode='r')
Rp = O['R'].squeeze()

# s-polarized light
Jones_vector = torch.tensor([0., 1.], dtype=torch.cfloat, device=device)
O = calc_intensity(N, Euler_angles, D, Ni, Nt, Theta, wavelenths, Jones_vector, device='cpu', mode='r')
Rs = O['R'].squeeze()

print("Execution time: %s seconds." % (str(time.time() - start_time)))

sio.savemat('Absorber_Reflectance.mat', {'Rp': Rp.squeeze().cpu().numpy(), 'Rs': Rs.squeeze().cpu().numpy()})
