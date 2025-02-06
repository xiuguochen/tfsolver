from tfsolver import calc_intensity
import scipy.io as sio
import torch
import time

device = 'cpu' # Define the device to run the simulation

wavelenths = torch.tensor([600], dtype=torch.cfloat, device=device) * 1e-7 # Define the spectral range, unit in cm
num_wavelengths = wavelenths.shape[0]

Theta = torch.linspace(10, 80, 71, dtype=torch.cfloat, device=device) * torch.pi / 180 # Define the incident angle, unit in rad

mode = 'r'  # 'r' for reflectance

#
no = 1.83
ne = 1.46

num_stacks = 1  # number of stacks
num_layers = 37  # number of layers

# define material properties
Ni = torch.tensor([1.33], dtype=torch.cfloat, device=device) # set the complex refractive index of the Air
Nt = torch.tensor([1.33], dtype=torch.cfloat, device=device) # set the complex refractive index of the BK7

# create N
N = torch.zeros((num_stacks, num_layers, num_wavelengths, 3, 3), dtype=torch.cfloat, device=device)
# set the refractive index of the layer A
N[:, ::2, :, 0, 0] = torch.tensor(ne, dtype=torch.cfloat, device=device)
N[:, ::2, :, 1, 1] = torch.tensor(no, dtype=torch.cfloat, device=device)
N[:, ::2, :, 2, 2] = torch.tensor(no, dtype=torch.cfloat, device=device)

# set the refractive index of the layer B
N[:, 1::2, :, 0, 0] = torch.tensor(no, dtype=torch.cfloat, device=device)
N[:, 1::2, :, 1, 1] = torch.tensor(no, dtype=torch.cfloat, device=device)
N[:, 1::2, :, 2, 2] = torch.tensor(ne, dtype=torch.cfloat, device=device)

# create D
D = torch.empty((num_stacks, num_layers), dtype=torch.cfloat, device=device)
D[:, ::2] = 100 * 1e-7
D[:, 1::2] = 150 * 1e-7

# create Euler_angles
Euler_angles = torch.zeros((num_stacks, num_layers, 3), dtype=torch.cfloat, device=device)
Euler_angles[:, ::2, 0] = torch.linspace(0, torch.pi/2, D[:, ::2].shape[1], dtype=torch.cfloat, device=device)

start_time = time.time()
Jones_vector = torch.tensor([1., 0.], dtype=torch.cfloat, device=device)
O = calc_intensity(N, Euler_angles, D, Ni, Nt, Theta, wavelenths, Jones_vector, device=device, mode=mode)
Rp_p = O['Rp']
Rs_p = O['Rs']
Jones_vector = torch.tensor([0., 1.], dtype=torch.cfloat, device=device)
O = calc_intensity(N, Euler_angles, D, Ni, Nt, Theta, wavelenths, Jones_vector, device=device, mode=mode)
Rp_s = O['Rp']
Rs_s = O['Rs']

print("Execution time: %s seconds." % (str(time.time() - start_time)))

sio.savemat('Multilayer_reflector.mat', {'Rp_p': Rp_p.squeeze().cpu().numpy(),
                                         'Rs_p': Rs_p.squeeze().cpu().numpy(),
                                         'Rp_s': Rp_s.squeeze().cpu().numpy(),
                                         'Rs_s': Rs_s.squeeze().cpu().numpy(),
                                         'Theta': torch.real(Theta).squeeze().cpu().numpy()})
