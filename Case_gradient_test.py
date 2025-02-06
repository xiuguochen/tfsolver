from tfsolver import calc_pol_state, calc_intensity
import torch
from torch import optim

device = 'cpu' # Define the device to run the simulation

wavelenths = torch.tensor([0.0108], dtype=torch.cfloat, device=device) # Define the spectral range, unit in cm
num_wavelengths = wavelenths.shape[0]

Theta = torch.tensor([0], dtype=torch.cfloat, device=device) # Define the incident angle, unit in rad

mode = 't'  # 't' for transmission mode

num_stacks = 1    # number of stacks
num_layers = 3  # number of layers

# define material properties
Ni = torch.ones(num_wavelengths, dtype=torch.cfloat, device=device) # Define the refractive index of the incident medium
Nt = torch.ones(num_wavelengths, dtype=torch.cfloat, device=device) # Define the refractive index of the exit medium

# create N
N = torch.zeros((num_stacks, num_layers, num_wavelengths, 3, 3), dtype=torch.cfloat, device=device)
Na = 2.9639 + 1j * 2.5165
Nb = 7.0436 + 1j * 1.5456

Nsi = 3.46

N[:, 0::2, :, 0, 0] = torch.tensor(Na, dtype=torch.cfloat, device=device)
N[:, 0::2, :, 1, 1] = torch.tensor(Nb, dtype=torch.cfloat, device=device)
N[:, 0::2, :, 2, 2] = torch.tensor(Nb, dtype=torch.cfloat, device=device)

N[:, 1, :, 0, 0] = torch.tensor(Nsi, dtype=torch.cfloat, device=device)
N[:, 1, :, 1, 1] = torch.tensor(Nsi, dtype=torch.cfloat, device=device)
N[:, 1, :, 2, 2] = torch.tensor(Nsi, dtype=torch.cfloat, device=device)

# create D
D = torch.empty((num_stacks, num_layers), dtype=torch.cfloat, device=device)
D[:, 0::2] = 30 * 1e-7
D[:, 1] = 2 * 1e-4

# create Euler_angles
Euler_angles = torch.zeros((num_stacks, num_layers, 3), dtype=torch.cfloat, device=device) # Define the euler angle, unit in rad

Jones_vector = torch.tensor([1., 1.], dtype=torch.cfloat, device=device) / torch.sqrt(torch.tensor(2.0))  # Define the Jones vector

# define the initial Azimuthal angle
Azimuthal_angle = torch.tensor([20.], dtype=torch.float, device=device, requires_grad=True)

# define the optimizer
optimizer = optim.Adam([Azimuthal_angle], lr=1)

# define the number of optimization steps
epoch = 100

# perform the optimization
for i in range(epoch):
    Azimuthal_angles = Azimuthal_angle * torch.pi / 180
    Euler_angles[:,-1, 0] = Azimuthal_angles + 0*1j

    # conduct the simulation
    O = calc_intensity(N, Euler_angles, D, Ni, Nt, Theta, wavelenths, Jones_vector, device=device, mode=mode)
    T = O['T'].squeeze()

    # calculate the loss
    loss = (T - torch.ones_like(T))**2

    # perform the optimization step
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()


    print(f"Azimuthal angle: {(Azimuthal_angles.item())*180/torch.pi}")

Azimuthal_angles = Azimuthal_angle * torch.pi / 180
Euler_angles[:,-1, 0] = Azimuthal_angles + 0*1j
O = calc_intensity(N, Euler_angles, D, Ni, Nt, Theta, wavelenths, Jones_vector, device=device, mode=mode)
T = O['T'].squeeze()
