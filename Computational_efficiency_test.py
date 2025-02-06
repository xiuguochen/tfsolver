from tfsolver import parallel_mm44
import scipy.io as sio
import torch
import time
import gc

def time_test(num_stacks):
    num_layers = 10  # number of layers
    num_angles = 1  # number of angles
    num_wavelengths = 100  # number of wavelengths

    wavelenths = torch.linspace(300, 1000, num_wavelengths, dtype=torch.cfloat,
                                device=device) * 1e-7  # Define the spectral range, unit in cm

    Theta = torch.linspace(55, 75, num_angles, dtype=torch.cfloat,
                           device=device) * torch.pi / 180  # Define the incident angle, unit in rad

    # create D  (10~500)
    D = (torch.rand((num_stacks, num_layers), dtype=torch.float, device=device).type(torch.cfloat) * (
                500 - 10) + 10) * 1e-7

    # define material properties
    Ni = torch.ones(num_wavelengths, dtype=torch.cfloat, device=device)  # set the complex refractive index of the Air
    Nt = torch.ones(num_wavelengths, dtype=torch.cfloat,
                    device=device)  # set the complex refractive index of the Substrate

    # create N (1~3)
    N = torch.zeros((num_stacks, num_layers, num_wavelengths, 3, 3), dtype=torch.cfloat, device=device)
    N[:, :, :, 0, 0] = torch.rand((num_stacks, num_layers, num_wavelengths), dtype=torch.float, device=device).type(
        torch.cfloat) * 2 + 1
    N[:, :, :, 1, 1] = torch.rand((num_stacks, num_layers, num_wavelengths), dtype=torch.float, device=device).type(
        torch.cfloat) * 2 + 1
    N[:, :, :, 2, 2] = torch.rand((num_stacks, num_layers, num_wavelengths), dtype=torch.float, device=device).type(
        torch.cfloat) * 2 + 1

    # create Euler_angles (0~pi/2)
    Euler_angles = torch.rand((num_stacks, num_layers, 3), dtype=torch.float, device=device).type(
        torch.cfloat) * torch.pi / 2

    start_time = time.time()
    O = parallel_mm44(N, Euler_angles, D, Ni, Nt, Theta, wavelenths, device=device)
    return time.time() - start_time

if __name__ == '__main__':
    torch.manual_seed(6)

    device = 'cpu'  # Define the device to run the simulation
    # device = 'cuda:0'

    time_list = []
    list = [1, 10]
    list1 = [i for i in range(100, 1000, 100)]
    list2 = [i for i in range(1000, 5001, 500)]
    num_stacks_list = list + list1 + list2

    for num_stacks in num_stacks_list:
        t = time_test(num_stacks)
        time_list.append(t)
        print("Execution time for %d stacks: %s seconds." % (num_stacks, str(t)))
        torch.cuda.empty_cache()
        gc.collect()
