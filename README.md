# TFSolver
TFSolver is a Python toolkit used for efficiently performing electromagnetic calculations of planar isotropic and anisotropic multilayer thin films, including amplitude reflection and transmission coefficients, reflectance, transmittance, light polarization states, ellipsometric parameters, and Mueller matrix spectra. TFSolver is based on the 4×4 matrix method and implemented using PyTorch.


## Features

 - **Multi-sample support**: It can simultaneously handle multiple multilayer samples, each with the same thin-film structure, but which can differ in material properties, material orientation and layer thickness.

 - **Multi-wavelength support**: It can analyze a set of wavelengths simultaneously, facilitating broadband optical studies.
   
 - **Multi-angle support**: It can process multiple incident angles at once for comprehensive angle analysis.
   
 - **GPU-accelerated** simulation
    
 -   Supporting **automatic differentiation** for optimization
 

## Installation
`TFSolver` package requires `PyTorch`, so it is recommended to install it first following [official instructions](https://pytorch.org/get-started/locally/).

Installation from [pipy.org](https://pypi.org/project/tfsolver/):
```sh
pip install tfsolver
```

Installation from [github.com](https://github.com/xiuguochen/tfsolver):
```sh
pip install git+https://github.com/xiuguochen/tfsolver
```


## Usage Guide

### Description of the input parameters
The key input parameters required to use the TFSolver are described in detail below.

**Ni**: Specifies the complex refractive indices of the isotropic incident medium at the relevant wavelengths, with a shape of [_w_], where _w_ is the number of wavelengths.

**Nt**: Defines the complex refractive indices of the isotropic exit medium at the relevant wavelengths, also with a shape of [_w_].

**N**: A tensor of shape [_s_, _l_, _w_, 3, 3] containing the complex refractive index tensors for each layer of each multilayer stack. Here, _s_ is the number of the thin-film stacks, and _l_ is the number of layers per stack. For example, N[_i_, _j_, _k_, :, :] represents the complex refractive index tensor of the (_j_+1)-th layer in the (_i_+1)-th stack at the (_k_+1)-th wavelength, noting that Python indexing starts at 0.

**Euler_angles**: Specifies the Euler angles for each layer, with a shape of [_s_, _l_, 3].

**D**: Defines the layer thickness for each film, in a shape of [_s_, _l_].

**Theta**: Represents the incident angles, with a shape of [_a_], where _a_ is the number of incident angles.

**wavelengths**: Sets the wavelengths under study.

**device**: Controls the computational device (CPU or GPU) on which the code will run.

**Jones_vector**: Defines the Jones vector of the incident light wave, with a shape of [2].

**mode**: Determines the simulation conditions (reflection, transmission, or both).

### Functions Overview
1. `parallel_mm44` calculates the amplitude reflection and transmission coefficients for a batch of multilayer thin-film samples at multiple incident angles and wavelengths. This function returns a dictionary, where each key corresponds to a tensor representing the amplitude reflection and transmission coefficients $r$ and $t$ of the multilayer thin-film stacks. For example, the key "r_sp" corresponds to the amplitude reflection coefficient $r_{sp}$, which represents reflected s-polarization generated by incident p-polarization when there is no incident s-polarization. Each coefficient is a tensor of datatype torch.cfloat with a shape of [_s_, _a_, _w_]. For example, r_sp[_i_, _j_, _k_] represents the amplitude reflection coefficient $r_{sp}$ for the (_i_+1)th multilayer thin film at the (_j_+1)th incident angle and the (_k_+1)th wavelength.
```python
def parallel_mm44(N, Euler_angles, D, Ni, Nt, Theta, wavelenths, device):
    …
   return {'r_pp': r_pp, 'r_sp': r_sp, 'r_ss': r_ss, 'r_ps': r_ps, 't_pp': t_pp, 't_sp': t_sp, 't_ss': t_ss, 't_ps': t_ps}
```
<br>

2. `calc_pol_state` calculates the Jones vector for the reflected or transmitted light from a batch of multilayer thin-film samples at multiple incident angles and wavelengths. The Jones vector describes the polarization state of the light. If the simulation mode is set to "r" (or "t"), this function will return the Jones vector for the reflected (or transmitted) light. If the simulation mode is set to "both", the function returns a dictionary where the keys "r" and "t" correspond to the Jones vectors for reflected and transmitted light, respectively. The returned tensors are of datatype torch.cfloat with a shape of [_s_, _a_, _w_, 2]. For example, in the case of reflection mode, Jv[_i_, _j_, _k_, :] represents the Jones vector of the reflected light for the (_i_+1)th multilayer thin film at the (_j_+1)th incident angle and the (_k_+1)th wavelength.
```python
def calc_pol_state(N, Euler_angles, D, Ni, Nt, Theta, wavelenths, Jones_vector, device, mode):
    …
    if mode == 'r':
        …
        return Jv
    elif mode == 't':
        …
        return Jv
    elif mode == 'both':
        …
        return {'r': Jv_r, 't': Jv_t}
```
<br>

3. `calc_intensity` calculates the reflectance and transmittance of a batch of multilayer thin-film samples at multiple incident angles and wavelengths. If the simulation mode is set to "r" (or "t"), the function will return the reflectance (or transmittance). If the simulation mode is set to "both", the function returns a dictionary where the keys "R", "Rp" and "Rs" correspond to the total reflectance, the ratio of reflected p-polarized light intensity to incident light intensity and the ratio of reflected s-polarized light intensity to incident light intensity, respectively. Each tensor in the dictionary is of datatype torch.cfloat with a shape of [s, a, w]. For example, Rp[i, j, k] represents the ratio of reflected p-polarized light intensity to incident light intensity for the the (_i_+1)th multilayer thin film at the (_j_+1)th incident angle and the (_k_+1)th wavelength.
```python
def calc_intensity (N, Euler_angles, D, Ni, Nt, Theta, wavelenths, Jones_vector, device, mode):
    …
    if mode == 'r':
        …
        return {'R': Rp + Rs, 'Rp': Rp, 'Rs': Rs}
    elif mode == 't':
        …
        return {'T': Tp + Ts, 'Tp': Tp, 'Ts': Ts}
    elif mode == 'both':
        …
        return {'R': Rp + Rs, 'Rp': Rp, 'Rs': Rs, 'T': Tp + Ts, 'Tp': Tp, 'Ts': Ts}
```
<br>

4. `calc_Ellips_param` calculates the ellipsometric parameters (amplitude ratio and phase difference) for a batch of multilayer thin-film samples at multiple incident angles and wavelengths. If the simulation mode is set to "r" (or "t"), the function returns a dictionary, where each key corresponds to an ellipsometric parameter for the reflection (or transmission) mode. If the simulation mode is set to "both", the function returns a dictionary where the keys "r" and "t" correspond to the ellipsometric parameters in reflection and transmission mode, respectively. For example, the keys "Psi_ps" and "Delta_ps" correspond to the amplitude ratio and phase difference of the amplitude reflection coefficients $r_{ps}$ and $r_{ss}$, respectively. Each ellipsometric parameter is a tensor of datatype torch.cfloat with a shape of [_s_, _a_, _w_]. For example, Psi_ps[_i_, _j_, _k_] represents the amplitude ratio of $r_{ps}$ to $r_{ss}$ for the (_i_+1)th multilayer thin film at the (_j_+1)th incident angle and the (_k_+1)th wavelength.
```python
def calc_Ellips_param (N, Euler_angles, D, Ni, Nt, Theta, wavelenths, device, mode):
    …
    if mode == 'r':
        …
        return {'Psi_pp': Psi_pp, 'Psi_ps': Psi_ps, 'Psi_sp': Psi_sp, 'Delta_pp': Delta_pp, 'Delta_ps': Delta_ps, 'Delta_sp': Delta_sp}
    elif mode == 't':
        …
        return {'Psi_pp': Psi_pp, 'Psi_ps': Psi_ps, 'Psi_sp': Psi_sp, 'Delta_pp': Delta_pp, 'Delta_ps': Delta_ps, 'Delta_sp': Delta_sp}
    elif mode == 'both':
        …
        return {'r’: {…}, ‘t’: {…}}
```
<br>

5. `calc_Mueller_matrix` calculates the Mueller matrices for a batch of multilayer thin-film samples at multiple incident angles and wavelengths. If the simulation mode is set to "r" (or "t"), the function returns the Mueller matrices for the reflection (or transmission) mode. If the simulation mode is set to "both," the function returns a dictionary containing the Mueller matrices for both reflection and transmission modes under the keys "r" and "t". Each tensor has datatype torch.cfloat with a shape of [_s_, _a_, _w_, 4, 4]. For example, in reflection mode, MM[_i_, _j_, _k_, :, :] represents the Mueller matrix for the (_i_+1)th multilayer thin film at the (_j_+1)th incident angle and the (_k_+1)th wavelength.
```python
def calc_Mueller_matrix (N, Euler_angles, D, Ni, Nt, Theta, wavelenths, device, mode):
    …
    if mode == 'r':
        …
        return MM
    elif mode == 't':
        …
        return MM
    elif mode == 'both':
        …
        return {'r’: MM_r, ‘t’: MM_t}
```

## TFSolver Examples
1. [Example 1](https://github.com/xiuguochen/tfsolver/blob/main/Case_Polarization_Sensitive_Absorber.py): Calculation of reflectance for a two-layer structure absorber.
    
2. [Example 2](https://github.com/xiuguochen/tfsolver/blob/main/Case_multilayer_reflector.py): Simulation of reflectance for a multilayer reflector.
    
3. [Example 3](https://github.com/xiuguochen/tfsolver/blob/main/Case_Mueller_Matrix.py): Calculation of Mueller Matrix spectra for an anisotropic material.
    
4. [Example 4](https://github.com/xiuguochen/tfsolver/blob/main/Computational_efficiency_test.py): Computational efficiency evaluation using a set of randomly generated thin films with a 10-layer structure.

5. [Example 5](https://github.com/xiuguochen/tfsolver/blob/main/Case_gradient_test.py): Gradient-based optimization of a three-layer thin film.


## Citation
If you use the code from this repository for your projects, please cite the following:

TFSolver: A numerical Python toolkit for parallel electromagnetic calculation of planar multilayer thin films at multi-wavelength and multi-angle,
Xiuguo Chen, 2025.
Available at: [GitHub repository link](https://github.com/xiuguochen/tfsolver).

## Acknowledgements
This work was supported by National Natural Science Foundation of China (52441511, 62175075, 52130504).
