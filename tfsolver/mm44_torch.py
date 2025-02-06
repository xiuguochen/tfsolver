from torch import sin, cos, conj, sqrt, atan, atan2, abs, pi, tan, exp, real, asin, imag
import torch

def parallel_mm44(N, Euler_angles, D, Ni, Nt, Theta, wavelenths, device='cpu'):
    """
    Compute the reflection and transmission coefficients for a set of thin-film stacks at different wavelengths and
    incidence angles.
    :param N: Tensor
        PyTorch Tensor of shape [S x L x W x 3 x 3] with complex or real entries which contain the
        refractive index tensors for each layer of each thin film at the wavelengths of interest:
        S is the number of multi-layer thin films, L is the number of layers for each thin film, W is the number of
        wavelength considered.
        For example, [:, :, :, 0, 0], [:, :, :, 1, 1], [:, :, :, 2, 2] represents Nx, Ny, Nz, which are the
        complex refractive indices along x, y, z axes.
    :param Euler_angles: Tensor
        Contains the rotation angles for coordinate transformations
        Euler is of shape [S x L x 3] and holds the Euler angles [rad] for coordinate transformation.
    :param D: Tensor
        Holds the layer thicknesses of the individual layers for a bunch of thin films in centimeter.
        D is of shape [S x L] with real-valued entries
    :param Ni: Tensor
        Holds the complex refractive index of the incident medium at the wavelengths of interest.
        Ni is of shape [W] with real-valued entries
   :param Nt: Tensor
        Holds the complex refractive index of the exit medium at the wavelengths of interest.
        Nt is of shape [W] with real-valued entries
    :param Theta: Tensor
        Theta is a tensor that determines the angles with which the light propagates in the incident medium.
        Theta is of shape [A] and holds the incidence angles [rad] in its entries.
    :param wavelenths: Tensor
        Vacuum wavelengths for optical calculations.
        It is of shape [W] and holds the wavelengths in centimeter.
    :param device: Str
        Computation device, accepts ether 'cuda' or 'cpu'; GPU acceleration can lower the computational time especially
        for computation involving large tensors
    :return: Dict
        Keys:
            'r_pp' : Tensor of the ratio of reflected p-polarization to incident p-polarization when there is
            no incident s-polarization for each stack (over angle and wavelength)
            'r_ps' : Tensor  of the ratio of reflected p-polarization to incident s-polarization when there is
            no incident p-polarization for each stack (over angle and wavelength)
            'r_sp' : Tensor of the ratio of reflected s-polarization to incident p-polarization when there is
            no incident s-polarization for each stack (over angle and wavelength)
            'r_ss' : Tensor of the ratio of reflected s-polarization to incident s-polarization when there is
            no incident p-polarization for each stack (over angle and wavelength)
            't_pp' : Tensor of the ratio of transmitted p-polarization to incident p-polarization when there is
             no incident s-polarization for each stack (over angle and wavelength)
            't_ps' : Tensor of the ratio of transmitted p-polarization to incident s-polarization when there is
             no incident p-polarization for each stack (over angle and wavelength)
            't_sp' : Tensor of the ratio of transmitted s-polarization to incident p-polarization when there is
             no incident s-polarization for each stack (over angle and wavelength)
            't_ss' : Tensor of the ratio of transmitted s-polarization to incident s-polarization when there is
             no incident p-polarization for each stack (over angle and wavelength)
            Each of these tensors or arrays is of shape [S x A x W]
    """
    # Get Dimensions
    num_stacks = D.shape[0]
    num_layers = D.shape[1]
    num_angles = Theta.shape[0]
    num_wavelengths = wavelenths.shape[0]

    # Calculate Sine and Cosine of Incident Angles
    sin_theta = sin(Theta)
    cos_theta = cos(Theta)

    # Calculate the delta matrix
    Kxx = torch.einsum('w,a->aw', Ni, sin_theta)    # Related to the x component of the incident wave vector
    Q = _calc_epsilon_tensor(N, Euler_angles)   # Calculate laboratory dielectric tensor
    delta_matrix = _calc_delta_matrix(Kxx, Q)

    # Compute partial transfer matrices for each Layer
    exp_term = torch.einsum('w,sl,salwij->salwij', 2 * pi / wavelenths, -D, delta_matrix)
    Tp_list = _calc_expm(1j * exp_term)

    # Initialize transfer matrices T for each sample
    T = torch.eye(4, dtype=torch.cfloat, device=device).expand(num_stacks, num_angles, num_wavelengths, -1, -1)

    # Accumulate Each Layer's partial transfer matrices for each stack at each angle and wavelength
    for i in range(num_layers):
        T = torch.einsum('sawij,sawjk->sawik', T, Tp_list[:, :, i, :, :, :])

    # Calculate transition matrices for incident and exit media
    Li_inverse = _calc_Li_inverse(Ni, cos_theta, num_stacks, num_angles, num_wavelengths, device)
    Lt = _calc_Lt(Ni, Nt, sin_theta, num_stacks, num_angles, num_wavelengths, device)

    # Final calculation of transfer matrix for each sample
    T = torch.einsum('sawij,sawjk,sawko->sawio', Li_inverse, T, Lt)

    # Calculate amplitude reflection and transmission coefficients:
    r_pp, r_sp, r_ss, r_ps, t_pp, t_sp, t_ss, t_ps = _calc_refl_trans_coeffs(T)

    return {'r_pp': r_pp, 'r_sp': r_sp, 'r_ss': r_ss, 'r_ps': r_ps,
            't_pp': t_pp, 't_sp': t_sp, 't_ss': t_ss, 't_ps': t_ps}

def calc_pol_state(N, Euler_angles, D, Ni, Nt, Theta, wavelenths, Jones_vector, device='cpu', mode='r'):
    '''
        Compute the Jones_vector for a given set of thin-film stacks at a set of wavelengths and incidence angles.
        :param N: Tensor
            PyTorch Tensor of shape [S x L x W x 3 x 3] with complex or real entries which contain the
            refractive index tensors for each layer of each thin film at the wavelengths of interest:
            S is the number of multi-layer thin films, L is the number of layers for each thin film, W is the number of
            wavelength considered.
            For example, [:, :, :, 0, 0], [:, :, :, 1, 1], [:, :, :, 2, 2] represents Nx, Ny, Nz, which are the
            complex refractive indices along x, y, z axes.
        :param Euler_angles: Tensor
            Contains the rotation angles for coordinate transformations
            Euler is of shape [S x L x 3] and holds the Euler angles [rad] for coordinate transformation.
        :param D: Tensor
            Holds the layer thicknesses of the individual layers for a bunch of thin films in centimeter.
            D is of shape [S x L] with real-valued entries
        :param Ni: Tensor
            Holds the complex refractive index of the incident medium at the wavelengths of interest.
            Ni is of shape [W] with real-valued entries
       :param Nt: Tensor
            Holds the complex refractive index of the exit medium at the wavelengths of interest.
            Nt is of shape [W] with real-valued entries
        :param Theta: Tensor
            Theta is a tensor that determines the angles with which the light propagates in the incident medium.
            Theta is of shape [A] and holds the incidence angles [rad] in its entries.
        :param wavelenths: Tensor
            Vacuum wavelengths for optical calculations.
            It is of shape [W] and holds the wavelengths in centimeter.
        :param Jones_vector: Tensor
            Holds the Jones vector for the incident light.
            It is of shape [2]. For example, [1, 0] represents the p-polarization and [0, 1] represents the s-polarization.
        :param device: Str
            Computation device, accepts ether 'cuda' or 'cpu'; GPU acceleration can lower the computational time especially
            for computation involving large tensors
        :param mode: str
            'r' for reflection mode, 't' for transmission mode, 'both' for both modes
        :return: Tensor or Dict
            If mode is 'r' or 't', returns a tensor of shape [S x A x W x 2] with the Jones vectors for each sample
            If mode is 'both', returns a dictionary with two keys 'r' and 't' which hold the Jones vectors for each
            mode for each sample.
        '''

    O = parallel_mm44(N, Euler_angles, D, Ni, Nt, Theta, wavelenths, device=device)

    if mode == 'r':
        Ep = O['r_pp'] * Jones_vector[0] + O['r_ps'] * Jones_vector[1]
        Es = O['r_sp'] * Jones_vector[0] + O['r_ss'] * Jones_vector[1]

        factor = sqrt(abs(Ep) ** 2 +  abs(Es) ** 2)
        phase_difference = atan2(imag(Ep), real(Ep)) - atan2(imag(Es), real(Es))

        return torch.stack((abs(Ep)/factor*(cos(phase_difference) + 1j*sin(phase_difference)),
                            abs(Es)/factor), dim=-1)

    elif mode == 't':
        Ep = O['t_pp'] * Jones_vector[0] + O['t_ps'] * Jones_vector[1]
        Es = O['t_sp'] * Jones_vector[0] + O['t_ss'] * Jones_vector[1]

        factor = sqrt(abs(Ep) ** 2 +  abs(Es) ** 2)
        phase_difference = atan2(imag(Ep), real(Ep)) - atan2(imag(Es), real(Es))

        return torch.stack((abs(Ep)/factor*(cos(phase_difference) + 1j*sin(phase_difference)),
                            abs(Es)/factor), dim=-1)

    elif mode == 'both':
        Ep_r = O['r_pp'] * Jones_vector[0] + O['r_ps'] * Jones_vector[1]
        Es_r = O['r_sp'] * Jones_vector[0] + O['r_ss'] * Jones_vector[1]
        factor_r = sqrt(abs(Ep_r) ** 2 +  abs(Es_r) ** 2)
        phase_difference_r = atan2(imag(Ep_r), real(Ep_r)) - atan2(imag(Es_r), real(Es_r))

        Ep_t = O['t_pp'] * Jones_vector[0] + O['t_ps'] * Jones_vector[1]
        Es_t = O['t_sp'] * Jones_vector[0] + O['t_ss'] * Jones_vector[1]
        factor_t = sqrt(abs(Ep_t) ** 2 +  abs(Es_t) ** 2)
        phase_difference_t = atan2(imag(Ep_t), real(Ep_t)) - atan2(imag(Es_t), real(Es_t))

        return {'r': torch.stack((abs(Ep_r)/factor_r*(cos(phase_difference_r) + 1j*sin(phase_difference_r)),
                                 abs(Es_r)/factor_r), dim=-1),
                't': torch.stack((abs(Ep_t)/factor_t*(cos(phase_difference_t) + 1j*sin(phase_difference_t)),
                                 abs(Es_t)/factor_t), dim=-1)}

def calc_intensity(N, Euler_angles, D, Ni, Nt, Theta, wavelenths, Jones_vector, device='cpu', mode='r'):
    '''
        Compute the Intensity for a given set of thin-film stacks at a set of wavelengths and incidence angles.
        :param N: Tensor
            PyTorch Tensor of shape [S x L x W x 3 x 3] with complex or real entries which contain the
            refractive index tensors for each layer of each thin film at the wavelengths of interest:
            S is the number of multi-layer thin films, L is the number of layers for each thin film, W is the number of
            wavelength considered.
            For example, [:, :, :, 0, 0], [:, :, :, 1, 1], [:, :, :, 2, 2] represents Nx, Ny, Nz, which are the
            complex refractive indices along x, y, z axes.
        :param Euler_angles: Tensor
            Contains the rotation angles for coordinate transformations
            Euler is of shape [S x L x 3] and holds the Euler angles [rad] for coordinate transformation.
        :param D: Tensor
            Holds the layer thicknesses of the individual layers for a bunch of thin films in centimeter.
            D is of shape [S x L] with real-valued entries
        :param Ni: Tensor
            Holds the complex refractive index of the incident medium at the wavelengths of interest.
            Ni is of shape [W] with real-valued entries
       :param Nt: Tensor
            Holds the complex refractive index of the exit medium at the wavelengths of interest.
            Nt is of shape [W] with real-valued entries
        :param Theta: Tensor
            Theta is a tensor that determines the angles with which the light propagates in the incident medium.
            Theta is of shape [A] and holds the incidence angles [rad] in its entries.
        :param wavelenths: Tensor
            Vacuum wavelengths for optical calculations.
            It is of shape [W] and holds the wavelengths in centimeter.
        :param Jones_vector: Tensor
            Holds the Jones vector for the incident light.
            It is of shape [2]. For example, [1, 0] represents the p-polarization and [0, 1] represents the s-polarization.
        :param device: Str
            Computation device, accepts ether 'cuda' or 'cpu'; GPU acceleration can lower the computational time especially
            for computation involving large tensors
        :param mode: str
            'r' for reflection mode, 't' for transmission mode, 'both' for both modes
        :return: Dict
            Keys:
                'R' : Tensor of the ratio of reflected intensity to incident intensity for each stack (over angle and wavelength)
                'Rp' : Tensor of the ratio of reflected p-polarization intensity to incident intensity for each stack (over angle and wavelength)
                'Rs' : Tensor of the ratio of reflected s-polarization intensity to incident intensity for each stack (over angle and wavelength)
                'T' : Tensor of the ratio of transmitted intensity to incident intensity for each stack (over angle and wavelength)
                'Tp' : Tensor of the ratio of transmitted p-polarization intensity to incident intensity for each stack (over angle and wavelength)
                'Ts' : Tensor of the ratio of transmitted s-polarization intensity to incident intensity for each stack (over angle and wavelength)
            Each of these tensors or arrays is of shape [S x A x W]
        '''

    O = parallel_mm44(N, Euler_angles, D, Ni, Nt, Theta, wavelenths, device=device)

    I = Jones_vector[0] * conj(Jones_vector[0]) + Jones_vector[1] * conj(Jones_vector[1])
    if mode == 'r':
        Ep = O['r_pp'] * Jones_vector[0] + O['r_ps'] * Jones_vector[1]
        Es = O['r_sp'] * Jones_vector[0] + O['r_ss'] * Jones_vector[1]
        Rp = real(Ep * conj(Ep) / I)
        Rs = real(Es * conj(Es) / I)

        return {'R': Rp + Rs, 'Rp': Rp, 'Rs': Rs}

    elif mode == 't':
        Theta_t = asin(torch.einsum('w,a->aw', Ni / Nt, sin(Theta)))
        factor = real((Nt * cos(Theta_t)) / (Ni * cos(Theta)))

        Ep = O['t_pp'] * Jones_vector[0] + O['t_ps'] * Jones_vector[1]
        Es = O['t_sp'] * Jones_vector[0] + O['t_ss'] * Jones_vector[1]
        Tp = real(Ep * conj(Ep) / I) * factor
        Ts = real(Es * conj(Es) / I) * factor

        return {'T': Tp + Ts, 'Tp': Tp, 'Ts': Ts}

    elif mode == 'both':
        Ep = O['r_pp'] * Jones_vector[0] + O['r_ps'] * Jones_vector[1]
        Es = O['r_sp'] * Jones_vector[0] + O['r_ss'] * Jones_vector[1]
        Rp = real(Ep * conj(Ep) / I)
        Rs = real(Es * conj(Es) / I)

        Theta_t = asin(torch.einsum('w,a->aw', Ni / Nt, sin(Theta)))
        factor = real((Nt * cos(Theta_t)) / (Ni * cos(Theta)))
        Ep = O['t_pp'] * Jones_vector[0] + O['t_ps'] * Jones_vector[1]
        Es = O['t_sp'] * Jones_vector[0] + O['t_ss'] * Jones_vector[1]
        Tp = real(Ep * conj(Ep) / I) * factor
        Ts = real(Es * conj(Es) / I) * factor

        return {'R': Rp + Rs, 'Rp': Rp, 'Rs': Rs, 'T': Tp + Ts, 'Tp': Tp, 'Ts': Ts}

def calc_Ellips_param(N, Euler_angles, D, Ni, Nt, Theta, wavelenths, device='cpu', mode='r'):
    '''
    Compute the Muller matrix for a given set of thin-film stacks at a set of wavelengths and incidence angles.
    :param N: Tensor
        PyTorch Tensor of shape [S x L x W x 3 x 3] with complex or real entries which contain the
        refractive index tensors for each layer of each thin film at the wavelengths of interest:
        S is the number of multi-layer thin films, L is the number of layers for each thin film, W is the number of
        wavelength considered.
        For example, [:, :, :, 0, 0], [:, :, :, 1, 1], [:, :, :, 2, 2] represents Nx, Ny, Nz, which are the
        complex refractive indices along x, y, z axes.
    :param Euler_angles: Tensor
        Contains the rotation angles for coordinate transformations
        Euler is of shape [S x L x 3] and holds the Euler angles [rad] for coordinate transformation.
    :param D: Tensor
        Holds the layer thicknesses of the individual layers for a bunch of thin films in centimeter.
        D is of shape [S x L] with real-valued entries
    :param Ni: Tensor
        Holds the complex refractive index of the incident medium at the wavelengths of interest.
        Ni is of shape [W] with real-valued entries
   :param Nt: Tensor
        Holds the complex refractive index of the exit medium at the wavelengths of interest.
        Nt is of shape [W] with real-valued entries
    :param Theta: Tensor
        Theta is a tensor that determines the angles with which the light propagates in the incident medium.
        Theta is of shape [A] and holds the incidence angles [rad] in its entries.
    :param wavelenths: Tensor
        Vacuum wavelengths for optical calculations.
        It is of shape [W] and holds the wavelengths in centimeter.
    :param device: Str
        Computation device, accepts ether 'cuda' or 'cpu'; GPU acceleration can lower the computational time especially
        for computation involving large tensors
    :param mode: str
        'r' for reflection mode, 't' for transmission mode, 'both' for both modes
    :return: Dict
        If mode is 'r' or 't', returns a dictionary with the following keys:
        'Psi_pp', 'Psi_ps', 'Psi_sp', 'Delta_pp', 'Delta_ps', 'Delta_sp'
        which hold the ellipsometry parameters for the given mode.
        If mode is 'both', returns a dictionary with two keys 'r' and 't' and the corresponding ellipsometry parameters for each mode.
    '''

    O = parallel_mm44(N, Euler_angles, D, Ni, Nt, Theta, wavelenths, device=device)

    if mode == 'r':
        return _Jones_matrix_to_Ellips_param(O['r_pp'], O['r_ps'], O['r_sp'], O['r_ss'])
    elif mode == 't':
        return _Jones_matrix_to_Ellips_param(O['t_pp'], O['t_ps'], O['t_sp'], O['t_ss'])
    elif mode == 'both':
        return {'r': _Jones_matrix_to_Ellips_param(O['r_pp'], O['r_ps'], O['r_sp'], O['r_ss']),
                't': _Jones_matrix_to_Ellips_param(O['t_pp'], O['t_ps'], O['t_sp'], O['t_ss'])}

def calc_Mueller_matrix(N, Euler_angles, D, Ni, Nt, Theta, wavelenths, device='cpu', mode='r'):
    '''
    Compute the Muller matrix for a given set of thin-film stacks at a set of wavelengths and incidence angles.
    :param N: Tensor
        PyTorch Tensor of shape [S x L x W x 3 x 3] with complex or real entries which contain the
        refractive index tensors for each layer of each thin film at the wavelengths of interest:
        S is the number of multi-layer thin films, L is the number of layers for each thin film, W is the number of
        wavelength considered.
        For example, [:, :, :, 0, 0], [:, :, :, 1, 1], [:, :, :, 2, 2] represents Nx, Ny, Nz, which are the
        complex refractive indices along x, y, z axes.
    :param Euler_angles: Tensor
        Contains the rotation angles for coordinate transformations
        Euler is of shape [S x L x 3] and holds the Euler angles [rad] for coordinate transformation.
    :param D: Tensor
        Holds the layer thicknesses of the individual layers for a bunch of thin films in centimeter.
        D is of shape [S x L] with real-valued entries
    :param Ni: Tensor
        Holds the complex refractive index of the incident medium at the wavelengths of interest.
        Ni is of shape [W] with real-valued entries
   :param Nt: Tensor
        Holds the complex refractive index of the exit medium at the wavelengths of interest.
        Nt is of shape [W] with real-valued entries
    :param Theta: Tensor
        Theta is a tensor that determines the angles with which the light propagates in the incident medium.
        Theta is of shape [A] and holds the incidence angles [rad] in its entries.
    :param wavelenths: Tensor
        Vacuum wavelengths for optical calculations.
        It is of shape [W] and holds the wavelengths in centimeter.
    :param device: Str
        Computation device, accepts ether 'cuda' or 'cpu'; GPU acceleration can lower the computational time especially
        for computation involving large tensors
    :param mode: str
        'r' for reflection mode, 't' for transmission mode, 'both' for both modes
    :return: Tensor or Dict
        If mode is 'r' or 't', returns a tensor of shape [S x A x W x 4 x 4] with the Muller matrix for the given mode.
        If mode is 'both', returns a dictionary with two keys 'r' and 't' and the corresponding Muller matrices for each mode.
    '''

    O = parallel_mm44(N, Euler_angles, D, Ni, Nt, Theta, wavelenths, device=device)

    if mode == 'r':
        return _Jones_matrix_to_Mueller_matrix(O['r_pp'], O['r_ps'], O['r_sp'], O['r_ss'])
    elif mode == 't':
        return _Jones_matrix_to_Mueller_matrix(O['t_pp'], O['t_ps'], O['t_sp'], O['t_ss'])
    elif mode == 'both':
        return {'r': _Jones_matrix_to_Mueller_matrix(O['r_pp'], O['r_ps'], O['r_sp'], O['r_ss']),
                't': _Jones_matrix_to_Mueller_matrix(O['t_pp'], O['t_ps'], O['t_sp'], O['t_ss'])}

def _Jones_matrix_to_Ellips_param(w_pp, w_ps, w_sp, w_ss):
    '''
    Compute the ellipsometry parameters from the Jones matrix.
    '''
    rho_pp = w_pp / w_ss
    rho_ps = w_ps / w_ss
    rho_sp = w_sp / w_ss

    Psi_pp = atan(abs(rho_pp))
    Psi_ps = atan(abs(rho_ps))
    Psi_sp = atan(abs(rho_sp))

    Delta_pp = atan2(rho_pp.imag, rho_pp.real)
    Delta_ps = atan2(rho_ps.imag, rho_ps.real)
    Delta_sp = atan2(rho_sp.imag, rho_sp.real)
    return {'Psi_pp': Psi_pp, 'Psi_ps': Psi_ps, 'Psi_sp': Psi_sp,
            'Delta_pp': Delta_pp, 'Delta_ps': Delta_ps, 'Delta_sp': Delta_sp}

def _Jones_matrix_to_Mueller_matrix(w_pp, w_ps, w_sp, w_ss):
    '''
    Compute the Muller matrix according to the Jones matrix.
    '''
    O = _Jones_matrix_to_Ellips_param(w_pp, w_ps, w_sp, w_ss)
    w_pp = tan(O['Psi_pp']) * exp(-1j * O['Delta_pp'])
    w_ps = tan(O['Psi_ps']) * exp(-1j * O['Delta_ps'])
    w_sp = tan(O['Psi_sp']) * exp(-1j * O['Delta_sp'])
    w_ss = torch.ones_like(w_ss)

    E1 = torch.real(w_pp * conj(w_pp))
    E2 = torch.real(w_ss * conj(w_ss))
    E3 = torch.real(w_ps * conj(w_ps))
    E4 = torch.real(w_sp * conj(w_sp))
    JJ12 = w_pp * conj(w_ss)
    JJ13 = w_pp * conj(w_ps)
    JJ14 = w_pp * conj(w_sp)
    JJ23 = w_ss * conj(w_ps)
    JJ24 = w_ss * conj(w_sp)
    JJ34 = w_ps * conj(w_sp)

    MM = torch.empty((w_pp.shape[0], w_pp.shape[1], w_pp.shape[2], 4, 4), dtype=torch.float32, device=w_pp.device)
    MM[:, :, :, 0, 0] = 0.5 * (E1 + E2 + E3 + E4)
    MM[:, :, :, 0, 1] = 0.5 * (E1 - E2 - E3 + E4)
    MM[:, :, :, 0, 2] = JJ13.real + JJ24.real
    MM[:, :, :, 0, 3] = JJ13.imag - JJ24.imag
    MM[:, :, :, 1, 0] = 0.5 * (E1 - E2 + E3 - E4)
    MM[:, :, :, 1, 1] = 0.5 * (E1 + E2 - E3 - E4)
    MM[:, :, :, 1, 2] = JJ13.real - JJ24.real
    MM[:, :, :, 1, 3] = JJ13.imag + JJ24.imag
    MM[:, :, :, 2, 0] = JJ14.real + JJ23.real
    MM[:, :, :, 2, 1] = JJ14.real - JJ23.real
    MM[:, :, :, 2, 2] = JJ12.real + JJ34.real
    MM[:, :, :, 2, 3] = JJ12.imag - JJ34.imag
    MM[:, :, :, 3, 0] = -JJ14.imag + JJ23.imag
    MM[:, :, :, 3, 1] = -JJ14.imag - JJ23.imag
    MM[:, :, :, 3, 2] = -JJ12.imag - JJ34.imag
    MM[:, :, :, 3, 3] = JJ12.real - JJ34.real

    MM /= MM[:, :, :, 0, 0].unsqueeze(-1).unsqueeze(-1)

    return MM

def _calc_Li_inverse(Ni, cos_theta, num_stacks, num_angles, num_wavelengths, device):
    '''
    Calculates the incident matrix.
    '''

    N0_cos_theta = torch.einsum('w,a->aw', Ni, cos_theta)
    Li_inverse = torch.zeros((num_stacks, num_angles, num_wavelengths, 4, 4), dtype=torch.cfloat, device=device)
    Li_inverse[:, :, :, :2, 1:2] = 1
    Li_inverse[:, :, :, 0, 2] = -1 / N0_cos_theta
    Li_inverse[:, :, :, 1, 2] = 1 / N0_cos_theta
    Li_inverse[:, :, :, 2, 0] = 1 / cos_theta.unsqueeze(-1)
    Li_inverse[:, :, :, 3, 0] = -1 / cos_theta.unsqueeze(-1)
    Li_inverse[:, :, :, 2:, 3] = 1 / Ni.unsqueeze(-1)
    return Li_inverse / 2

def _calc_Lt(Ni, Nt, sin_theta, num_stacks, num_angles, num_wavelengths, device):
    '''
    Calculates the exit matrix.
    '''

    cos_theta_t = sqrt(1 - torch.einsum('w,a->aw', (Ni / Nt) ** 2, sin_theta ** 2))

    Lt = torch.zeros((num_stacks, num_angles, num_wavelengths, 4, 4), dtype=torch.cfloat, device=device)
    Lt[:, :, :, 0, 2] = cos_theta_t
    Lt[:, :, :, 1, 0] = 1
    Lt[:, :, :, 2, 0] = torch.einsum('w,aw->aw', -Nt, cos_theta_t)
    Lt[:, :, :, 3, 2] = Nt

    return Lt

def _calc_refl_trans_coeffs(T):
    '''
    Calculate the amplitude reflection and transmission coefficients for each sample
    :param T: tensor of shape [S x A x W x 4 x 4]
        Holds the transfer matrices for each sample at each angle and wavelength
    :return: tuple of tensors of shape [S x A x W]
        Holds the amplitude reflection and transmission coefficients for each stack at each angle and wavelength
    :return:
    '''

    r_denom = T[:, :, :, 0, 0] * T[:, :, :, 2, 2] - T[:, :, :, 0, 2] * T[:, :, :, 2, 0]
    r_pp = (T[:, :, :, 0, 0] * T[:, :, :, 3, 2] - T[:, :, :, 0, 2] * T[:, :, :, 3, 0]) / r_denom
    r_sp = (T[:, :, :, 0, 0] * T[:, :, :, 1, 2] - T[:, :, :, 0, 2] * T[:, :, :, 1, 0]) / r_denom
    r_ss = (T[:, :, :, 1, 0] * T[:, :, :, 2, 2] - T[:, :, :, 1, 2] * T[:, :, :, 2, 0]) / r_denom
    r_ps = (T[:, :, :, 2, 2] * T[:, :, :, 3, 0] - T[:, :, :, 2, 0] * T[:, :, :, 3, 2]) / r_denom

    t_pp = T[:, :, :, 0, 0] / r_denom
    t_sp = -T[:, :, :, 0, 2] / r_denom
    t_ss = T[:, :, :, 2, 2] / r_denom
    t_ps = -T[:, :, :, 2, 0] / r_denom

    return r_pp, r_sp, r_ss, r_ps, t_pp, t_sp, t_ss, t_ps

def _calc_delta_matrix(Kxx, Q):
    '''
    Calculates the delta matrix.
    :param Kxx: tensor of shape [A W]
        Holds the x component of the incident wave vector
    :param Q: tensor of shape [S L W 3 3]
        Holds the laboratory dielectric tensor for each layer
    :return: tensor of shape [S A L W 4 4]
        Holds the delta matrix for each stack, layer, and wavelength
    '''
    # Get dimensions
    num_angles = Kxx.shape[0]
    num_wavelengths = Kxx.shape[1]
    num_stacks = Q.shape[0]
    num_layers = Q.shape[1]

    # Initialize delta matrix
    delta_matrix = torch.zeros((num_stacks, num_angles, num_layers, num_wavelengths, 4, 4), dtype=torch.cfloat, device=Q.device)
    delta_matrix[:, :, :, :, 0, 0] = torch.einsum('aw,slw->salw', -Kxx, Q[:, :, :, 2, 0] / Q[:, :, :, 2, 2])
    delta_matrix[:, :, :, :, 0, 1] = torch.einsum('aw,slw->salw', -Kxx, Q[:, :, :, 2, 1] / Q[:, :, :, 2, 2])
    delta_matrix[:, :, :, :, 0, 3] = 1 - torch.einsum('aw,slw->salw', Kxx ** 2, 1 / Q[:, :, :, 2, 2])
    delta_matrix[:, :, :, :, 1, 2] = -1
    delta_matrix[:, :, :, :, 2, 0] = (Q[:, :, :, 1, 2] * Q[:, :, :, 2, 0] / Q[:, :, :, 2, 2] - Q[:, :, :, 1, 0]).unsqueeze(1)
    delta_matrix[:, :, :, :, 2, 1] = Kxx.unsqueeze(1) ** 2 + (-Q[:, :, :, 1, 1] + Q[:, :, :, 1, 2] * Q[:, :, :, 2, 1] / Q[:, :, :, 2, 2]).unsqueeze(1)
    delta_matrix[:, :, :, :, 2, 3] = torch.einsum('aw,slw->salw', Kxx, Q[:, :, :, 1, 2] / Q[:, :, :, 2, 2])
    delta_matrix[:, :, :, :, 3, 0] = (Q[:, :, :, 0, 0] - Q[:, :, :, 0, 2] * Q[:, :, :, 2, 0] / Q[:, :, :, 2, 2]).unsqueeze(1)
    delta_matrix[:, :, :, :, 3, 1] = (Q[:, :, :, 0, 1] - Q[:, :, :, 0, 2] * Q[:, :, :, 2, 1] / Q[:, :, :, 2, 2]).unsqueeze(1)
    delta_matrix[:, :, :, :, 3, 3] = torch.einsum('aw,slw->salw', -Kxx, Q[:, :, :, 0, 2] / Q[:, :, :, 2, 2])
    return delta_matrix

def _calc_epsilon_tensor(N, Euler_angles):
    '''
    Calculates the laboratory dielectric tensors for each layer.
    :param N: tensor of shape [S L W 3 3]
        Holds the complex refractive indices for each layer
    :param Euler_angles: tensor of shape [S L 3]
        Contains the rotation angles for coordinate transformations
    :return: tensor of shape [S L W 3 3]
         Holds the laboratory dielectric tensor for each layer
    '''

    # [S L 3 3]
    B_t = torch.empty((Euler_angles.shape[0], Euler_angles.shape[1], 3, 3), dtype=torch.cfloat, device=Euler_angles.device)
    cos_psi = cos(Euler_angles[:, :, 2])
    sin_psi = sin(Euler_angles[:, :, 2])
    cos_theta = cos(Euler_angles[:, :, 1])
    sin_theta = sin(Euler_angles[:, :, 1])
    cos_phi = cos(Euler_angles[:, :, 0])
    sin_phi = sin(Euler_angles[:, :, 0])
    B_t[:, :, 0, 0] = cos_phi * cos_psi - sin_phi * cos_theta * sin_psi
    B_t[:, :, 0, 1] = -cos_phi * sin_psi - sin_phi * cos_theta * cos_psi
    B_t[:, :, 0, 2] = sin_phi * sin_theta
    B_t[:, :, 1, 0] = sin_phi * cos_psi + cos_phi * cos_theta * sin_psi
    B_t[:, :, 1, 1] = -sin_phi * sin_psi + cos_phi * cos_theta * cos_psi
    B_t[:, :, 1, 2] = -cos_phi * sin_theta
    B_t[:, :, 2, 0] = sin_theta * sin_psi
    B_t[:, :, 2, 1] = sin_theta * cos_psi
    B_t[:, :, 2, 2] = cos_theta
    B = B_t.transpose(-1, -2)

    Q = torch.einsum('slij,slwjk,slko->slwio', B_t, N**2, B)

    return Q

def _calc_expm(A):
    '''
    Compute the matrix exponential for each matrix in the last two dimensions of a multi-dimensional tensor
    using the Pade approximation.

    :param A: Tensor
        Input tensor of shape (..., m, m), where each slice along the first dimension is a square matrix.
    :return: Tensor
        Output tensor of the same shape with matrix exponentials computed for each slice.
    '''

    Original_shape = A.shape
    A = A.view(-1, Original_shape[-2], Original_shape[-1])  # Reshape to (batch_size, m, m)

    # Get the batch size and matrix size
    batch_size, m, _ = A.shape

    # Scale A by a power of 2 so that its infinity norm is < 1
    norm_A = torch.norm(A, p=float('inf'), dim=(-2, -1))    # Compute norm over last two dimensions
    _, e = torch.frexp(torch.max(norm_A))   # Get the exponent of the norm
    s = torch.max(torch.zeros_like(e), e)   # Get scaling factor
    A_scaled = A / (2 ** s.item())  # Scale A

    # Initialize output tensor E
    E = torch.eye(m, dtype=A.dtype, device=A.device).expand(batch_size, -1, -1).clone()
    D = E.clone()

    E += 0.5 * A_scaled
    D -= 0.5 * A_scaled

    X = A_scaled.clone()
    c = 1 / 2
    q = 8 # Pade approximation order
    p = 1 # Flag to alternate addition/subtraction

    # Perform the Pade series approximation for each matrix
    for k in range(2, q + 1):
        c *= (q - k + 1) / (k * (2 * q - k + 1))
        X = torch.matmul(A_scaled, X)
        cX = c * X
        E += cX
        D += cX if p else -cX
        p = 1 - p   # Toggle between 1 and 0

    # Solve the system D * E = E 
    E = torch.linalg.solve(D, E)

    # Undo the scaling by repeated squaring (s times)
    if s.item() > 0:
        E = torch.matrix_power(E, 2 ** s.item())

    return E.view(Original_shape)

