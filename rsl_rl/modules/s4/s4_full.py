"""Standalone version of Structured State Space sequence model (S4)."""
from collections import defaultdict
from typing import Optional, Mapping, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Function aliases
contract = torch.einsum

_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
_c2r = torch.view_as_real
_r2c = torch.view_as_complex
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()


"""Structured matrix kernels"""

# Fallback versions
def cauchy_naive(v, z, w):
    """
    v: (..., N)
    z: (..., L)
    w: (..., N)
    returns: (..., L) \sum v/(z-w)
    """
    v = _conj(v)
    w = _conj(w)
    cauchy_matrix = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1)) # (... N L)
    return torch.sum(cauchy_matrix, dim=-2)

def log_vandermonde_naive(v, x, L, conj=True):
    """
    v: (..., N)
    x: (..., N)
    returns: (..., L) \sum v x^l
    """
    vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x)) # (... N L)
    vandermonde_prod = contract('... n, ... n l -> ... l', v, vandermonde_matrix) # (... L)
    return 2*vandermonde_prod.real

def log_vandermonde_transpose_naive(u, v, x, L):
    vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x)) # (... N L)
    vandermonde_prod = contract('... l, ... n, ... n l -> ... n', u.to(x), v.to(x), vandermonde_matrix) # (... L)
    return vandermonde_prod



""" Simple nn.Module components """

def Activation(activation=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'softplus':
        return nn.diag()
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))

def LinearActivation(
        d_input, d_output, bias=True,
        activation=None,
        activate=False, # Apply activation as part of this module
        **kwargs,
    ):
    """Returns a linear nn.Module with control over axes order, initialization, and activation."""

    # Construct core module
    linear_cls = nn.Linear
    if activation is not None and activation == 'glu': d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    if activate and activation is not None:
        activation = Activation(activation, dim=-1)
        linear = nn.Sequential(linear, activation)
    return linear

"""Misc functional utilities"""

def power(L, A, v=None):
    """Compute A^L and the scan sum_i A^i v_i.

    A: (..., N, N)
    v: (..., N, L)
    """

    I = torch.eye(A.shape[-1]).to(A) # , dtype=A.dtype, device=A.device)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1: I = powers[-1] @ I
        L //= 2
        if L == 0: break
        l *= 2
        if v is None:
            powers = [powers[-1] @ powers[-1]]
        else:
            powers.append(powers[-1] @ powers[-1])

    if v is None: return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)

def dplr(
    init='hippo',
    N=64, rank=1, H=1,
    dtype=torch.float,
    real_random=False,
    real_scale=1.0,
    imag_random=False,
    imag_scale=1.0,
    B_init='constant',
    B_scale=1.0,
    P_scale=1.0,
    normalize=False,
):
    """Directly construct a DPLR matrix.

    Args:
    - init: (str) ['rand', 'lin', inv', 'real', 'hippo'] Choices for initialization of A.
          Most of these affect the imaginary part of A, except for 'real'.
    - real_random: (bool) Initialize A.real in -U[0, 1]. Otherwise, initialize to -1/2.
    - real_scale: (float) Scaling factor of real part of A.
    - imag_random: (bool) Initialize A.imag randomly.
    - imag_scale: (bool) Scaling factor of imaginary part of A.
    - B_init: (str) ['constant' | 'random' | 'alternating' | 'unit-cw' | 'unit-ccw' | 'hippo']
          Choices for initialization of B.
    - B_scale: (float) Scaling factor for B
    - P_scale: (float) Scaling factor for P
    - normalize: (bool) Apply an automatic normalization factor on B
    """
    assert dtype == torch.float or dtype == torch.double
    dtype = torch.cfloat if dtype == torch.float else torch.cdouble

    pi = torch.tensor(math.pi)

    # Construct real part of diagonal A (must be non-negative)
    if real_random:
        real_part = torch.rand(H, N//2)
    else:
        real_part = .5 * torch.ones(H, N//2)
    real_part = real_scale * real_part

    # Construct imaginary part of diagonal A (must be non-negative)
    if imag_random:
        imag_part = N//2 * torch.rand(H, N//2)
    else:
        imag_part = repeat(torch.arange(N//2), 'n -> h n', h=H)

    if init in ['random', 'rand']:
        imag_part = torch.exp(torch.randn(H, N//2))
    elif init == 'real':
        imag_part = 0 * imag_part
        if real_random:
            real_part = torch.rand(H, N//2) * N//2
        else:
            # This is the S4D-Real method described in the S4D paper
            # The A matrix is diag(-1, -2, ..., -N), which are the eigenvalues of the HiPPO matrix
            real_part = 1 + repeat(torch.arange(N//2), 'n -> h n', h=H)
    elif init in ['linear', 'lin']:
        imag_part = pi * imag_part
    elif init in ['inverse', 'inv']: # Based on asymptotics of the default HiPPO matrix
        imag_part = 1/pi * N * (N/(1+2*imag_part)-1)
    elif init in ['inverse2', 'inv2']:
        imag_part = 1/pi * N * (N/(1+imag_part)-1)
    elif init in ['quadratic', 'quad']:
        imag_part = 1/pi * (1+2*imag_part)**2
    else: raise NotImplementedError
    imag_part = imag_scale * imag_part

    # Construct diagonal A
    A = -real_part - 1j * imag_part  # Force negative real and imag
    assert torch.all(A.real < 1e-4) and torch.all(A.imag <= 0.0)  # Allow some tolerance for numerical precision on real part

    # Initialize B
    if B_init == 'constant':
        B = torch.ones(H, N//2, dtype=dtype)
    elif B_init == 'random':
        B = torch.randn(H, N//2, dtype=dtype)
    elif B_init == 'alternating':  # Seems to track 'constant' exactly for some reason
        B = torch.ones(H, N//4, 2, dtype=dtype)
        B[:, :, 1] *= -1
        B = B.view(H, N//2)
    elif B_init == 'unit-cw':
        z = torch.tensor(torch.exp(-2j * pi / N), dtype=dtype)
        B = z ** torch.arange(0, N // 2)
        B = repeat(B, 'n -> h n', h=H).clone().contiguous()
    elif B_init == 'unit-ccw':
        z = torch.tensor(torch.exp(2j * pi / N), dtype=dtype)
        B = z ** torch.arange(0, N // 2)
        B = repeat(B, 'n -> h n', h=H).clone().contiguous()
    else: raise NotImplementedError
    B *= B_scale

    # Experimental feature that appeared in earlier versions of HTTYH (not extensively tested)
    # Seems more principled for normalization theoretically, but seemed to hurt on PathX
    if normalize:
        norm = -B/A # (H, N) # Result if you integrate the kernel with constant 1 function
        zeta = 2*torch.sum(torch.abs(norm)**2, dim=-1, keepdim=True) # Variance with a random C vector
        B = B / zeta**.5

    # Initialize P
    if B_init in ['legs', 'hippo']:
        # P constructed earlier
        P = repeat(P, 'r n -> r h n', h=H).clone().contiguous()
    else:
        P = torch.randn(rank, H, N//2, dtype=dtype)
        P = P * P_scale

    # Initialize V (only used in testing)
    V = torch.eye(N, dtype=dtype)[:, :N//2]
    V = repeat(V, 'n m -> h n m', h=H)

    return A, P, B, V

def ssm(init, N, R, H, **ssm_args):
    """Dispatcher to create single SSM initialization

    N: state size
    R: rank (for DPLR parameterization)
    H: number of independent SSM copies
    """

    if init.startswith("diag"):
        ssm_args["P_scale"] = 0.0
    args = init[4:].split("-")
    assert args[0] == ""
    if len(args) > 1:
        ssm_args["init"] = args[1]
    A, P, B, V = dplr(N=N, rank=R, H=H, **ssm_args)
    return A, P, B, V

combinations = {
    'hippo': ['legs', 'fourier'],
    'diag': ['diag-inv', 'diag-lin'],
    'all': ['legs', 'fourier', 'diag-inv', 'diag-lin'],
}

def combination(inits, N, R, S, **ssm_args):
    if isinstance(inits, str):
        inits = combinations[inits] if inits in combinations else [inits]

    assert S % len(inits) == 0, f"{S} independent trainable SSM copies must be multiple of {len(inits)} different inits"
    A, P, B, V = zip(
        *[ssm(init, N, R, S // len(inits), **ssm_args) for init in inits]
    )
    A = torch.cat(A, dim=0) # (S N)
    P = torch.cat(P, dim=1) # (R S N)
    B = torch.cat(B, dim=0) # (S N)
    V = torch.cat(V, dim=0) # (S N N)
    return A, P, B, V


"""SSM convolution kernels"""

def inv_transform(param, transform='none'):
    """Initialize a (positive) parameter under a transform."""
    param = torch.clamp(param, min=1e-4)
    if transform == 'none':
        return param
    elif transform == 'exp':
        return torch.log(param) # Some of the HiPPO methods have real part 0
    elif transform == 'relu':
        return param
    elif transform == 'sigmoid':
        return torch.logit(param)
    elif transform == 'softplus':
        return torch.log(torch.exp(param)-1)
    else: raise NotImplementedError

def param_transform(param, transform='none'):
    """Get a (positive) parameter under a transform."""
    if transform == 'none':
        p = param
    elif transform == 'exp':
        p = torch.exp(param)
    elif transform == 'relu':
        # JAX version seems to NaN if you allow 0's, although this code was fine without it
        p = F.relu(param)+1e-4
    elif transform == 'sigmoid':
        p = F.sigmoid(param)
    elif transform == 'softplus':
        p = F.softplus(param)
    else: raise NotImplementedError
    return p

class Kernel(nn.Module):
    """Interface for modules that produce convolution kernels.

    A main distinction between these and normal Modules is that the forward pass
    does not take inputs. It is a mapping from parameters to a tensor that can
    be used in other modules, in particular as a convolution kernel.

    Because of the unusual parameterization, these kernels may often want special
    hyperparameter settings on their parameters. The `register` method provides
    an easy interface for controlling this, and is intended to be used with an
    optimizer hook that can be found in train.py or example.py.

    This class also defines an interface for interacting with kernels *statefully*,
    in particular for state space models (SSMs). This interface handles the setting
    when a model can be converted from a "CNN" into an "RNN".
    _setup_step()
    step()
    default_state()
    forward_state()

    See ConvKernel for the simplest instantiation of this interface.
    """

    def __init__(
        self,
        d_model: int = 0,
        channels: int = 1,
        l_max: Optional[int] = None,
        lr: Union[float, Optional[Mapping]] = None,
        wd: Union[float, Optional[Mapping]] = 0.0,
        **kwargs,
    ):
        """General interface.

        d_model (H): Model dimension, or number of independent convolution kernels created.
        channels (C): Extra dimension in the returned output (see .forward()).
            - One interpretation is that it expands the input dimension giving it C separate "heads" per feature.
              That is convolving by this kernel maps shape (B L D) -> (B L C D)
            - This is also used to implement a particular form of bidirectionality in an efficient way.
            - In general for making a more powerful model, instead of increasing C
              it is recommended to set channels=1 and adjust H to control parameters instead.
        l_max (L): Maximum kernel length (optional). If unspecified, most Kernel instantiations
            will return kernels of arbitrary length as passed into .forward().
        lr: Optional dictionary specifying special hyperparameters for .register().
            Passing in a number (e.g. 0.001) sets attributes of SSM parameters (A, B, dt).
            A custom optimizer hook is needed to configure the optimizer to set the learning rates appropriately for these parameters.
        wd: Same as lr, but for weight decay.
        """
        super().__init__()
        assert d_model > 0
        self.H = self.d_model = d_model
        self.L = self.l_max = l_max
        self.channels = channels
        self.lr = lr
        self.wd = wd

        # Add a catch-all **kwargs to make it easier to change kernels
        # without manually moving other options passed in the config.
        # Good to log these just so it's explicit.

        # Logic for registering parameters
        # Case 1: lr: None | float
        #   All params should have this lr (None means inherit from global lr)
        # Case 2: lr: dict
        #   Specified params should have that lr, all others should be None
        if self.lr is None or isinstance(self.lr, float):
            self.lr_dict = defaultdict(lambda: self.lr)
        else:
            self.lr_dict = defaultdict(lambda: None)
            self.lr_dict.update(self.lr)

        # Same logic for weight decay
        # (but is always just set to 0.0 and hasn't been ablated)
        if self.wd is None or isinstance(self.wd, float):
            self.wd_dict = defaultdict(lambda: self.wd)
        else:
            self.wd_dict = defaultdict(lambda: None)
            self.wd_dict.update(self.wd)

    def forward(self, state=None, rate=1.0, L=None):
        """General interface to generate a global convolution kernel.

        state: Initial state for recurrent updates.
            E.g. for SSMs, this should have shape (B, H, N) (batch, d_model, d_state).
        rate: Relative sampling rate.
        L: Target kernel length.

        Returns:
          - (C, H, L) (channels, d_model, l_kernel) The convolution kernel.
          - (B, H, L) (batch, d_model, l_kernel)
              Extra information for how the state affects the output of convolving by kernel.
        """
        raise NotImplementedError

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)

    def _setup_step(self, **kwargs):
        """Convert a model into a recurrent mode for autoregressive inference."""
        raise NotImplementedError

    def step(self, x, state, **kwargs):
        """Step the model for one timestep with input x and recurrent state."""
        raise NotImplementedError

    def default_state(self, *args, **kwargs):
        """Return a default initial state."""
        raise NotImplementedError

    @torch.no_grad()
    def forward_state(self, u, state):
        """Forward the state through a sequence, i.e. computes the state after passing chunk through the kernel."""
        raise NotImplementedError

    @property
    def d_state(self):
        """Implement this for interfaces that want to interact with a stateful layer (i.e. SSMs).

        Currently the only codepath that might use this is the StateDecoder, which is not used.
        """
        raise NotImplementedError

    @property
    def state_to_tensor(self):
        """Same as d_state, only needed for niche codepaths involving recurrent state."""
        raise NotImplementedError

class SSMKernel(Kernel):
    """Parent class for different SSM parameterizations.

    This class is abstract and only defines some initializations and flags that are common to all SSM variants.
    It is instantiated by subclasses SSMKernel{Dense,Real,Diag,DPLR}.

    Options:
    d_state (N): State size (dimensionality of parameters A, B, C). Generally shouldn't need to be adjusted and doens't affect speed much for most kernels (e.g. S4, S4D).
    deterministic: Use a deterministic initialization for dt, A, B, C.
        Useful for debugging as well as constructing a simple exponential decay kernel (e.g. used in S4ND image->video inflation).

    dt_min, dt_max: min and max values for the step size dt
    dt_tie: Keep dt tied across the N dimensions of the state. Although this theoretically makes more sense, models such as S5 and Mega have found slightly improvements by setting it to False.
    dt_transform: Transform function for parameterization of dt (default 'softplus', used to be 'exp')

    rank: Rank of low-rank correction for DPLR mode. Needs to be increased for init "legt".
    n_ssm: Number of independent trainable (A, B) SSMs, e.g.
        `n_ssm=1` means all A/B parameters are tied across the H different instantiations of C.
        `n_ssm=None` means all H SSMs are completely independent.
        Generally, changing this option can save parameters but doesn't affect performance or speed much.
        This parameter must divide H.
    init: Options for initialization of (A, B). For DPLR mode, recommendations are "legs", "fout", "hippo" (combination of both). For Diag mode, recommendations are "diag-inv", "diag-lin", "diag-legs", and "diag" (combination of diag-inv and diag-lin).
    init_args: Extra arguments passed into initialization function (see dplr.py for options).
    """

    def init_dt(self):
        # Generate dt
        if self.deterministic:  # Meant for debugging
            assert self.dt_tie, "Deterministic dt initialization is tied"
            assert self.dt_transform == 'exp', "Deterministic dt transform should be 'exp' for simplicity"
            inv_dt = torch.exp(torch.linspace(math.log(self.dt_min), math.log(self.dt_max), self.H)).unsqueeze(-1) # (H 1)
        else:
            shape = (self.H, 1) if self.dt_tie else (self.H, self.N//2)
            # Initialize log dt
            inv_dt = torch.rand(*shape, dtype=self.dtype) * (
                math.log(self.dt_max) - math.log(self.dt_min)
            ) + math.log(self.dt_min)
            if self.dt_transform != 'exp':
                inv_dt = inv_transform(torch.exp(inv_dt), self.dt_transform)

        return inv_dt

    def init_ssm_dplr(self):
        """Returns DPLR (A, P, B, C) parameters for init options."""
        A, P, B, V = combination(self.init, self.N, self.rank, self.n_ssm, **self.init_args)

        # Broadcast C to have H channels
        if self.deterministic:
            C = torch.zeros(self.channels, self.n_ssm, self.N, dtype=self.cdtype)
            C[:, :, :1] = 1.
            C = contract('hmn, chn -> chm', V.conj().transpose(-1, -2), C) # V^* C
            C = repeat(C, 'c t n -> c (v t) n', v=self.H // C.size(-2)).clone().contiguous()
        else:
            C = torch.randn(self.channels, self.H, self.N//2, dtype=self.cdtype)

        # Broadcast other parameters to have n_ssm copies
        assert self.n_ssm % B.size(-2) == 0 \
                and self.n_ssm % P.size(-2) == 0 \
                and self.n_ssm % A.size(-2) == 0

        # Broadcast tensors to n_ssm copies
        # These will be the parameters, so make sure tensors are materialized and contiguous
        B = repeat(B, 't n -> (v t) n', v=self.n_ssm // B.size(-2)).clone().contiguous()
        P = repeat(P, 'r t n -> r (v t) n', v=self.n_ssm // P.size(-2)).clone().contiguous()
        A = repeat(A, 't n -> (v t) n', v=self.n_ssm // A.size(-2)).clone().contiguous()

        # Because these complex parameterizations assume conjugate symmetry,
        # halve the value of self.N for convenience
        self.N //= 2

        return A, P, B, C

    def __init__(
        self,
        # General Kernel arguments for parent class
        d_model: int = 0,
        channels: int = 1,
        l_max: Optional[int] = None,
        lr: Union[float, Optional[Mapping]] = None,
        wd: Union[float, Optional[Mapping]] = 0.0,
        # SSM arguments
        d_state: int = 64,
        deterministic: bool = False,
        # dt options
        dt_min: float = 0.001,
        dt_max: float = 1.0,
        dt_tie: bool = True,
        dt_transform: str = 'softplus',
        # (A, B, C) options
        rank: int = 1,
        n_ssm: Optional[int] = None,
        init: Optional[str] = "diag",
        # Extra hyperparameters for initialization
        **init_args,
    ):
        super().__init__(d_model=d_model, channels=channels, l_max=l_max, lr=lr, wd=wd)
        self.N = d_state
        self.dtype, self.cdtype = torch.float, torch.cfloat
        self.deterministic = deterministic
        # dt options
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_tie = dt_tie
        self.dt_transform = dt_transform
        # SSM options (A, B, C)
        self.rank = rank
        self.n_ssm = n_ssm if n_ssm is not None else self.H

        self.init = init
        self.init_args = init_args

    @torch.no_grad()
    def forward_state(self, u, state):
        """Forward the state through a sequence, i.e. computes the state after passing chunk through SSM

        This is a generic version of this functionality that works for SSMs.
        It is currently used by SSMKernelDense and SSMKernelDPLR.
        This is a suboptimal implementation; it is recommended to use SSMKernelDiag
        if this functionality is desired.

        state: (B, H, N)
        u: (B, H, L)

        Returns: (B, H, N)
        """

        # Construct dA, dB matrices
        dA, dB = self._setup_state() # (H N N) (H N)

        conj = state.size(-1) != dA.size(-1)
        if conj: state = _conj(state)

        v = contract('h n, b h l -> b h n l', dB, u.flip(-1))
        AL, v = power(u.size(-1), dA, v)
        next_state = contract("h m n, b h n -> b h m", AL, state)
        next_state = next_state + v

        if conj: next_state = next_state[..., : next_state.size(-1) // 2]
        return next_state

    def _setup_state(self):
        """Register dA and dB to module."""
        raise NotImplementedError

    @property
    def d_state(self):
        """d_state and state_to_tensor are used by specific decoders.

        These were used in earlier versions and should not be needed in general.
        """
        return self.H * self.N

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)


class SSMKernelDiag(SSMKernel):
    """SSM kernel using diagonal state matrix (S4D model).

    Options:
    disc: ['zoh' | 'bilinear' ] Discretization options.
    dt_fast:  (experimental) Parameterize inv_dt under sinh function.
        (Ohno et al. "Fast Saturating Gate for Learning Long Time Scales with RNNs")
    real_transform, imag_transform: ['none' | 'exp' | 'relu' | 'sigmoid' | 'softplus']
        Parameterize the real/imag parts of the diagonal of A under this function.
    bandlimit: Mask high frequencies of the kernel (indices corresponding to
        diagonal elements with large imaginary part). Introduced in S4ND paper.
    is_real : Real-valued SSM; can be interpreted as EMA.
    """

    def __init__(
        self,
        disc: str = 'zoh',  # Change to 'bilinear' to match S4, but should make little difference either way
        dt_fast: bool = False,
        real_transform: str = 'exp',
        imag_transform: str = 'none',
        bandlimit: Optional[float] = None,
        is_real: bool = False,
        **kwargs,
    ):
        # Special case: for real-valued, d_state semantics change
        if is_real and 'd_state' in kwargs:
            kwargs['d_state'] = kwargs['d_state'] * 2
        super().__init__(**kwargs)
        self.disc = disc
        self.dt_fast = dt_fast
        self.real_transform = real_transform
        self.imag_transform = imag_transform
        self.bandlimit = bandlimit
        self.is_real = is_real

        # Initialize dt, A, B, C
        inv_dt = self.init_dt()
        A, P, B, C = self.init_ssm_dplr()
        # Note that in the Diag case, P will be ignored
        # The DPLR case subclasses this and uses P
        self.register_params(A, B, C, inv_dt, P)

    def register_params(self, A, B, C, inv_dt, P):
        """Process the initialization into form of trainable parameters.

        A: (S, N) diagonal matrix
        B: (S, N)
        C: (C, H, N)
        dt: (H) timescale per feature

        Dimensions:
        N (or d_state): state size
        H (or d_model): total SSM copies
        S (or n_ssm): number of trainable copies of (A, B, dt); must divide H
        C (or channels): system is 1-dim to C-dim

        The forward pass of this Module returns a tensor of shape (C, H, L)

        Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
        """

        if self.dt_fast: inv_dt = torch.asinh(inv_dt)

        # Rank of low-rank correction
        assert self.H == inv_dt.size(0)
        assert self.N == A.size(-1) == B.size(-1) == C.size(-1)
        assert self.n_ssm == A.size(-2) == B.size(-2) # Number of independent SSMs trained
        self.repeat = self.H // A.size(0)

        # Check that diagonal part has negative real and imag part
        # (allow some tolerance for numerical precision on real part
        # since it may be constructed by a diagonalization)
        assert torch.all(A.real < 1e-4) and torch.all(A.imag <= 0.0)

        # Broadcast everything to correct shapes
        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N))) # (C, H, N)  # TODO originally this was only in DPLR, check safe for Diag
        B = B.unsqueeze(0) # (1, H, N)
        assert self.channels == C.shape[0]

        # Register dt
        self.register("inv_dt", inv_dt, self.lr_dict['dt'], self.wd_dict['dt'])
        # Register ABC
        if self.is_real:
            self.register("C", C.real, self.lr_dict['C'], None)
            self.register("B", B.real, self.lr_dict['B'], self.wd_dict['B'])
            self.register("A_real", inv_transform(-A.real, self.real_transform), self.lr_dict['A'], self.wd_dict['A'])
        else:
            self.register("C", _c2r(_resolve_conj(C)), self.lr_dict['C'], None)
            self.register("B", _c2r(B), self.lr_dict['B'], self.wd_dict['B'])
            self.register("A_real", inv_transform(-A.real, self.real_transform), self.lr_dict['A'], self.wd_dict['A'])
            self.register("A_imag", inv_transform(-A.imag, self.imag_transform), self.lr_dict['A'], self.wd_dict['A'])

    def _get_params(self, rate=1.0):
        """Process the internal parameters."""

        # (S N) where S=n_ssm
        if self.is_real:
            A = -param_transform(self.A_real, self.real_transform)
            B = self.B # (1 S N)
            C = self.C # (C H N)
        else:
            A = -param_transform(self.A_real, self.real_transform) - 1j * param_transform(self.A_imag, self.imag_transform)
            B = _r2c(self.B) # (1 S N)
            C = _r2c(self.C) # (C H N)

        if self.dt_fast: inv_dt = torch.sinh(self.inv_dt)
        else: inv_dt = self.inv_dt
        dt = param_transform(inv_dt, self.dt_transform) * rate # (H N)

        if self.bandlimit is not None:
            freqs = dt / rate * A.imag.abs() / (2*math.pi) # (H N)
            mask = torch.where(freqs < self.bandlimit * .5, 1, 0)
            C = C * mask

        # Incorporate dt into A and B
        A = repeat(A, 't n -> (v t) n', v=self.repeat)  # (H N)
        B = repeat(B, 'b t n -> b (v t) n', v=self.repeat)  # (1 H N)

        # TODO: The downstream algorithm should only need to access dt*A
        # However the current DPLR kernel still uses dt and A separately
        # Once that is fixed, this should return dtA instead of dt and A
        dtA = dt * A  # (H N)

        return dt, A, B, C

    def forward(self, L, state=None, rate=1.0):
        """See Kernel.forward() for argument documentation."""

        dt, A, B, C = self._get_params(rate)
        dtA = dt * A

        # Augment B with state
        if state is not None:
            s = state / dt
            if self.disc == 'bilinear':
                s = s * (1. + dtA/2)
            elif self.disc == 'zoh':
                s = s * dtA * dtA.exp() / (dtA.exp() - 1.)
            B = torch.cat([s, B], dim=-3) # (1+B H N)


        # Combine B and C
        C = (B[:, None, :, :] * C).view(-1, self.H, self.N)

        # Dispatch which Vandermonde kernel to use
        log_vandermonde = log_vandermonde_naive

        # Main kernel
        if self.disc == 'zoh':
            # Power up
            C = C * (torch.exp(dtA)-1.) / A
            K = log_vandermonde(C, dtA, L) # (H L)
        elif self.disc == 'bilinear':
            C = C * (1. - dtA/2).reciprocal() * dt # or * dtA / A
            dA = (1. + dtA/2) / (1. - dtA/2)
            K = log_vandermonde(C, dA.log(), L)
        else: raise ValueError(f"Discretization {self.disc} not supported")

        K = K.view(-1, self.channels, self.H, L) # (1+B C H L)

        if state is not None:
            K_state = K[:-1, :, :, :] # (B C H L)
        else:
            K_state = None
        K = K[-1, :, :, :] # (C H L)

        return K, K_state

    def _setup_step(self):
        """Set up dA, dB, dC discretized parameters for stepping."""

        dt, A, B, C, = self._get_params()
        # Incorporate dt into A
        dtA = dt * A  # (H N)

        if self.disc == 'zoh':
            self.dA = torch.exp(dtA) # (H N)
            self.dB = B * (torch.exp(dtA)-1.) / A # (C H N)
        elif self.disc == 'bilinear':
            self.dA = (1. + dtA/2) / (1. - dtA/2)
            self.dB = B * (1. - dtA/2).reciprocal() * dt # or * dtA / A
        else: raise ValueError(f"Discretization {self.disc} not supported")
        self.dB = rearrange(self.dB, '1 h n -> h n')
        self.dC = C

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        next_state = contract("h n, b h n -> b h n", self.dA, state) \
                + contract("h n, b h -> b h n", self.dB, u)
        y = contract("c h n, b h n -> b c h", self.dC, next_state)
        return 2*y.real, next_state

    def forward_state(self, u, state):
        """Pass the state forward through an entire sequence."""
        self._setup_step()
        AL = self.dA ** u.size(-1)
        u = u.flip(-1).to(self.dA).contiguous() # (B H L)
        # Dispatch which Vandermonde kernel to use
        log_vandermonde_transpose = log_vandermonde_transpose_naive
        v = log_vandermonde_transpose(u, self.dB, self.dA.log(), u.size(-1))
        next_state = AL * state + v
        return next_state



kernel_registry = {
    's4d': SSMKernelDiag,
    'diag': SSMKernelDiag,
}

class FFTConv(nn.Module):
    """Implements an FFT Convolution around a convolution kernel.

    d_model (H): Model dimension (in CNN terminology, this would be "channels").
    l_max (L): The maximum kernel length. Set l_max=None to always use a global kernel.
    channels: Can be interpreted as a number of "heads"; the SSM is a map from a 1-dim to C-dim sequence. It's not recommended to change this; instead, increase d_model for larger models.
    activation: Activation after the full convolution.
    mode: Which kernel algorithm to use. 'nplr' is the full S4 model; 'diag' is the simpler S4D. Other options can be found in the kernel registry.

    kernel_args: See the class .kernel.SSMKernel for the kernel constructor which accepts kernel_args. Relevant options that are worth considering and tuning include "mode", "init", "dt_min", "dt_max", "lr"
    """

    def __init__(
        self,
        d_model,
        l_max=None,
        channels=1,
        swap_channels=False,
        activation='gelu', # Activation after layer
        dropout=0.0,
        drop_kernel=0.0,
        mode='diag',
        kernel=None,
        **kernel_args,  # Arguments passed into inner convolution kernel
    ):
        super().__init__()
        self.d_model = d_model
        self.L = self.l_max = l_max
        self.channels = channels
        self.swap_channels = swap_channels

        if activation is not None and activation.startswith('glu'):
            channels *= 2
        self.activation = Activation(activation, dim=-1)

        self.D = nn.Parameter(torch.randn(channels, self.d_model))

        # Inner convolution kernel
        if mode is not None:
            assert kernel is None, "Pass either mode or kernel but not both"
            kernel, mode = mode, kernel
        kernel_cls = kernel_registry[kernel]
        self.kernel = kernel_cls(
            d_model=self.d_model,
            l_max=self.l_max,
            channels=channels,
            **kernel_args,
        )

        dropout_fn = nn.Dropout
        self.drop = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_kernel = nn.Dropout(drop_kernel) if drop_kernel > 0.0 else nn.Identity()

    def forward(self, x, state=None, rate=1.0, **kwargs): # absorbs return_output and transformer src mask
        """
        x: (B L D)
        """

        # Always work with (B D L) dimension in this module
        x = x.transpose(-1, -2)
        L = x.size(-1)

        # Compute SS Kernel
        l_kernel = L if self.L is None else min(L, round(self.L / rate))
        k, k_state =  self.kernel(L=l_kernel, rate=rate, state=state) # (C H L) (B C H L)

        # Kernel dropout
        k = self.drop_kernel(k)

        # In principle, we could pad to l_kernel+L-1 instead of l_kernel+L, but we choose the latter for
        # equational simplicity. Additionally, we have not experimented to compare the efficiency of the two.
        k_f = torch.fft.rfft(k, n=l_kernel+L) # (C H L)
        x_f = torch.fft.rfft(x, n=l_kernel+L) # (B H L)
        y_f = contract('bhl,chl->bchl', x_f, k_f)
        y = torch.fft.irfft(y_f, n=l_kernel+L)[..., :L] # (B C H L)


        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', x, self.D)

        # Compute state update
        if state is not None:
            y = y + k_state #
            next_state = self.kernel.forward_state(x, state)
        else:
            next_state = None


        # Reshape to flatten channels
        if self.swap_channels:
            y = rearrange(y, 'b c h l -> b (h c) l')
        else:
            y = rearrange(y, 'b c h l -> b (c h) l')

        y = self.drop(y)

        y = y.transpose(-1, -2)
        y = self.activation(y)

        return y, next_state


    def setup_step(self, **kwargs):
        self.kernel._setup_step(**kwargs)

    def step(self, x, state):
        """ Step one time step as a recurrent model. Intended to be used during validation.

        x: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """

        y, next_state = self.kernel.step(x, state) # (B C H)
        y = y + x.unsqueeze(-2) * self.D
        y = rearrange(y, 'b c h -> b (c h)')
        y = self.activation(y)
        return y, next_state

    def default_state(self, *batch_shape):
        # kernel is not a SequenceModule so it doesn't need to adhere to same interface
        # the kernel will know the device of its own parameters
        return self.kernel.default_state(*batch_shape)

    @property
    def d_output(self):
        return self.d_model * self.channels


class S4Block(nn.Module):
    """General block design wrapping an inner layer. Currently only layer=FFTConv is supported, but easy to incorporate others.

    Arguments:
    - bottleneck: Reduce dimension of inner layer (e.g. used in GSS).
    - gate: Add multiplicative gating (e.g. used in GSS), which is essentially a multiplicative instead of additive residual branch.
    - gate_act: Activation function to apply on the gate residual branch.
    - mult_act: Activation function to apply after gate multiplication (e.g. GELU in GSS).
    - final_act: Activation function to apply after final linear layer. 'id' for no activation, None for no linear layer at all.

    - transposed: Choose backbone axis ordering of (B, L, H) (if False) or (B, H, L) (if True) [B=batch size, L=sequence length, H=model dimension]

    Other options are all experimental and should not need to be configured.
    """

    def __init__(
        self,
        d_model,
        bottleneck=None,
        gate=None,
        gate_act=None,
        mult_act=None,
        final_act='gelu',
        dropout=0.0,
        **layer_args,  # Arguments into inner layer (e.g. FFTConv)
    ):
        super().__init__()

        self.d_model = d_model

        self.gate = gate
        self.bottleneck = bottleneck

        if bottleneck is not None:
            self.d_model = self.d_model // bottleneck
            self.input_linear = LinearActivation(
                self.d_model,
                self.d_model,
                activation=None,
                activate=False,
            )

        if gate is not None:
            self.input_gate = LinearActivation(
                self.d_model,
                self.d_model * gate,
                activation=gate_act,
                activate=True,
            )
            if self.layer.d_output != self.d_model * gate:
                self.output_gate = LinearActivation(
                    self.d_model*self.channels,
                    self.d_model * gate,
                    activation=None,
                    activate=False,
                )

        # Currently this module only uses FFTConv for its inner module
        # But the options here are all agnostic to the inner block
        # If other types of inner layers are desired, it is easy
        # to add an option to swap a different module in
        self.layer = FFTConv(d_model, dropout=dropout, **layer_args)

        # Pointwise operations

        # Activation after (optional) multiplication by gate branch
        self.mult_activation = Activation(mult_act)
        
        dropout_fn = nn.Dropout
        self.drop = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        if final_act is None:
            self.output_linear = nn.Identity()
        else:
            # position-wise output transform to mix features
            self.output_linear = LinearActivation(
                self.d_model*gate if gate is not None else self.layer.d_output,
                self.d_model,
                activation=final_act,
                activate=True,
            )



    def forward(self, x, lengths=None, **kwargs): # absorbs return_output and transformer src mask
        """
        x: (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as x
        """
        L = x.size(1)

        # Mask out padding tokens
        # TODO handle option for mask - instead of lengths, which assumes suffix padding
        if isinstance(lengths, int):
            if lengths != L:
                lengths = torch.tensor(lengths, dtype=torch.long, device=x.device)
            else:
                lengths = None
        if lengths is not None:
            assert isinstance(lengths, torch.Tensor) and lengths.ndim == 1 and lengths.size(0) in [1, x.size(0)]
            mask = torch.where(torch.arange(L, device=lengths.device)[:, None] < lengths[:, None, None], 1., 0.)
            x = x * mask

        if self.gate is not None:
            v = self.input_gate(x)
        if self.bottleneck is not None:
            x = self.input_linear(x)

        y, state = self.layer(x, **kwargs)


        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.mult_activation(y)
        y = self.drop(y)
        y = self.output_linear(y)

        return y, state

    def setup_step(self, **kwargs):
        self.layer.setup_step(**kwargs)

    def step(self, x, state):
        """Step one time step as a recurrent model. Intended to be used during validation.

        x: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """

        if self.gate is not None:
            v = self.input_gate(x)
        if self.bottleneck is not None:
            x = self.input_linear(x)
        y, next_state = self.layer.step(x, state) # (B C H)
        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.mult_activation(y)
        y = self.drop(y)
        y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape):
        # kernel is not a SequenceModule so it doesn't need to adhere to same interface
        # the kernel will know the device of its own parameters
        return self.layer.default_state(*batch_shape)

    @property
    def d_output(self):
        return self.d_model
