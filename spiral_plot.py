import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# -------------------
# data: 2D spiral
# -------------------
def make_spiral(
    n: int = 20000,
    n_turns: float = 2.0,
    r_start: float = 0.1,
    r_end: float = 1.0,
    noise_scale: float = 0.0,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a standardized 2D spiral dataset.

    Parameters:
        n: Number of data points to generate (must be positive, default: 20000)
        n_turns: Total number of spiral turns (1 turn = 2π radians, default: 2.0)
        r_start: Starting radius of the spiral (default: 0.1)
        r_end: Ending radius of the spiral (default: 1.0)
        noise_scale: Standard deviation of Gaussian noise added to coordinates (default: 0.0)

    Returns:
        Numpy array of shape `(n, 2)` where rows are `(x, y)` coordinates (standardized).

    Notes:
        - Angles increase linearly to produce `n_turns` loops around the origin.
        - Radii increase linearly from `r_start` to `r_end`, creating an outward spiral.
        - Optional Gaussian noise simulates measurement perturbations.
        - The dataset is standardized by its global standard deviation to keep unit scale.
    """
    assert isinstance(n, int) and n > 0, f"Number of data points 'n' must be a positive integer. Current input: {n}"
    assert r_start > 0 and r_end > 0, f"Radii 'r_start' and 'r_end' must be positive. Current inputs: {r_start}, {r_end}"
    assert n_turns > 0, f"Number of spiral turns 'n_turns' must be positive. Current input: {n_turns}"
    assert noise_scale >= 0, f"Noise standard deviation 'noise_scale' cannot be negative. Current input: {noise_scale}"

    t = np.linspace(0, 2 * np.pi * n_turns, n)  # Angles: 0 to (turns × 2π)
    r = np.linspace(r_start, r_end, n)          # Radii: linear increase from start to end
    x = r * np.cos(t)
    y = r * np.sin(t)

    # Add Gaussian noise to coordinates if requested
    if noise_scale > 0:
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=(n, 2))
        x += noise[:, 0]
        y += noise[:, 1]

    # Standardize to unit scale (global std), preserving overall variance
    data = np.stack([x, y], axis=1, dtype=np.float32)
    data_std = data.std()
    if data_std > 0:  # Avoid division by zero in edge cases (e.g., n=1)
        data /= data_std

    return data


class DenoiseMLP(nn.Module):
    """
    5-Layers MLP denoiser for DDPM.

    Parameters:
        dim: Feature dimension of input samples (D).
        config: Configuration dictionary containing the noise schedule.
            Required key: `"T"` (total number of timesteps).
        hidden: Number of hidden units in each layer (default: 256).
    """

    def __init__(self, dim, config, hidden=256):
        super().__init__()
        n = config['hidden_layers']
        assert n >= 2, f"Hidden layers 'n' must be at least 2. Current input: {n}"
        self.net = nn.ModuleList([
            nn.Linear(dim + 1, hidden),
            nn.ReLU(),
        ])
        for _ in range(n - 2):
            self.net.extend([
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            ])
        self.net.append(nn.Linear(hidden, dim))

        self.T = config['T']

    def forward(self, x, t):
        """
        Predicts the target representation at timestep `t`.

        - Normalizes `t` to [0, 1] for stable conditioning
        - Concatenates normalized time with features and feeds the MLP

        Parameters:
            x: Input features of shape `(B, D)`
            t: Integer timesteps of shape `(B,)` with values in `[0, T-1]`

        Returns:
            Tensor of shape `(B, D)` matching the chosen parameterization target
        """
        t = t.float().unsqueeze(1) / (self.T - 1)
        h = torch.cat([x, t], dim=1)
        for layer in self.net:
            h = layer(h)
        return h

# -------------------
# training for one D, one parameterization
# -------------------
def train_one(
        D, data2d, config, param="x",
        num_steps=4000, batch_size=512, lr=1e-3,
        val_split=0.2,
        patience=5,
        min_delta=1e-5,
):
    device = config['device']
    data_tensor = torch.from_numpy(data2d).to(device)
    N = data_tensor.shape[0]
    val_size = int(N * val_split)

    perm = torch.randperm(N, device=device)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    rand = torch.randn(D, 2)
    P, _ = torch.linalg.qr(rand)
    P = P.to(device)
    train_x0_D = data_tensor[train_idx] @ P.T
    val_x0_D = data_tensor[val_idx] @ P.T

    model = DenoiseMLP(D, config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3, factor=0.8)

    best_val_loss = float('inf')
    patience_counter = 0

    train_idx_all = torch.arange(train_x0_D.shape[0], device=device)
    with tqdm(total=num_steps, desc=f"Train D={D}, param={param}", leave=False) as pbar:
        for step in range(1, num_steps + 1):
            idx = train_idx_all[torch.randint(0, train_x0_D.shape[0], (batch_size,), device=device)]
            x0 = train_x0_D[idx]
            t = torch.randint(0, config['T'], (batch_size,), device=device)

            noise = torch.randn_like(x0)
            a_bar = config['alphas_bar'][t].view(-1, 1)
            sqrt_ab = torch.sqrt(a_bar).clamp(min=1e-6)
            sqrt_1mab = torch.sqrt(1 - a_bar).clamp(min=1e-6)
            xt = sqrt_ab * x0 + sqrt_1mab * noise
            v_true = sqrt_1mab * noise - sqrt_ab * x0

            pred = model(xt, t)
            if param == "ε":
                pred = pred + torch.randn_like(pred) * 0.005
                x0_hat = (xt - sqrt_1mab * pred) / sqrt_ab
                v_pred = sqrt_1mab * pred - sqrt_ab * x0_hat
            elif param == "x":
                eps_hat = (xt - sqrt_ab * pred) / sqrt_1mab
                v_pred = sqrt_1mab * eps_hat - sqrt_ab * pred
            elif param == "v":
                v_pred = pred
            else:
                raise ValueError(f"Unsupported param: {param}")

            loss = F.mse_loss(v_pred, v_true)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 500 == 0 or step == num_steps:
                val_loss = compute_val_loss(model, val_x0_D, config, param, batch_size=batch_size)
                sch.step(val_loss)
                pbar.set_postfix({"train_loss": f"{loss.item():.4f}", "val_loss": f"{val_loss:.4f}"})

                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        pbar.write(f"Early stopping triggered at step {step}! Best val loss: {best_val_loss:.4f}")
                        break
            else:
                pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})

            pbar.update(1)

    return model, P


def compute_val_loss(model, val_x0_D, config, param, batch_size=512):
    """
    Computes the validation loss for a given model and parameterization.

    Parameters:
        model: Trained `DenoiseMLP`
        val_x0_D: Validation set high-dimensional data (N_val, D)
        config: Diffusion configuration with `alphas_bar`
        param: One of `{"x", "eps", "v"}`
        batch_size: Batch size for validation

    Returns:
        val_loss: Average MSE loss over validation set
    """
    model.eval()
    device = config['device']
    N_val = val_x0_D.shape[0]
    total_loss = 0.0
    with torch.no_grad():
        for i in range(0, N_val, batch_size):
            batch_x0 = val_x0_D[i:i + batch_size]
            batch_t = torch.randint(0, config['T'], (batch_x0.shape[0],), device=device)

            noise = torch.randn_like(batch_x0)
            a_bar = config['alphas_bar'][batch_t].view(-1, 1)
            sqrt_ab = torch.sqrt(a_bar).clamp(min=1e-6)
            sqrt_1mab = torch.sqrt(1 - a_bar).clamp(min=1e-6)
            xt = sqrt_ab * batch_x0 + sqrt_1mab * noise
            v_true = sqrt_1mab * noise - sqrt_ab * batch_x0

            pred = model(xt, batch_t)
            if param == "ε":
                x0_hat = (xt - sqrt_1mab * pred) / sqrt_ab
                v_pred = sqrt_1mab * pred - sqrt_ab * x0_hat
            elif param == "x":
                eps_hat = (xt - sqrt_ab * pred) / sqrt_1mab
                v_pred = sqrt_1mab * eps_hat - sqrt_ab * pred
            elif param == "v":
                v_pred = pred
            else:
                raise ValueError(f"Unsupported param: {param}")

            batch_loss = F.mse_loss(v_pred, v_true)
            total_loss += batch_loss.item() * batch_x0.shape[0]

    model.train()
    return total_loss / N_val

# -------------------
# convert model output -> (x0_hat, eps_hat)
# -------------------
def model_to_x0_eps(model, x_t, t, param, alphas_bar):
    """
    Converts model output into a consistent pair `(x0_hat, eps_hat)`.

    For a given parameterization of the model output, compute both the
    reconstructed clean sample `x0_hat` and the noise `eps_hat` so that
    downstream sampling logic can be unified.

    Parameters:
        model: Denoiser mapping `(x_t, t)` → target parameterization
        x_t: Noisy samples of shape `(B, D)`
        t: Integer timesteps of shape `(B,)`
        param: One of `{"x", "eps", "v"}`
        alphas_bar: Cumulative product schedule `ᾱ` of shape `(T,)`

    Returns:
        Tuple `(x0_hat, eps_hat)` each of shape `(B, D)`
    """
    a_bar = alphas_bar[t].view(-1, 1)
    sqrt_ab = torch.sqrt(a_bar).clamp(min=1e-6)
    sqrt_1mab = torch.sqrt(1 - a_bar).clamp(min=1e-6)

    out = model(x_t, t)

    if param == "ε":
        eps_hat = out
        x0_hat = (x_t - sqrt_1mab * eps_hat) / sqrt_ab
    elif param == "x":
        x0_hat = out
        eps_hat = (x_t - sqrt_ab * x0_hat) / sqrt_1mab
    elif param == "v":
        v_hat = out
        x0_hat = (x_t - v_hat) / (2 * sqrt_ab)
        eps_hat = (x_t + v_hat) / (2 * sqrt_1mab)
    else:
        raise ValueError(f"Unsupported param: {param}, must be 'ε'/'x'/'v'")

    return x0_hat, eps_hat

# -------------------
# DDIM-style sampling
# -------------------
@torch.no_grad()
def sample(model, D, P, config, param="x", n_samples=4000):
    """
    Generates samples via a deterministic DDIM-style update and projects to 2D.

    Starting from standard normal noise in `R^D`, iteratively reconstruct
    `x_t` by predicting `(x0_hat, eps_hat)` and applying the closed-form
    update using the previous cumulative alpha `ᾱ_{t-1}`. Finally, map
    the result back to the original 2D space using the projection matrix.

    Parameters:
        model: Trained `DenoiseMLP`
        D: Feature dimension used during training
        P: Projection matrix of shape `D×2` (columns orthonormal)
        config: Diffusion schedule and device configuration
        param: Parameterization used by the model, one of `{"x","eps","v"}`
        n_samples: Number of samples to generate

    Returns:
        `np.ndarray` of shape `(n_samples, 2)` containing 2D coordinates
    """
    x_t = torch.randn(n_samples, D, device=config['device'])

    for t_step in reversed(range(config['T'])):
        t = torch.full((n_samples,), t_step, device=config['device'], dtype=torch.long)
        x0_hat, eps_hat = model_to_x0_eps(model, x_t, t, param, config['alphas_bar'])

        if t_step == 0:
            x_t = x0_hat
        else:
            a_bar_prev = config['alphas_bar'][t_step - 1]
            a_bar_prev = a_bar_prev.view(-1, 1)
            x_t = torch.sqrt(a_bar_prev) * x0_hat + torch.sqrt(1 - a_bar_prev) * eps_hat

    # Project back to 2D: x0 = y @ P^T  =>  y = x0 @ P
    x0_2d = x_t @ P           # (n_samples, 2)
    return x0_2d.cpu().numpy()

def scatter_panel(ax, points, title):
    """
    Helper to visualize 2D points as a scatter plot without axes ticks.
    """
    color = "orange" if "ground-truth" in title else "blue"
    ax.scatter(points[:, 0], points[:, 1], s=1, color=color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    base2d = make_spiral()

    # Linear beta schedule for a simple diffusion process
    config = {
        'T':200,
        'beta_start':1e-4,
        'beta_end':0.02,
        'hidden_layers':5,
    }
    config['betas'] = torch.linspace(config['beta_start'], config['beta_end'], config['T'], device=device)
    config['alphas'] = 1.0 - config['betas']
    config['alphas_bar'] = torch.cumprod(config['alphas'], dim=0)
    config['device'] = device

    Ds = [2, 8, 16, 512]
    params = ["x", "ε", "v"]
    results = {}  # (D, param) -> (samples_2d, P)

    for D in Ds:
        for param in params:
            model, P = train_one(
                D, base2d, config, param=param,
                num_steps=4000,
                lr=1e-3,
                val_split=0.2,
                patience=5,
                min_delta=1e-5,
            )
            samples_2d = sample(model, D, P, config, param=param, n_samples=4000)
            results[(D, param)] = (samples_2d, P)

    plt.figure(figsize=(10, 10))
    for row_idx, D in enumerate(Ds):
        # Ground truth (same for all rows)
        ax = plt.subplot(len(Ds), 4, row_idx * 4 + 1)
        scatter_panel(ax, base2d, f"ground-truth\nD={D}")
        for col_idx, param in enumerate(params):
            samples_2d, _ = results[(D, param)]
            title = {"x":"x-pred", "ε":"ε-pred", "v":"v-pred"}[param]
            ax = plt.subplot(len(Ds), 4, row_idx * 4 + 2 + col_idx)
            scatter_panel(ax, samples_2d, title)
    plt.tight_layout()
    plt.savefig("diffusion_samples.png")
    plt.close()

if __name__ == "__main__":
    main()
