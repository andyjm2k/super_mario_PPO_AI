"""GPU runtime tweaks and hyper-parameter presets for large NVIDIA cards."""

import torch


def configure_torch_gpu(device="auto"):
    """Enable cuDNN autotune / TF32 so fixed-size CNN batches run faster."""
    # Resolve auto to the concrete device string used by SB3.
    resolved = device
    # Prefer CUDA when available under auto mode.
    if resolved == "auto":
        # Pick cuda when a device is visible, else cpu.
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    # Skip GPU knobs entirely on CPU-only machines.
    if not str(resolved).startswith("cuda") or not torch.cuda.is_available():
        # Announce that training will stay on CPU.
        print("Torch GPU tweaks skipped (device is CPU).")
        # Return the resolved device for callers.
        return resolved
    # Allow cuDNN to pick the fastest conv algorithms for our fixed obs size.
    torch.backends.cudnn.benchmark = True
    # Prefer TF32 tensor cores on Ampere+ (RTX 30/40/50) for matmuls.
    torch.backends.cuda.matmul.allow_tf32 = True
    # Also allow TF32 inside cuDNN convolutions.
    torch.backends.cudnn.allow_tf32 = True
    # Use high-precision float32 matmul mode when the API exists.
    if hasattr(torch, "set_float32_matmul_precision"):
        # high ≈ TF32 on supported GPUs without changing dtypes.
        torch.set_float32_matmul_precision("high")
    # Log which GPU will run the policy updates.
    name = torch.cuda.get_device_name(0)
    # Print a short confirmation for the training booth / console.
    print(f"Torch GPU tweaks enabled on {name} (cudnn.benchmark + TF32).")
    # Return the concrete device string.
    return resolved


def suggest_gpu_hyperparams(algorithm="RecurrentPPO"):
    """Return larger n_envs / n_steps / batch sizes aimed at high-end GPUs."""
    # Base preset shared by feed-forward and recurrent PPO.
    preset = {
        # More workers → larger batched rollout inference on GPU.
        "n_envs": 24,
        # Longer rollouts → bigger update buffers between env waits.
        "n_steps": 1024,
        # Large minibatches keep an RTX 5090 busier during train().
        "batch_size": 2048,
        # More epochs spend relatively more wall time on the GPU.
        "n_epochs": 8,
        # Force CUDA when the user clicks the GPU preset.
        "device": "cuda",
        # Live frames steal CPU from PyBoy workers — off by default in preset.
        "show_visuals": False,
        # Still show a couple of panes if the user re-enables visuals.
        "visual_agents": 2,
    }
    # RecurrentPPO sequence packing prefers batch_size dividing the buffer.
    if algorithm == "RecurrentPPO":
        # 24 * 1024 = 24576, divisible by 2048.
        preset["batch_size"] = 2048
    # Plain PPO can push an even larger minibatch safely.
    elif algorithm == "PPO":
        # Slightly larger batch for feed-forward CNN updates.
        preset["batch_size"] = 4096
        # 24 * 1024 = 24576; 24576 % 4096 != 0, so nudge n_envs.
        preset["n_envs"] = 32
        # 32 * 1024 = 32768, divisible by 4096.
        preset["batch_size"] = 4096
    # QR-DQN is off-policy; larger batch helps the GPU replay updates.
    elif algorithm == "QRDQN":
        # Replay minibatches can be large without n_steps coupling.
        preset["batch_size"] = 1024
        # Still scale env count for faster data collection.
        preset["n_envs"] = 16
        # n_steps unused by DQN; keep a harmless default.
        preset["n_steps"] = 128
        # Epochs unused by DQN; keep a harmless default.
        preset["n_epochs"] = 1
    # Return the algorithm-specific preset dict.
    return preset


def validate_batch_divides_buffer(n_envs, n_steps, batch_size, algorithm):
    """Return an error string if PPO buffer size is incompatible with batch_size."""
    # DQN does not use the on-policy buffer constraint.
    if algorithm == "QRDQN":
        # No validation needed.
        return None
    # Total transitions collected per rollout.
    buffer_size = int(n_envs) * int(n_steps)
    # batch_size must divide the rollout buffer for SB3 PPO.
    if buffer_size % int(batch_size) != 0:
        # Explain how to fix the UI values.
        return (
            f"n_envs*n_steps ({buffer_size}) must be divisible by "
            f"batch_size ({batch_size}). Try the GPU PRESET or adjust values."
        )
    # Compatible configuration.
    return None
