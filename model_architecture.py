"""Helpers for configurable SB3 / sb3-contrib policy architectures."""

import torch.nn as nn


# Map UI activation names onto PyTorch module classes.
ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


def parse_layer_sizes(spec, default=None):
    """Parse a comma-separated layer-size string into a list of ints."""
    # Fall back to the provided default when the field is empty.
    if spec is None or str(spec).strip() == "":
        # Use the caller-supplied default architecture.
        return list(default) if default is not None else [512, 512]
    # Split on commas and strip whitespace from each token.
    parts = [part.strip() for part in str(spec).split(",")]
    # Drop empty tokens from trailing commas.
    parts = [part for part in parts if part]
    # Reject empty architectures after cleaning.
    if not parts:
        # Raise a clear error for the UI / CLI.
        raise ValueError(f"Invalid layer sizes: {spec!r}")
    # Convert every token into a positive integer neuron count.
    sizes = []
    # Walk each token in order.
    for part in parts:
        # Parse the neuron count for this layer.
        value = int(part)
        # Reject non-positive sizes.
        if value <= 0:
            # Surface a user-facing validation error.
            raise ValueError(f"Layer size must be > 0, got {value} in {spec!r}")
        # Keep the validated size.
        sizes.append(value)
    # Return the ordered list of hidden sizes.
    return sizes


def resolve_activation(name):
    """Resolve an activation name string to a torch.nn module class."""
    # Normalize to lowercase for dictionary lookup.
    key = str(name or "relu").strip().lower()
    # Reject unknown activation names early.
    if key not in ACTIVATION_MAP:
        # List supported names in the error message.
        raise ValueError(
            f"Unsupported activation {name!r}. "
            f"Choose one of: {', '.join(sorted(ACTIVATION_MAP))}"
        )
    # Return the matching module class.
    return ACTIVATION_MAP[key]


def build_policy_kwargs(config):
    """Build SB3 policy_kwargs from a TrainingConfig architecture section."""
    # Parse the actor MLP hidden sizes from the config string.
    pi_layers = parse_layer_sizes(config.pi_arch, default=[512, 512])
    # Parse the critic MLP hidden sizes from the config string.
    vf_layers = parse_layer_sizes(config.vf_arch, default=[512, 512])
    # Resolve the activation function class.
    activation_fn = resolve_activation(config.activation)
    # Shared kwargs used by PPO / RecurrentPPO / QRDQN feature extractors.
    kwargs = {
        # Pass CNN feature dimension into NatureCNN.
        "features_extractor_kwargs": dict(features_dim=int(config.features_dim)),
        # Activation used after each MLP layer.
        "activation_fn": activation_fn,
        # Orthogonal weight init helps PPO stability.
        "ortho_init": bool(config.ortho_init),
    }
    # RecurrentPPO needs LSTM + separate actor/critic heads.
    if config.algorithm == "RecurrentPPO":
        # Actor/critic MLP sizes after the LSTM.
        kwargs["net_arch"] = dict(pi=pi_layers, vf=vf_layers)
        # LSTM hidden state width.
        kwargs["lstm_hidden_size"] = int(config.lstm_hidden_size)
        # Number of stacked LSTM layers.
        kwargs["n_lstm_layers"] = int(config.n_lstm_layers)
        # Whether actor and critic share one LSTM.
        kwargs["shared_lstm"] = bool(config.shared_lstm)
        # Whether the critic also keeps an LSTM.
        kwargs["enable_critic_lstm"] = bool(config.enable_critic_lstm)
    # Classic PPO uses feed-forward actor/critic MLPs after the CNN.
    elif config.algorithm == "PPO":
        # Separate actor and critic heads.
        kwargs["net_arch"] = dict(pi=pi_layers, vf=vf_layers)
    # QR-DQN uses a single Q-network MLP after the CNN.
    elif config.algorithm == "QRDQN":
        # Parse the Q-network hidden sizes (reuse pi_arch by default).
        q_layers = parse_layer_sizes(
            config.q_arch if config.q_arch else config.pi_arch,
            default=[512, 512],
        )
        # QRDQN expects a flat list for net_arch.
        kwargs["net_arch"] = q_layers
    else:
        # Reject unknown algorithms before training starts.
        raise ValueError(f"Unsupported algorithm: {config.algorithm}")
    # Return the kwargs dict passed into the SB3 constructor.
    return kwargs


def default_policy_for_algorithm(algorithm):
    """Return the default SB3 policy class name for an algorithm."""
    # RecurrentPPO needs the CNN+LSTM policy.
    if algorithm == "RecurrentPPO":
        # CnnLstmPolicy matches image observations with recurrence.
        return "CnnLstmPolicy"
    # Feed-forward CNN for PPO and QRDQN.
    return "CnnPolicy"
