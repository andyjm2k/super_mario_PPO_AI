"""Unit tests for configurable model architecture helpers."""

import unittest

import torch.nn as nn

from model_architecture import (
    build_policy_kwargs,
    default_policy_for_algorithm,
    parse_layer_sizes,
    resolve_activation,
)
from training_config import TrainingConfig


class TestModelArchitecture(unittest.TestCase):
    """Cover layer parsing and policy_kwargs construction."""

    def test_parse_layer_sizes(self):
        # Comma-separated sizes should become an int list.
        self.assertEqual(parse_layer_sizes("512,256"), [512, 256])
        # Empty strings fall back to the provided default.
        self.assertEqual(parse_layer_sizes("", default=[64]), [64])

    def test_parse_layer_sizes_rejects_invalid(self):
        # Zero-sized layers are invalid.
        with self.assertRaises(ValueError):
            # Trigger validation on a non-positive size.
            parse_layer_sizes("128,0")

    def test_resolve_activation(self):
        # relu should map to nn.ReLU.
        self.assertIs(resolve_activation("relu"), nn.ReLU)
        # Unknown names should raise.
        with self.assertRaises(ValueError):
            # Trigger an unsupported activation error.
            resolve_activation("swish")

    def test_build_policy_kwargs_recurrent(self):
        # Build kwargs for the default RecurrentPPO config.
        config = TrainingConfig(
            algorithm="RecurrentPPO",
            pi_arch="256,128",
            vf_arch="256",
            features_dim=256,
            lstm_hidden_size=128,
            n_lstm_layers=2,
        )
        # Construct the policy kwargs dict.
        kwargs = build_policy_kwargs(config)
        # Actor/critic heads must match the parsed sizes.
        self.assertEqual(kwargs["net_arch"]["pi"], [256, 128])
        # Critic head must match.
        self.assertEqual(kwargs["net_arch"]["vf"], [256])
        # LSTM size must propagate.
        self.assertEqual(kwargs["lstm_hidden_size"], 128)
        # LSTM layer count must propagate.
        self.assertEqual(kwargs["n_lstm_layers"], 2)
        # CNN feature dim must propagate.
        self.assertEqual(kwargs["features_extractor_kwargs"]["features_dim"], 256)

    def test_build_policy_kwargs_qrdqn(self):
        # Build kwargs for QRDQN using q_arch.
        config = TrainingConfig(algorithm="QRDQN", q_arch="64,32")
        # Construct the policy kwargs dict.
        kwargs = build_policy_kwargs(config)
        # QRDQN uses a flat net_arch list.
        self.assertEqual(kwargs["net_arch"], [64, 32])

    def test_default_policy_for_algorithm(self):
        # RecurrentPPO should default to CnnLstmPolicy.
        self.assertEqual(default_policy_for_algorithm("RecurrentPPO"), "CnnLstmPolicy")
        # PPO should default to CnnPolicy.
        self.assertEqual(default_policy_for_algorithm("PPO"), "CnnPolicy")


if __name__ == "__main__":
    # Run the architecture unit tests.
    unittest.main()
