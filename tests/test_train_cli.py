"""The training CLI must accept every dataset that hcr.data can load."""

from hcr.data import dataset_names
from hcr.train import _parse_args


def test_every_registered_dataset_is_a_valid_cli_choice():
    names = dataset_names()
    assert "emnist-byclass" in names
    for name in names:
        assert _parse_args(["--dataset", name, "--arch", "cnn"]).dataset == name
