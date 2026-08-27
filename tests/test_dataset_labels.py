"""Tests des labels du dataset local de préférences."""

from bestshot.dataset.labels import (
    FrameLabel,
    from_storage_value,
    is_training_label,
    to_storage_value,
)


def test_skip_is_an_absent_label_and_never_a_training_example() -> None:
    assert to_storage_value(FrameLabel.SKIP) is None
    assert from_storage_value(None) is FrameLabel.SKIP
    assert is_training_label(FrameLabel.SKIP) is False
    assert is_training_label(FrameLabel.KEEP) is True
    assert is_training_label(FrameLabel.REJECT) is True
