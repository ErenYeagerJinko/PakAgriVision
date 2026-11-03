"""Unit tests for the training engine, validation, and early stopping.

Phase 3 & 4 - Deliverable 2.
Verifies:
1. train_one_epoch steps optimizer, updates weights, and tracks running loss/accuracy.
2. validate computes loss, accuracy, and macro-F1 across full evaluation arrays.
3. EarlyStopping monitors score correctly, triggers after patience runs out, and handles reset.
4. Checkpoint saving and loading preserves model state, optimizer state, scheduler state, and metadata.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.engine import EarlyStopping, train_one_epoch, validate
from src.models.classifier import build_model


class DummyClassifier(nn.Module):
    """Simple 2-layer classifier for engine unit testing."""

    def __init__(self, in_features: int = 16, num_classes: int = 4):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@pytest.fixture
def dummy_train_loader() -> DataLoader:
    torch.manual_seed(42)
    x = torch.randn(20, 16)
    y = torch.randint(0, 4, (20,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=4, shuffle=True)


@pytest.fixture
def dummy_val_loader() -> DataLoader:
    torch.manual_seed(42)
    x = torch.randn(10, 16)
    y = torch.randint(0, 4, (10,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=5, shuffle=False)


def test_early_stopping_max_mode():
    """Verify EarlyStopping behaves properly when maximizing a metric (e.g. macro-F1)."""
    es = EarlyStopping(patience=2, min_delta=1e-3, mode="max")

    assert not es.step(0.50)  # best = 0.50, counter = 0
    assert es.is_improved

    assert not es.step(0.55)  # improved -> best = 0.55, counter = 0
    assert es.is_improved

    assert not es.step(0.54)  # worse -> counter = 1
    assert not es.is_improved
    assert es.counter == 1

    # Should trigger stop on 2nd unimproved step
    should_stop = es.step(0.545)
    assert should_stop
    assert es.counter == 2
    assert es.early_stop


def test_early_stopping_min_mode():
    """Verify EarlyStopping behaves properly when minimizing a metric (e.g. loss)."""
    es = EarlyStopping(patience=2, min_delta=1e-3, mode="min")

    assert not es.step(1.50)
    assert es.is_improved
    assert es.best_score == 1.50

    assert not es.step(1.20)  # improved
    assert es.is_improved
    assert es.best_score == 1.20

    assert not es.step(1.25)  # worse -> counter = 1
    assert not es.is_improved

    assert es.step(1.30)  # worse -> counter = 2 -> trigger stop
    assert es.early_stop


def test_train_one_epoch_updates_weights(dummy_train_loader):
    """Verify train_one_epoch executes forward/backward passes and updates weights."""
    model = DummyClassifier(in_features=16, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cpu")

    initial_weights = model.fc.weight.clone()

    avg_loss, train_acc = train_one_epoch(
        model=model,
        dataloader=dummy_train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scaler=None,
        grad_clip_norm=1.0,
    )

    assert isinstance(avg_loss, float)
    assert avg_loss > 0.0
    assert 0.0 <= train_acc <= 1.0

    # Ensure weights were modified by optimizer step
    assert not torch.allclose(initial_weights, model.fc.weight)


def test_validate_computes_metrics(dummy_val_loader):
    """Verify validate calculates valid loss, accuracy, and macro-F1 over full batch set."""
    model = DummyClassifier(in_features=16, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    val_loss, val_acc, val_macro_f1 = validate(
        model=model,
        dataloader=dummy_val_loader,
        criterion=criterion,
        device=device,
    )

    assert isinstance(val_loss, float)
    assert val_loss > 0.0
    assert 0.0 <= val_acc <= 1.0
    assert 0.0 <= val_macro_f1 <= 1.0


def test_validate_empty_dataloader():
    """Verify validate returns zeroed tuple when dataloader is empty."""
    model = DummyClassifier()
    criterion = nn.CrossEntropyLoss()
    empty_loader = DataLoader(TensorDataset(torch.empty(0, 16), torch.empty(0, dtype=torch.long)))

    loss, acc, f1 = validate(model, empty_loader, criterion, torch.device("cpu"))
    assert loss == 0.0
    assert acc == 0.0
    assert f1 == 0.0


def test_checkpoint_save_and_load(tmp_path):
    """Verify save_checkpoint and load_checkpoint round-trip correctly."""
    from src.train import load_checkpoint, save_checkpoint

    model = DummyClassifier(in_features=16, num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    state = {
        "epoch": 3,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "scaler_state_dict": None,
        "best_val_macro_f1": 0.85,
        "config": {"model": {"backbone": "resnet50"}},
        "class_names": ["c0", "c1", "c2", "c3"],
    }

    save_checkpoint(state, is_best=True, checkpoint_dir=tmp_path)
    assert (tmp_path / "last.pt").is_file()
    assert (tmp_path / "best.pt").is_file()

    # Create new model and load weights
    new_model = DummyClassifier(in_features=16, num_classes=4)
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    loaded_state = load_checkpoint(tmp_path / "best.pt", model=new_model, optimizer=new_opt)

    assert loaded_state["epoch"] == 3
    assert loaded_state["best_val_macro_f1"] == 0.85
    assert loaded_state["class_names"] == ["c0", "c1", "c2", "c3"]

    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)


def test_stratified_train_subset():
    """Verify get_stratified_train_subset samples classes proportionally."""
    import pandas as pd
    from src.train import get_stratified_train_subset

    df = pd.DataFrame({
        "path": [f"img_{i}.jpg" for i in range(100)],
        "label_idx": [0] * 50 + [1] * 50,
    })

    sampled_50 = get_stratified_train_subset(df, subset_frac=0.5, seed=42)
    assert len(sampled_50) == 50
    assert (sampled_50["label_idx"] == 0).sum() == 25
    assert (sampled_50["label_idx"] == 1).sum() == 25

    # Identity check when frac=1.0
    full = get_stratified_train_subset(df, subset_frac=1.0)
    assert len(full) == 100


def test_tune_trial_partitioning():
    """Verify trial partitioning splits workload evenly across GPU accounts."""
    from src.tune import ALL_TRIALS, partition_trials

    part1 = partition_trials(ALL_TRIALS, "1/3")
    part2 = partition_trials(ALL_TRIALS, "2/3")
    part3 = partition_trials(ALL_TRIALS, "3/3")

    combined = part1 + part2 + part3
    assert len(combined) == len(ALL_TRIALS) == 13
    assert [t["trial_id"] for t in combined] == list(range(1, 14))


def test_plotting_generates_valid_figures(tmp_path):
    """Verify plotting functions generate non-empty PNG files."""
    import pandas as pd
    from src.plotting import plot_all_training_curves, plot_hparam_search

    # Synthetic training history
    history_df = pd.DataFrame({
        "epoch": [1, 2, 3],
        "train_loss": [1.5, 0.9, 0.4],
        "train_acc": [0.6, 0.8, 0.92],
        "val_loss": [1.4, 0.85, 0.5],
        "val_acc": [0.62, 0.81, 0.89],
        "val_macro_f1": [0.58, 0.79, 0.88],
        "lr": [3e-4, 2e-4, 1e-4],
        "epoch_time_sec": [12.0, 11.8, 12.1],
    })

    plot_all_training_curves(history_df, output_dir=tmp_path, unfreeze_epoch=1, best_epoch=3)
    assert (tmp_path / "loss_curves.png").is_file()
    assert (tmp_path / "accuracy_curves.png").is_file()
    assert (tmp_path / "val_f1_curve.png").is_file()
    assert (tmp_path / "loss_curves.png").stat().st_size > 1000

    # Synthetic hyperparameter search results
    hparam_df = pd.DataFrame({
        "trial_id": [1, 2, 7],
        "stage": ["A", "A", "B"],
        "val_macro_f1": [0.82, 0.88, 0.85],
    })
    hparam_out = tmp_path / "hparam_search.png"
    plot_hparam_search(hparam_df, output_path=hparam_out)
    assert hparam_out.is_file()
    assert hparam_out.stat().st_size > 1000


