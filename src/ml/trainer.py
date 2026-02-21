"""Full training loop for the product classifier."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Callable, Protocol

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from src.core.config import CONFIG
from src.core.device import DEVICE

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Callback protocol
# ------------------------------------------------------------------

class TrainerCallback(Protocol):
    """Minimal callback interface for the training loop."""

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None: ...
    def on_training_complete(self, best_metrics: dict[str, float]) -> None: ...


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------

class Trainer:
    """Supervised fine-tuning loop for :class:`src.ml.classifier.Classifier`.

    Parameters
    ----------
    classifier : src.ml.classifier.Classifier
        The classifier instance (backbone + head) to train.
    train_dataset : Dataset
        Training dataset (must return ``(tensor, label)`` pairs).
    val_dataset : Dataset
        Validation dataset.
    config : src.core.config.Config | None
        Application config. Defaults to ``CONFIG``.
    device : torch.device | None
        Compute device. Defaults to ``DEVICE``.
    """

    def __init__(
        self,
        classifier: Any,  # src.ml.classifier.Classifier (avoid circular import)
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: Any | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.classifier = classifier
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or CONFIG
        self.device = device or DEVICE

        self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        epochs: int | None = None,
        lr: float | None = None,
        callbacks: list[TrainerCallback] | None = None,
    ) -> dict[str, list[float]]:
        """Run the full training loop.

        Parameters
        ----------
        epochs : int | None
            Number of training epochs. Defaults to ``config.train_epochs``.
        lr : float | None
            Initial learning rate. Defaults to ``config.train_lr``.
        callbacks : list[TrainerCallback] | None
            Optional list of callback objects.

        Returns
        -------
        dict[str, list[float]]
            History dict with keys ``train_loss``, ``val_loss``,
            ``val_accuracy`` — each a list with one value per epoch.
        """
        epochs = epochs or self.config.train_epochs
        lr = lr or self.config.train_lr
        callbacks = callbacks or []

        # DataLoaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # Prepare backbone for fine-tuning
        self.classifier.backbone.unfreeze_last_n_blocks(
            self.config.fine_tune_blocks
        )

        # Collect trainable parameters from backbone + head
        params = [
            {"params": self.classifier.backbone.model.parameters(), "lr": lr * 0.1},
            {"params": self.classifier.head.parameters(), "lr": lr},
        ]
        optimizer = AdamW(params, weight_decay=self.config.train_weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        # History tracking
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        best_accuracy = 0.0
        best_state: dict[str, Any] | None = None

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer)
            val_loss, val_accuracy = self._validate_epoch(val_loader)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "lr": optimizer.param_groups[-1]["lr"],
            }

            logger.info(
                "Epoch %d/%d — train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
                epoch,
                epochs,
                train_loss,
                val_loss,
                val_accuracy,
            )

            # Checkpoint best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_state = {
                    "backbone_state": copy.deepcopy(
                        self.classifier.backbone.model.state_dict()
                    ),
                    "head_state": copy.deepcopy(
                        self.classifier.head.state_dict()
                    ),
                }
                checkpoint_path = (
                    self.config.classifier_dir / "classifier.pth"
                )
                self.classifier.save(checkpoint_path)
                logger.info("  -> Saved best checkpoint (acc=%.4f)", best_accuracy)

            for cb in callbacks:
                cb.on_epoch_end(epoch, metrics)

        # Restore best weights
        if best_state is not None:
            self.classifier.backbone.model.load_state_dict(
                best_state["backbone_state"]
            )
            self.classifier.head.load_state_dict(best_state["head_state"])

        best_metrics = {
            "best_val_accuracy": best_accuracy,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
        }
        for cb in callbacks:
            cb.on_training_complete(best_metrics)

        return history

    # ------------------------------------------------------------------
    # Epoch helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer
    ) -> float:
        """Run one training epoch. Returns average loss."""
        self.classifier.backbone.model.train()
        self.classifier.head.train()

        total_loss = 0.0
        num_batches = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward
            features = self.classifier.backbone.model(images)
            logits = self.classifier.head(features)
            loss = self.criterion(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _validate_epoch(self, loader: DataLoader) -> tuple[float, float]:
        """Run one validation epoch. Returns ``(avg_loss, accuracy)``."""
        self.classifier.backbone.model.eval()
        self.classifier.head.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                features = self.classifier.backbone.model(images)
                logits = self.classifier.head(features)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                num_batches += 1

                _, predicted = logits.max(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy
