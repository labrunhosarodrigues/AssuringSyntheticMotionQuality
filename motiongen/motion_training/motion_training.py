# -*- coding: utf-8 -*-
"""
motiongen.motion_training
-------------------------

Training loop for Activity recognition model on top of large dataset.

:copyright: (c) 2025 by Cognitive Systems Lab.
:license: MIT
"""
# Imports
# built-in
import os

# local
from motion_training.motion_classifier import SemanticEncoder
from motion_training.dataset import (
    MotionDataSet, infect_data, load_large_dataset
)

# 3rd-party
import torch
import torch.nn as nn
import torch.optim as optim
import lightning
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from torcheval.metrics.functional import multiclass_f1_score

# CSL


class LitSemanticEncoder(lightning.LightningModule):
    def __init__(self, encoder, loss):
        super().__init__()
        self.encoder = encoder
        self.loss = loss

    def training_step(self, batch, batch_idx):
        """Training Loop"""
        x, y = batch
        y_pred = self.encoder(x)
        loss = self.loss(y_pred, y)
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation Loop"""
        x, y = batch
        y_pred = self.encoder(x)
        loss = self.loss(y_pred, y)
        self.log("validation_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        """Test Loop"""
        x, y = batch
        y_pred = self.encoder(x)
        loss = self.loss(y_pred, y)

        labels_pred = torch.argmax(y_pred, dim=1)
        labels_true = torch.argmax(y, dim=1)
        f1_score = multiclass_f1_score(
            labels_pred, labels_true,
            num_classes=10, average=None
        )
        self.log_dict({"test_loss": loss})
        for label in range(10):
            self.log_dict({f"test_f1_class_{label}": f1_score[label]})
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=.5e-4)
        return optimizer


def post_train_evaluation(
    test_clean, test_dirty,
    architecture, loss,
    checkpoints_path
):
    _, folder, _ = next(os.walk(checkpoints_path))

    for run in folder:
        model = LitSemanticEncoder.load_from_checkpoint(
            os.path.join(
                checkpoints_path,
                run,
                "checkpoints",
                "epoch=999-step=101000.ckpt"
            ),
            encoder=architecture(),
            loss=loss
        )
        trainer = lightning.Trainer(max_epochs=500)
        trainer.logger.log_hyperparams({"version": run})
        model.eval()

        clean_test_loader = DataLoader(
            MotionDataSet("data/large_dataset", test_clean),
            batch_size=len(test_clean),
            shuffle=False,
            num_workers=24
        )
        trainer.test(model, dataloaders=clean_test_loader)

        dirty_test_loader = DataLoader(
            MotionDataSet("data/large_dataset", test_dirty),
            batch_size=len(test_clean),
            shuffle=False,
            num_workers=24
        )
        trainer.test(model, dataloaders=dirty_test_loader)


def evaluation(
    train_clean, test_clean,
    train_dirty, test_dirty,
    architecture, loss,
    is_infected=False,
    batch_size=64,
    n_clean_data=-1
):
    # Set Cross Validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    folds = kfold.split(train_clean["y"], train_clean["y"])

    for i, (train_idx, test_idx) in enumerate(folds):
        model = LitSemanticEncoder(architecture(), loss)

        # train/validate
        train_data = train_clean.iloc[train_idx]
        if n_clean_data > 0:
            train_data, _ = train_test_split(
                train_data, test_size=1 - (n_clean_data / len(train_data)),
                random_state=42
            )
        if is_infected:
            train_data = infect_data(train_data, train_dirty)
        train_loader = DataLoader(
            MotionDataSet("data/large_dataset", train_data),
            batch_size=batch_size,
            shuffle=True,
            num_workers=24
        )
        validation_loader = DataLoader(
            MotionDataSet("data/large_dataset", train_clean.iloc[test_idx]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=24
        )

        trainer = lightning.Trainer(max_epochs=500)
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=validation_loader
        )

        # Evaluate
        model.eval()
        clean_test_loader = DataLoader(
            MotionDataSet("data/large_dataset", test_clean),
            batch_size=len(test_clean),
            shuffle=False,
            num_workers=24
        )
        trainer.test(model, dataloaders=clean_test_loader)

        dirty_test_loader = DataLoader(
            MotionDataSet("data/large_dataset", test_dirty),
            batch_size=len(test_clean),
            shuffle=False,
            num_workers=24
        )
        trainer.test(model, dataloaders=dirty_test_loader)


def prepare_training(
        num_joints, num_feats,
        num_frames,
        infection_rate
):
    def architecture():
        return SemanticEncoder(
            input_feats=num_joints * num_feats,
            num_frames=num_frames,
            latent_dim=256,
            transformer_feedforward_dim=512,
            num_layers=8,
            num_heads=4,
            dropout=0.2,
            semantic_pool_type='global_avg_pool',
            out_dim=10
        )

    loss = nn.CrossEntropyLoss()

    # Load Clean Data and Dirty Data
    clean_data, dirty_data = load_large_dataset(
        "data/large_dataset/filtering.csv")

    # Separate Test and train
    train_clean, test_clean = train_test_split(
        clean_data, test_size=.3,
        random_state=42
    )
    train_dirty, test_dirty = train_test_split(
        dirty_data, test_size=.3,
        random_state=42
    )

    if infection_rate:
        n_clean_data = len(train_dirty) / infection_rate
    else:
        n_clean_data = -1

    return (
        architecture, loss,
        train_clean, test_clean,
        train_dirty, test_dirty,
        n_clean_data
    )


def cli():
    """Command Line Interface"""
    main()


def main():
    BATCH_SIZE = 64
    num_joints = 22
    num_feats = 3
    # Maximal number of frames. For shorter recordings you may want
    # to repeat the last frame until this number of frames is reached.
    # Longer ones must be cut.
    num_frames = 120
    infection_rate = 0.25

    TRAIN = False

    (
        architecture, loss,
        train_clean, test_clean,
        train_dirty, test_dirty,
        n_clean_data
    ) = prepare_training(
        num_joints, num_feats,
        num_frames,
        infection_rate
    )

    if TRAIN:
        evaluation(
            train_clean, test_clean,
            train_dirty, test_dirty,
            architecture, loss,
            is_infected=False,
            batch_size=BATCH_SIZE,
            n_clean_data=n_clean_data
        )

        evaluation(
            train_clean, test_clean,
            train_dirty, test_dirty,
            architecture, loss,
            is_infected=True,
            batch_size=BATCH_SIZE,
            n_clean_data=n_clean_data
        )
    else:
        post_train_evaluation(
            test_clean, test_dirty,
            architecture, loss,
            "64_5E-5_1000_logs"
        )


if __name__ == '__main__':
    cli()
