# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Training code for the model
"""

import logging

from sklearn.metrics import accuracy_score
import torch

from tqdm import tqdm

from contextlib import nullcontext

logger = logging.getLogger()


def train(
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        enable_bf16: bool,
        epochs: int = 5,
        max_grad_norm: float = 10) -> None:
    """train a model on the given dataset

    Args:
        dataloader (torch.utils.data.DataLoader): training dataset
        model (torch.nn.Module): model to train
        optimizer (torch.optim.Optimizer): optimizer to use
        enable_bf16 (bool): Enable bf16 mixed precision training
        epochs (int, optional): number of training epochs. Defaults to 5.
        max_grad_norm (float, optional): gradient clipping. Defaults to 10.
    """

    model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0
        train_preds = []
        train_labels = []
        for _, (batch, labels) in tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                desc=f"Epoch {epoch}"):

            optimizer.zero_grad()
            # use mixed precision bf16 training only if enabled
            with torch.cpu.amp.autocast() if enable_bf16 else nullcontext():

                ids = batch['input_ids']
                mask = batch['attention_mask']
                token_type_ids = batch['token_type_ids']

                out = model(
                    input_ids=ids,
                    attention_mask=mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

                loss = out.loss
                train_preds.extend(out.logits.argmax(-1))
                train_labels.extend(labels)

                # clip gradients for stability
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(), max_norm=max_grad_norm
                )

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
        logger.info("Epoch Train Accuracy %.4f",
                    accuracy_score(train_preds, train_labels)
                    )

        test_preds = []
        test_labels = []
        for _, (batch, labels) in enumerate(val_dataloader):
            # use mixed precision bf16 training only if enabled
            with torch.cpu.amp.autocast() if enable_bf16 else nullcontext():
                ids = batch['input_ids']
                mask = batch['attention_mask']
                token_type_ids = batch['token_type_ids']

                pred = model(
                    input_ids=ids,
                    attention_mask=mask,
                    token_type_ids=token_type_ids
                )
                test_preds.extend(pred.logits.argmax(-1))
                test_labels.extend(labels)
        logger.info("Test Accuracy %.4f",
                    accuracy_score(test_preds, test_labels)
                    )

        logger.info("Epoch %d, Loss %.4f", epoch, running_loss)
