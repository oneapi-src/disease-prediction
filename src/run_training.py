# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Run training benchmarks
"""

import argparse
import logging
import pathlib
import time

from sklearn.metrics import accuracy_score
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from utils.process_data import read_and_preprocess_data
from utils.train import train

# training parameters

torch.manual_seed(0)


def main(flags):
    """Benchmark model training

    Args:
        flags: benchmarking configuration
    """

    if flags.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        path = pathlib.Path(flags.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=flags.logfile, level=logging.DEBUG)
    logger = logging.getLogger()

    logger.info("Trying to use %s as a pretrained BERT", flags.bert_model)
    tokenizer = AutoTokenizer.from_pretrained(flags.bert_model)

    # Read in the datasets and crate dataloaders
    logger.debug(
        "Reading in the data from %s",
        flags.data_dir
    )
    path = pathlib.Path(flags.data_dir)

    try:
        train_dataset = read_and_preprocess_data(
            path / "Training.csv",
            tokenizer,
            max_length=flags.seq_length
        )
        test_dataset = read_and_preprocess_data(
            path / "Testing.csv",
            tokenizer,
            max_length=flags.seq_length
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=flags.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=flags.batch_size, shuffle=True
        )
    except FileNotFoundError as exc:
        logger.error("Please follow instructions to download data.")
        logger.error(exc, exc_info=True)
        return

    start = time.time()

    # Load the C BERT sequence classifier
    model = AutoModelForSequenceClassification.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT",
        num_labels=41
    )

    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)

    # if using intel, optimize the model and the optimizer
    if flags.intel:
        import intel_extension_for_pytorch as ipex
        if flags.bf16:
            dtype = torch.bfloat16
        else:
            dtype = None # default dtype for ipex.optimize()
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=dtype)

    logger.debug("Training the model...")
    train(
        train_loader,
        val_loader,
        model,
        optimizer,
        flags.bf16,
        epochs=flags.epochs,
        max_grad_norm=flags.grad_norm,
    )

    training_time = time.time()
    model.eval()

    test_preds = []
    test_labels = []
    for _, (batch, labels) in enumerate(val_loader):
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

    # Save model
    if flags.save_model_dir:
        path = pathlib.Path(flags.save_model_dir)
        path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(path)
        model.save_pretrained(path)
        logger.info("Saved model files to %s", path)

    logger.info("=======> Test Accuracy : %.2f",
                accuracy_score(test_preds, test_labels)
                )
    logger.info("=======> Training Time : %.3f secs", training_time - start)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help="Directory with Training.csv and Testing.csv"
    )

    parser.add_argument(
        '--logfile',
        type=str,
        default="",
        help="Log file to output benchmarking results to.")

    parser.add_argument(
        '--intel',
        default=False,
        action="store_true",
        help="Use intel accelerated technologies where available."
    )

    parser.add_argument(
        '--save_model_dir',
        default=None,
        type=str,
        required=False,
        help="Directory to save model under."
    )

    parser.add_argument(
        '--seq_length',
        default=64,
        type=int,
        help="Sequence length to use when training."
    )

    parser.add_argument(
        '--batch_size',
        default=30,
        type=int,
        help="Batch size to use when training."
    )

    parser.add_argument(
        '--epochs',
        default=3,
        type=int,
        help="Number of training epochs."
    )

    parser.add_argument(
        '--grad_norm',
        default=10.0,
        type=float,
        help="Gradient clipping cutoff."
    )

    parser.add_argument(
        '--bert_model',
        default="emilyalsentzer/Bio_ClinicalBERT",
        type=str,
        help="Bert base model to fine tune."
    )
    
    parser.add_argument(
        '--bf16',
        default=False,
        action="store_true",
        help="Enable bf16 training"
    )

    FLAGS = parser.parse_args()

    if not ((pathlib.Path(FLAGS.data_dir) / 'Training.csv').exists() and
            (pathlib.Path(FLAGS.data_dir) / 'Testing.csv').exists()):
        print("Please download data files.")
    else:
        main(FLAGS)
