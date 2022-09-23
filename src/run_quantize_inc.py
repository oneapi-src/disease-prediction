# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Quantize a model using intel extension for pytorch
"""

import argparse
import os
import logging
import shutil

from neural_compressor.experimental import Quantization, common
from sklearn.metrics import accuracy_score
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from utils.process_data import read_and_preprocess_data


class INCDataset:
    """Dataset wrapper for INC
    """

    def __init__(self, dloader, n_elements=None):
        self.dloader = dloader
        self.n_elements = n_elements

    def __getitem__(self, index):
        item = self.dloader[index]

        x_vals = {
            "input_ids": item['input_ids'],
            "attention_mask": item['attention_mask']
        }
        y_vals = (item['labels'], item['class_label'])

        return (x_vals), y_vals

    def __len__(self):
        if self.n_elements is None:
            return len(self.dloader)
        return self.n_elements


def quantize_model(model, test_loader, inc_config_file):
    """Quantizes the model using the given dataset and INC config

    Args:
        model : PyTorch model to quantize.
        test_loader : Dataset to use for quantization.
        inc_config_file : Path to INC config.
    """

    def evaluate_accuracy(model_q) -> float:
        test_preds = []
        test_labels = []
        for _, (batch, labels) in enumerate(test_loader):
            ids = batch['input_ids']
            mask = batch['attention_mask']
            token_type_ids = batch['token_type_ids']

            pred = model_q(
                input_ids=ids,
                attention_mask=mask,
                token_type_ids=token_type_ids
            )
            test_preds.extend(pred.logits.argmax(-1))
            test_labels.extend(labels)

        return accuracy_score(test_preds, test_labels)

    # quantize model using provided configuration
    quantizer = Quantization(inc_config_file)
    cmodel = common.Model(model)
    quantizer.model = cmodel
    quantizer.calib_dataloader = test_loader
    quantizer.eval_func = evaluate_accuracy
    quantized_model = quantizer()

    return quantized_model


def main(flags) -> None:
    """Calibrate model for int 8 and serialize as a .pt

    Args:
        flags: benchmarking flags
    """

    # Validate Flags
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()

    if not os.path.exists(flags.saved_model_dir):
        logger.error("Saved model %s not found!", flags.saved_model_dir)
        return

    if not os.path.exists(flags.inc_config_file):
        logger.error("INC configuration %s not found!", flags.inc_config_file)
        return

    tokenizer = AutoTokenizer.from_pretrained(flags.saved_model_dir)

    # Load dataset for quantization
    try:
        test_dataset = read_and_preprocess_data(
            flags.input_file,
            tokenizer,
            max_length=flags.seq_length
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=flags.batch_size, shuffle=True
        )
    except FileNotFoundError as exc:
        logger.error("Please follow instructions to download data.")
        logger.error(exc, exc_info=True)
        return

    model = AutoModelForSequenceClassification.from_pretrained(
        flags.saved_model_dir
    )

    quantized_model = quantize_model(model, test_loader, flags.inc_config_file)
    quantized_model.save(flags.output_dir)

    # Rename files to better match the saved transformer model format
    if os.path.exists(os.path.join(flags.output_dir, "best_model.pt")):

        shutil.copytree(
            flags.saved_model_dir,
            flags.output_dir,
            ignore=shutil.ignore_patterns('*.bin*'),
            dirs_exist_ok=True
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_file',
        required=True,
        help="input to make predictions on",
        type=str
    )

    parser.add_argument(
        '--batch_size',
        default=10,
        type=int,
        help="batch size to use. if -1, uses all entries in input."
    )

    parser.add_argument(
        '--saved_model_dir',
        required=True,
        help="saved pretrained model to benchmark",
        type=str
    )

    parser.add_argument(
        '--output_dir',
        required=True,
        help="directory to save quantized model to",
        type=str
    )

    parser.add_argument(
        '--seq_length',
        default=512,
        help="sequence length to use",
        type=int
    )

    parser.add_argument(
        '--inc_config_file',
        help="INC conf yaml",
        required=True
    )

    FLAGS = parser.parse_args()

    main(FLAGS)
