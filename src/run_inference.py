# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Run inference benchmarks
"""

import argparse
import logging
import os
import pathlib
import time
from contextlib import nullcontext

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import BertConfig, BertForSequenceClassification

from utils.process_data import read_and_preprocess_data, REVERSE_MAPPING


def inference(predict_fn, batch, flags) -> float:
    """Run inference using the provided `predict_fn`

    Args:
        predict_fn: prediction function to use
        batch: data batch from a data loader
        n_runs: number of benchmark runs to time

    Returns:
        float : Average prediction time
    """
    n_runs = flags.n_runs
    enable_bf16 = flags.bf16
    times = []
    predictions = []
    with torch.no_grad():
        # use mixed precision bf16 inference only if enabled
        with torch.cpu.amp.autocast() if enable_bf16 else nullcontext():
            for _ in range(2 + n_runs):
                start = time.time()
                res = predict_fn(batch)
                end = time.time()
                predictions.append(res)
                times.append(end - start)

    avg_time = np.mean(times[2:])
    return avg_time


def main(flags) -> None:
    """Setup model for inference and perform benchmarking

    Args:
        FLAGS: benchmarking flags
    """

    if flags.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        path = pathlib.Path(flags.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=flags.logfile, level=logging.DEBUG)
    logger = logging.getLogger()

    if not os.path.exists(flags.saved_model_dir):
        logger.error("Saved model %s not found!", flags.saved_model_dir)
        return

    # Load dataset into memory
    tokenizer = AutoTokenizer.from_pretrained(flags.saved_model_dir)

    try:
        test_dataset = read_and_preprocess_data(
            flags.input_file,
            tokenizer,
            max_length=flags.seq_length,
            include_label=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=flags.batch_size, shuffle=False
        )
    except FileNotFoundError as exc:
        logger.error("Please follow instructions to download data.")
        logger.error(exc, exc_info=True)
        return

    # Load model into memory, if INC, need special loading
    if flags.is_inc_int8:
        from neural_compressor.utils.pytorch import load
        config = BertConfig.from_json_file(
            os.path.join(flags.saved_model_dir, "config.json")
        )
        model = BertForSequenceClassification(config=config)
        model = load(flags.saved_model_dir, model)

        # re-establish logger because it breaks from above
        logging.getLogger().handlers.clear()

        if flags.logfile == "":
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(filename=flags.logfile, level=logging.DEBUG)
        logger = logging.getLogger()

    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            flags.saved_model_dir
        )

    # JIT model for faster execution
    batch = next(iter(test_loader))
    token_ids = batch['input_ids']
    mask = batch['attention_mask']

    jit_inputs = (token_ids, mask)

    if flags.intel:
        # if using intel, optimize the model
        import intel_extension_for_pytorch as ipex

        logger.info("Using IPEX to optimize model")

        model.eval()
        
        # select dtype based on the flag
        if flags.bf16:
            dtype = torch.bfloat16
        else:
            dtype = None # default dtype for ipex.optimize()
        
        with torch.no_grad(), torch.cpu.amp.autocast() if flags.bf16 else nullcontext():
            model = ipex.optimize(model, dtype=dtype)
            model = torch.jit.trace(
                model,
                jit_inputs,
                check_trace=False,
                strict=False
            )
            model = torch.jit.freeze(model)
    
    else:
        if flags.is_inc_int8:
            logger.info("Using INC Quantized model")
        else:    
            logger.info("Using stock model")

        model.eval()
        model = torch.jit.trace(
            model,
            jit_inputs,
            check_trace=False,
            strict=False
        )
        model = torch.jit.freeze(model)

    def predict(
        batch
    ) -> torch.Tensor:
        """Predicts the output for the given batch
            using the given PyTorch model.

        Args:
            batch (torch.Tensor): data batch from data loader
                transformers tokenizer

        Returns:
            torch.Tensor: predicted quantities
        """
        res = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'])
        return res

    if flags.benchmark_mode:
        logger.info("Running experiment n = %d, b = %d, l = %d",
                    flags.n_runs, flags.batch_size, flags.seq_length)

        average_time = inference(predict, batch, FLAGS)
        logger.info('Avg time per batch : %.3f s', average_time)
    else:
        predictions = []
        index = 0
        for _, batch in enumerate(test_loader):
            pred_probs = torch.softmax(
                predict(batch)['logits'], axis=1
            ).detach().numpy()
            for i in range(len(pred_probs)):
                probs = {
                    REVERSE_MAPPING[x]: pred_probs[i, x]
                    for x in np.argsort(pred_probs[i, :])[::-1][:5]
                }
                predictions.append(
                    {'id': index, 'prognosis': probs}
                )
                index += 1
        print({"predictions": predictions})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--saved_model_dir',
        required=True,
        help="saved pretrained model to benchmark",
        type=str
    )

    parser.add_argument(
        '--input_file',
        required=True,
        help="input to make predictions on",
        type=str
    )

    parser.add_argument(
        '--batch_size',
        default=-1,
        type=int,
        help="batch size to use. if -1, uses all entries in input."
    )

    parser.add_argument(
        '--benchmark_mode',
        default=False,
        help="Benchmark instead of get predictions.",
        action="store_true"
    )

    parser.add_argument(
        '--intel',
        default=False,
        action="store_true",
        help="use intel accelerated technologies. defaults to False."
    )

    parser.add_argument(
        '--is_inc_int8',
        default=False,
        action="store_true",
        help="saved model dir is a quantized int8 model. defaults to False."
    )

    parser.add_argument(
        '--seq_length',
        default=512,
        help="sequence length to use. defaults to 512.",
        type=int
    )

    parser.add_argument(
        '--logfile',
        help="logfile to use.",
        default="",
        type=str
    )

    parser.add_argument(
        '--n_runs',
        default=100,
        help="number of trials to test. defaults to 100.",
        type=int
    )
    
    parser.add_argument(
        '--bf16',
        default=False,
        action="store_true",
        help="Enable bf16 inference"
    )

    FLAGS = parser.parse_args()

    main(FLAGS)
