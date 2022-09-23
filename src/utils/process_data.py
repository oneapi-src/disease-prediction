# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Utility functions to create datasets.
"""

from typing import List

import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

MAPPING = {
    'Fungal infection': 0,
    'Allergy': 1,
    'GERD': 2,
    'Chronic cholestasis': 3,
    'Drug Reaction': 4,
    'Peptic ulcer diseae': 5,
    'AIDS': 6,
    'Diabetes ': 7,
    'Gastroenteritis': 8,
    'Bronchial Asthma': 9,
    'Hypertension ': 10,
    'Migraine': 11,
    'Cervical spondylosis': 12,
    'Paralysis (brain hemorrhage)': 13,
    'Jaundice': 14,
    'Malaria': 15,
    'Chicken pox': 16,
    'Dengue': 17,
    'Typhoid': 18,
    'hepatitis A': 19,
    'Hepatitis B': 20,
    'Hepatitis C': 21,
    'Hepatitis D': 22,
    'Hepatitis E': 23,
    'Alcoholic hepatitis': 24,
    'Tuberculosis': 25,
    'Common Cold': 26,
    'Pneumonia': 27,
    'Dimorphic hemmorhoids(piles)': 28,
    'Heart attack': 29,
    'Varicose veins': 30,
    'Hypothyroidism': 31,
    'Hyperthyroidism': 32,
    'Hypoglycemia': 33,
    'Osteoarthristis': 34,
    'Arthritis': 35,
    '(vertigo) Paroymsal  Positional Vertigo': 36,
    'Acne': 37,
    'Urinary tract infection': 38,
    'Psoriasis': 39,
    'Impetigo': 40
}

REVERSE_MAPPING = {v: k for k, v in MAPPING.items()}


class DiseasePrognosisDataset(Dataset):
    """Dataset with symptom strings to predict disease.

    Args:
        symptoms (List[str]): list of symptom strings
        prognosis (List[str]): list of corresponding prognosis
    """

    MAPPING = MAPPING

    def __init__(
            self,
            symptoms: List[str],
            prognosis: List[str],
            tokenizer: PreTrainedTokenizer,
            max_length: int = 64):

        self.symptoms = symptoms
        self.prognosis = prognosis
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.symptoms)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.symptoms[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True)
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        if self.prognosis is not None:
            return (item, torch.as_tensor(self.MAPPING[self.prognosis[idx]]))
        return item


def read_and_preprocess_data(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 64,
    include_label: bool = True
) -> Dataset:
    """read, preprocess data, and create a Dataset with a pretrained tokenizer

    Args:
        data_path (str): path to dataset
        tokenizer (PreTrainedTokenizer): tokenizer to use
        max_length (int): max length for tokenization
        include_label (bool): Whether to include the label field for

    Return:
        Dataset : preprocessed Dataset
    """

    data = pd.read_csv(data_path)
    if include_label:
        prognosis = data['prognosis']
    symptoms = data['symptoms']

    return DiseasePrognosisDataset(
        symptoms.values,
        prognosis.values if include_label else None,
        tokenizer,
        max_length=max_length
    )
