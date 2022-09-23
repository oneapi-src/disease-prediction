# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Transform raw data into sentences to use within BERT.
"""

import random

import pandas as pd
from sklearn.model_selection import train_test_split

random.seed(0)
templates = [
    "Patient is experiencing {}.",
    "Reported signs of {}.",
    "Occasional {} experienced by patient.",
    "{}.",
    "Issues of frequent {}.",
    "{} over the last few days.",
    "Sporadic {}.",
    "Mild case of {}."
]

neg_templates = [
    "Patient reports no {}.",
    "No evidence of {} seen.",
    "{} is not present."
]


def to_symptoms_string(row: pd.Series) -> str:
    """transform an indicator row of symptoms into a shuffled string

    Args:
        row (pd.Series): indicator row of symptoms

    Returns:
        str: shuffled string representation of indicator symptoms
    """
    symptoms = row.index.values[row.values == 1].tolist()
    non_symptoms = row.index.values[row.values == 0].tolist()
    random.shuffle(symptoms)
    symptom_sentences = []
    for symptom in symptoms:
        symptom_sentences.append(
            templates[random.randint(0, len(templates) - 1)]
            .replace("{}", symptom)
        )

    total_negative = 0
    for non_symptom in non_symptoms:
        if random.random() < 0.1 and total_negative < 3:
            symptom_sentences.append(
                neg_templates[random.randint(0, len(neg_templates) - 1)]
                .replace("{}", non_symptom)
            )
            total_negative += 1

    random.shuffle(symptom_sentences)

    res = " ".join(symptom_sentences) \
        .replace(" _", " ") \
        .replace("_ ", " ") \
        .replace("_", " ")

    return res


def add_noise(data):
    if 'Unnamed: 133' in data.columns:
        data = data.drop('Unnamed: 133', axis=1)
    data['symptoms'] = data.apply(to_symptoms_string, axis=1)
    return data[['symptoms', 'prognosis']]


training = pd.read_csv("disease-prediction/Training.csv")
testing = pd.read_csv("disease-prediction/Testing.csv")

data = pd.concat([training, testing])

train, test = train_test_split(
    data,
    test_size=0.2,
    random_state=0,
    stratify=data.prognosis
)

add_noise(train).to_csv("disease-prediction/Training.csv", index=None)
add_noise(test).to_csv("disease-prediction/Testing.csv", index=None)
