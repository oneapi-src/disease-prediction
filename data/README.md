## Dataset Description

The dataset used for this demo is a synthetic symptom and prognosis dataset obtained from https://www.kaggle.com/kaushil268/disease-prediction-using-machine-learning.  In the dataset, each row corresponds to a list of symptom names and the corresponding prognosis for that particular set of syptoms.  
The original dataset consists of indicators for the symptom names however for our purposes to test an NLP workload, we first transform the data from indicators to string descriptions to emulate a situation where the symptoms come in the form of text.  An example input row for this after transformation would be 

> symptoms: itching, dischromic patches, nodal skin eruptions, skin rash
> prognosis: fungal infection

Furthermore, to add a bit of natural language variation, each list of symptoms is padded with a few random phrases to imitate the possible situation of noise and a few negative phrases as well.  For example, a possible variation of above list of symptoms could be.

> symptoms:  Itching. Reported signs of dischromic patches. Patient reports no patches in throat. Issues of frequent skin rash. Patient reports no spotting urination. Patient reports no stomach pain. nodal skin eruptions over the last few days.

### Setting Up the Data

The benchmarking scripts expects all of the data files to be present in `data/disease-prediction/` directory and transformed as above.

To setup the data for benchmarking under these requirements, run the following set of commands from the 
`data` directory.  

```shell
kaggle datasets download kaushil268/disease-prediction-using-machine-learning
unzip disease-prediction-using-machine-learning.zip -d disease-prediction
python prepare_data.py
```

A kaggle account is necessary to use the kaggle CLI.  Instructions can be found at https://github.com/Kaggle/kaggle-api.

> **Please see this data set's applicable license for terms and conditions. Intel Corporation does not own the rights to this data set and does not confer any rights to it.**
