**A graph embedding model for drug-drug interactions and side-effect predicting:**

This is a graph embedding model aims to predict drug interactions and side effects.

**Introduction:**

This repository contains source code and dataset for the drug-drug interaction and side-effect prediciton task. A model is designed to solve these two biomedical task. The model is implemented through Tensorflow. The preprocessed data is split into 3 sets including training set, testing set and validation set.

**System requirements:**

The model is tested on Windows. It supports Tensorflow GPU version 2.x and Python >= 3.6. 

**Installation:** 
Using the following command to install the repository directly:

`
$ pip install git+https://git-teaching.cs.bham.ac.uk/mod-msc-proj-2020/cxj973.git
`

**How to produce the results:**

The whole evaluation precess can be divided into two task: A drug-drug interaction (DDI) prediction task and a side-effect prediction task.

- The DDI task:
1. Use the already precessed drug combinations data stored in the "data" file to perform DDI task. Replace the path in the code with your own path.
2. Run the "pred.py" in "embed" file to obtain the results.

- The side-effect predicting task:
1. Use the already precessed side effects data stored in the "data/pse" file to perform this task. Replace the path in the code with your own path.
2. Run the "prediction.py" to train the model and obtain the results.

**Examples:**

An example of DDI task can be found in [text](https://git-teaching.cs.bham.ac.uk/mod-msc-proj-2020/cxj973/-/blob/main/embed/pred.ipynb). And an example of the side effect prediction task can be found in [text](https://git-teaching.cs.bham.ac.uk/mod-msc-proj-2020/cxj973/-/blob/main/prediction.ipynb).

**Acknowledgement:**

Some concepts of the DDI prediciton model are based on [text](https://github.com/xiangyue9607/BioNEV) by xiangyue et al.
