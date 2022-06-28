# Complex Event Identification
This repository contains the code for the paper Capturing the Content of a Document through Complex Event Identification
## Abstract
Granular events, instantiated in a document by predicates, can usually be grouped into more general events, called {\it complex events}. Together, they capture the major content of the document. Recent work grouped granular events by defining event regions, filtering out sentences that are irrelevant to the main content. However, this approach assumes that a given complex event is always described in consecutive sentences, which does not always hold in practice. 
In this paper, we introduce the task of complex event identification. We address this task as a pipeline, first predicting whether two granular events mentioned in the text belong to the same complex event, independently of their position in the text, and then using this to cluster them into complex events. Due to the difficulty of predicting whether two granular events belong to the same complex event in isolation, we propose a context-augmented representation learning approach \textsc{ContextRL} that adds additional context to better model the pairwise relation of granular events. We show that our approach outperforms strong baselines on the complex event identification task and further present a promising case study showing the effectiveness of using complex events as input for document-level argument extraction.
## Requirements
The package  is built around <b>Python 3.6</b>

<i>For creating virtual environment with conda:</i>
```
conda create -n cei python=3.6 
conda activate cei
```
<i>For installing dependencies:</i>
```
pip install -r requirements.txt
```

## Dataset
To convert the data from the HiEve datatset to the complex event identification annotation and generate complex event cluster files for clustering metric evaluation, run
```
python complex_reader.py
```
The outputs are stored in **data/complex**.

## Model
Our model is stored at [Google Drive](https://drive.google.com/file/d/1a4oyeI5y6kPdhsIEQItRyUSkI5UFS90D/view?usp=sharing). Please download it and put it in the **models** folder.

## Model Training
After creating directory named "outputs", run the following-
```
bash train_attn.sh
```
The input is a pair of granular events and the output is the probability of belonging to the same complex event.
## Model Evaluation
To evaluate on the test set, run
```
bash test_attn.sh
```
This file includes three steps:
1. Generate the probability of belonging to the same complex event for each pair of events in the dev and test set.
2. Cluster events into complex events through agglomerative clustering. The threshold is finetuned on the development set.
3. Output the clustering evaluation metric scores of the test set.
4. Output two files. **ce_events.txt** shows granular event ids in the same complex event. **ce_context.txt** shows the context of complex events. The context is the sentences that contain the predicates of granular events in the same complex event.
