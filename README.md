# Complex Event Identification

## Requirements
```
pip install requirements
```

## Data Conversion
To convert the data from the HiEve datatset to the complex event identification annotation and generate complex event cluster files for clustering metric evaluation, run
```
python complex_reader.py
```
The outputs are stored in **data/complex**.

## Model Training
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
