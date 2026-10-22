# MRLN

**This is the data and code for our paper** `From Sparse Evidence to Shared Semantics: Robust Medication Recommendation via Multi-Granularity Representation Learning`.

This repository contains the implementation of MRLN for drug recommendation.

## Model Overview

At the $t$-th visit of patient, MRLN consists of three main components:

1. **Multi-granularity Input Representation Module**: Encodes diagnosis and procedure information from ICD and CCS branches to get patient representations $N_{d}^{t}$ and $N_{s}^{t}$.

2. **Contrastive Alignment Strategy**: Aligns the semantic space between different code granularities (ICD and CCS).

3. **Drug Representation Module**: Implements a retrieval augmented prediction mechanism to predict new drugs based on patient presentation and retrieve reusable drugs from historical medications, obtaining the final drug prediction combination $\hat{M}^{t}$.

4. **ME (Matching Enhancement)**: A component that enhances the matching between patient representations and historical medications.

## Prerequisites

Make sure your local environment has the following installed:


* `pytorch>=1.12.1 & <=1.9`
* `spacy == 2.1.9`
* `tensorboardx == 2.0`
* `tokenizers == 0.7.0`
* `tokenizers == 0.7.0`
* `numpy == 1.15.1`
* `python == 3.7`
* `transformers == 2.9.1`

## Data source 

We provide the dataset in the [datas](datas/) folder.

| Data      | Source                                                   | Description                                                  |
| --------- | -------------------------------------------------------- | ------------------------------------------------------------ |
| MIMIC-III | [This link](https://physionet.org/content/mimiciii/1.4/) | MIMIC-III is freely-available database from 2001 to 2012, which is associated with over forty thousand patients who stayed in critical care units |
| MIMIC-IV  | [This link](https://physionet.org/content/mimiciv/2.2/)  | MIMIC-IV is freely-available database between 2008 - 2019, which is associated with 299,712 patients who stayed in critical care units |


### Statistics of the Datasets

| Items                | MIMIC-III  | MIMIC-III-Rare | MIMIC-IV    | MIMIC-IV-Rare |
| -------------------- | ---------- | -------------- | ----------- | ------------- |
| Patients             | 6350       | 1587           | 8533        | 2133          |
| Clinical events      | 15032      | 3500           | 19251       | 4443          |
| ICD Diagnoses        | 1958       | 1907           | 1892        | 1883          |
| CCS Diagnoses        | 241        | 241            | 438         | 215           |
| ICD Procedures       | 1430       | 1202           | 1319        | 1231          |
| CCS Procedures       | 202        | 193            | 387         | 173           |
| Medicines            | 131        | 129            | 130         | 126           |


## Project Structure

```
MRLN-main/
├── models/                    # Model modules
│   ├── __init__.py
│   ├── multi_granularity_encoder.py  # Multi-granularity input representation
│   ├── contrastive_alignment.py      # Contrastive alignment strategy
│   ├── drug_decoder.py               # Drug representation module
│   └── mrl_net.py                    # Main MRLN model
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── data_loader.py         # Data loading utilities
│   ├── metrics.py             # Evaluation metrics
│   ├── losses.py              # Loss functions
│   └── memory_bank.py         # Memory bank for contrastive learning
├── datas/                     # Data directory
│   ├── voc_final.pkl
│   ├── diag_proc_ccs.pkl
│   ├── records_final.pkl
│   └── ddi_A_final.pkl
├── config.py                  # Configuration file
├── train_mrln.py             # Main training script
├── outer_models.py           # Additional model components
└── util.py                   # Utility functions
```
Clinical Classifications Software (CCS) for ICD-9-CM is a tool from HCUP.
Next, download the zip package from [web](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Single_Level_CCS_2015.zip) and unzip the file ```dx2015.csv``` and ```pr2015.csv```, respectively. 
use the script ```python ICD2CCS.py``` to obtain CCS labels and attach them on corresponding csv files. After the paper is accepted, we will further upload the relevant data preprocessing files.


## Usage

### Training

Run the main training script:

```bash
python train_mrln.py
```

### Configuration

Edit `config.py` to adjust hyperparameters:
- Model dimensions (`EMB_DIM`, `MED_DIM`)
- Training parameters (`LEARNING_RATE`, `EPOCHS`, `BATCH_SIZE`)
- Loss weights (`LOSS_WEIGHT_ICD`, `LOSS_WEIGHT_CCS`, `LOSS_WEIGHT_ALIGNMENT`)
- Contrastive learning parameters (`TEMPERATURE`, `MEMORY_BANK_SIZE`)

## Datasets

The code is designed to work with MIMIC-III and MIMIC-IV datasets. Place your processed data files in the `datas/` directory:
- `voc_final.pkl`: Vocabulary file
- `diag_proc_ccs.pkl`: CCS mapping file
- `records_final.pkl`: Patient records
- `ddi_A_final.pkl`: Drug-drug interaction matrix

## Model Architecture

### Multi-granularity Encoder
- Processes both ICD and CCS code branches
- Uses Transformer encoders for sequence modeling
- Integrates diagnosis and procedure information

### Contrastive Alignment
- Aligns ICD and CCS patient representations
- Uses memory bank for hard negative sampling
- Implements InfoNCE-style contrastive loss

### Drug Decoder
- Predicts new drugs based on patient presentation
- Retrieves reusable drugs from historical medications
- Uses matching enhancement (ME) component for better retrieval


## TODO

More training scripts for easy training will be added soon.




