# MRLN
**This is the data and code for our paper** `Retrieval Augmented Multi-granularity Representation Learning for Medication Recommendation`.

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

## Datastes

We provide the dataset in the [datas](datas/) folder.

| Data      | Source                                                   | Description                                                  |
| --------- | -------------------------------------------------------- | ------------------------------------------------------------ |
| MIMIC-III | [This link](https://physionet.org/content/mimiciii/1.4/) | MIMIC-III is freely-available database from 2001 to 2012, which is associated with over forty thousand patients who stayed in critical care units |
| MIMIC-IV  | [This link](https://physionet.org/content/mimiciv/2.2/)  | MIMIC-IV is freely-available database between 2008 - 2019, which is associated with 299,712 patients who stayed in critical care units |

## Documentation

```
--src
  │--README.md
  │--data_loader.py
  │--train.py
  │--model_net.py
  │--outer_models.py
  │--util.py
  |--ICD2CCS.py
  
--datas
  │--ddi_A_final.pkl
  |--diag_proc_ccs
  |--diag_proc_ccs_4.pkl
  |--ehr_adj_final.pkl
  |--records_final.pkl
  |--records_final_4.pkl
  |--voc_final.pkl
  |--voc_final_4.pkl

```


Clinical Classifications Software (CCS) for ICD-9-CM is a tool from HCUP.
Next, download the zip package from [web](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Single_Level_CCS_2015.zip) and unzip the file ```dx2015.csv``` and ```pr2015.csv```, respectively. 
use the script ```python ICD2CCS.py``` to obtain CCS labels and attach them on corresponding csv files. After the paper is accepted, we will further upload the relevant data preprocessing files.

## Train

Please run `train.py` to begin training and testing.

On a single NVIDIA® GeForce RTX™ 3080 Ti (10GB) GPU, a typical run takes 2.5 hours to complete.

## TODO
More training scripts for easy training will be added soon.

Please feel free to contact me xiaobi.li@dlmu.edu.cn for any question.

Partial credit to previous reprostories:

- https://github.com/sjy1203/GAMENet
- https://github.com/ycq091044/SafeDrug
- https://github.com/BarryRun/COGNet

