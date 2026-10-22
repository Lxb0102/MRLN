"""
Configuration file for MRLN model
"""
import os

# Data paths (relative to project root)
DATA_DIR = 'datas'
VOCAB_FILE = os.path.join(DATA_DIR, 'voc_final.pkl')
CCS_VOCAB_FILE = os.path.join(DATA_DIR, 'diag_proc_ccs.pkl')
RECORDS_FILE = os.path.join(DATA_DIR, 'records_final.pkl')
DDI_MATRIX_FILE = os.path.join(DATA_DIR, 'ddi_A_final.pkl')
EHR_ADJ_FILE = os.path.join(DATA_DIR, 'ehr_adj_final.pkl')

# Model hyperparameters
EMB_DIM = 64
NHEAD = 2
DROPOUT = 0.2
MED_DIM = EMB_DIM * 2

# Training hyperparameters
LEARNING_RATE = 0.0001
EPOCHS = 50
BATCH_SIZE = 1
DEVICE = 'cuda'

# Loss weights
LOSS_WEIGHT_ICD = 0.7
LOSS_WEIGHT_CCS = 0.3
LOSS_WEIGHT_ALIGNMENT = 0.05
LOSS_WEIGHT_MARGIN = 0.02  # (dispensable)

# Contrastive alignment
TEMPERATURE = 0.07
MEMORY_BANK_SIZE = 512
MEMORY_BANK_DIM = 128
NUM_NEGATIVES = 4
CANDIDATE_POOL = 32

# Model save directory
SAVE_DIR = 'logs/state_dict'
LOG_DIR = 'logs'

# Data split
TRAIN_SPLIT = 2/3
EVAL_SPLIT = 1/2

