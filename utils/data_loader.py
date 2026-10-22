"""
Data Loader for MRLN
"""
from torch.utils import data
import torch


class mimic_data(data.Dataset):
    """MIMIC dataset loader"""
    
    def __init__(self, data):
        super().__init__()
        self.data = data
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


def pad_batch_v2_train(batch):
    """
    Pad batch for training
    
    Args:
        batch: List of patient sequences
    
    Returns:
        Padded tensors for diagnosis, procedure, medication, and masks
    """
    seq_length = torch.tensor([len(data) for data in batch])
    batch_size = len(batch)
    max_seq = max(seq_length)
    
    d_length_matrix = []
    p_length_matrix = []
    m_length_matrix = []
    d_max_num = 0
    p_max_num = 0
    m_max_num = 0
    
    for data_item in batch:
        d_buf, p_buf, m_buf = [], [], []
        for idx, seq in enumerate(data_item):
            d_buf.append(len(seq[0]))
            p_buf.append(len(seq[1]))
            m_buf.append(len(seq[2]))
            d_max_num = max(d_max_num, len(seq[0]))
            p_max_num = max(p_max_num, len(seq[1]))
            m_max_num = max(m_max_num, len(seq[2]))
        d_length_matrix.append(d_buf)
        p_length_matrix.append(p_buf)
        m_length_matrix.append(m_buf)
    
    # Generate masks
    d_mask_matrix = torch.full((batch_size, max_seq, d_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(d_length_matrix[i])):
            d_mask_matrix[i, j, :d_length_matrix[i][j]] = 0.
    
    p_mask_matrix = torch.full((batch_size, max_seq, p_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(p_length_matrix[i])):
            p_mask_matrix[i, j, :p_length_matrix[i][j]] = 0.
    
    # Generate tensors
    disease_tensor = torch.full((batch_size, max_seq, d_max_num), 0)
    procedure_tensor = torch.full((batch_size, max_seq, p_max_num), 0)
    medication_tensor = torch.full((batch_size, max_seq, m_max_num), 0)
    age_tensor = torch.full((batch_size, max_seq, 1), 0)
    gender_tensor = torch.full((batch_size, max_seq, 1), 0)
    
    # Fill tensors
    for b_id, data_item in enumerate(batch):
        for s_id, adm in enumerate(data_item):
            disease_tensor[b_id, s_id, :len(adm[0])] = torch.tensor(adm[0])
            procedure_tensor[b_id, s_id, :len(adm[1])] = torch.tensor(adm[1])
            medication_tensor[b_id, s_id, :len(adm[2])] = torch.tensor(adm[2])
            try:
                age_tensor[b_id, s_id, 0] = torch.tensor(adm[3])
                gender_tensor[b_id, s_id, 0] = torch.tensor(adm[4])
            except:
                pass
    
    try:
        return disease_tensor, procedure_tensor, medication_tensor, \
               d_mask_matrix, p_mask_matrix, age_tensor, gender_tensor
    except:
        return disease_tensor, procedure_tensor, medication_tensor, \
               d_mask_matrix, p_mask_matrix


def pad_batch_v2_eval(batch):
    """
    Pad batch for evaluation
    
    Args:
        batch: List of patient sequences
    
    Returns:
        Padded tensors for diagnosis, procedure, medication, and masks
    """
    seq_length = torch.tensor([len(data) for data in batch])
    batch_size = len(batch)
    max_seq = max(seq_length)
    
    d_length_matrix = []
    p_length_matrix = []
    m_length_matrix = []
    d_max_num = 0
    p_max_num = 0
    m_max_num = 0
    
    for data_item in batch:
        d_buf, p_buf, m_buf = [], [], []
        for idx, seq in enumerate(data_item):
            d_buf.append(len(seq[0]))
            p_buf.append(len(seq[1]))
            m_buf.append(len(seq[2]))
            d_max_num = max(d_max_num, len(seq[0]))
            p_max_num = max(p_max_num, len(seq[1]))
            m_max_num = max(m_max_num, len(seq[2]))
        d_length_matrix.append(d_buf)
        p_length_matrix.append(p_buf)
        m_length_matrix.append(m_buf)
    
    # Generate masks
    d_mask_matrix = torch.full((batch_size, max_seq, d_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(d_length_matrix[i])):
            d_mask_matrix[i, j, :d_length_matrix[i][j]] = 0.
    
    p_mask_matrix = torch.full((batch_size, max_seq, p_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(p_length_matrix[i])):
            p_mask_matrix[i, j, :p_length_matrix[i][j]] = 0.
    
    # Generate tensors
    disease_tensor = torch.full((batch_size, max_seq, d_max_num), 0)
    procedure_tensor = torch.full((batch_size, max_seq, p_max_num), 0)
    medication_tensor = torch.full((batch_size, max_seq, m_max_num), 0)
    age_tensor = torch.full((batch_size, max_seq, 1), 0)
    gender_tensor = torch.full((batch_size, max_seq, 1), 0)
    
    # Fill tensors
    for b_id, data_item in enumerate(batch):
        for s_id, adm in enumerate(data_item):
            disease_tensor[b_id, s_id, :len(adm[0])] = torch.tensor(adm[0])
            procedure_tensor[b_id, s_id, :len(adm[1])] = torch.tensor(adm[1])
            medication_tensor[b_id, s_id, :len(adm[2])] = torch.tensor(adm[2])
            try:
                age_tensor[b_id, s_id, 0] = torch.tensor(adm[3])
                gender_tensor[b_id, s_id, 0] = torch.tensor(adm[4])
            except:
                pass
    
    try:
        return disease_tensor, procedure_tensor, medication_tensor, \
               d_mask_matrix, p_mask_matrix, age_tensor, gender_tensor
    except:
        return disease_tensor, procedure_tensor, medication_tensor, \
               d_mask_matrix, p_mask_matrix

