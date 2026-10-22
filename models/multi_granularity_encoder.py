"""
Multi-granularity Input Representation Module
Encodes diagnosis and procedure information from ICD and CCS branches
to get patient representations N^t_d and N^t_s
"""
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from outer_models import trd_encoder


class MultiGranularityEncoder(nn.Module):
    """Multi-granularity input representation module"""
    
    def __init__(self, voc_size, emb_dim=64, nhead=2, device='cpu'):
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.med_dim = emb_dim * 2
        self.nhead = nhead
        
        # Embeddings for ICD branch (diagnosis and procedure)
        self.diag_emb = nn.Embedding(voc_size[0], emb_dim, padding_idx=0, device=device)
        self.proc_emb = nn.Embedding(voc_size[1], emb_dim, padding_idx=0, device=device)
        
        # Embeddings for CCS branch (diagnosis and procedure)
        self.ccs_diag_emb = nn.Embedding(voc_size[3], emb_dim, padding_idx=0, device=device)
        self.ccs_proc_emb = nn.Embedding(voc_size[4], emb_dim, padding_idx=0, device=device)
        
        self.dropout = nn.Dropout(p=0.2)
        
        # Linear transformations for ICD branch
        self.diag_linear_1 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*2, device=device),
            nn.Tanh(),
            nn.Linear(emb_dim*2, emb_dim, device=device),
            nn.Dropout(0.3)
        )
        self.proc_linear_1 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*2, device=device),
            nn.Tanh(),
            nn.Linear(emb_dim*2, emb_dim, device=device),
            nn.Dropout(0.3)
        )
        
        # Final linear layers
        self.final_linear2 = nn.Sequential(
            nn.Linear(self.med_dim, self.med_dim * 4, device=device),
            nn.Tanh(),
            nn.Linear(self.med_dim * 4, self.med_dim * 4, device=device),
            nn.Tanh(),
            nn.Linear(self.med_dim * 4, self.med_dim, device=device),
            nn.Dropout(0.3)
        )
        
        # Transformer encoders
        self.diag_encoder = nn.TransformerEncoderLayer(
            emb_dim, self.nhead, batch_first=True, dropout=0.2, device=device
        )
        self.proc_encoder = nn.TransformerEncoderLayer(
            emb_dim, self.nhead, batch_first=True, dropout=0.2, device=device
        )
        
        # Integration layers (bidirectional GRU)
        self.diag_integ = trd_encoder(emb_dim=int(self.med_dim/2), device=device)
        self.proc_integ = trd_encoder(emb_dim=int(self.med_dim/2), device=device)
    
    def forward(self, diag_seq, proc_seq, diag_mask, proc_mask,
                ccs_diag_seq, ccs_proc_seq, ccs_diag_mask, ccs_proc_mask):
        """
        Encode multi-granularity inputs
        
        Returns:
            patient_rep: ICD patient representation N^t_d [B, V, D]
            ccs_patient_rep: CCS patient representation N^t_s [B, V, D]
        """
        max_diag_num = diag_seq.size()[-1]
        max_proc_num = proc_seq.size()[-1]
        ccs_max_diag_num = ccs_diag_seq.size()[-1]
        ccs_max_proc_num = ccs_proc_seq.size()[-1]
        max_visit_num = diag_seq.size()[1]
        batch_size = diag_seq.size()[0]
        
        # Embed and transform ICD branch
        diag_seq = self.diag_linear_1(
            self.diag_emb(diag_seq).view(batch_size * max_visit_num, max_diag_num, self.emb_dim)
        )
        proc_seq = self.proc_linear_1(
            self.proc_emb(proc_seq).view(batch_size * max_visit_num, max_proc_num, self.emb_dim)
        )
        
        # Embed and transform CCS branch
        ccs_diag_seq = self.diag_linear_1(
            self.ccs_diag_emb(ccs_diag_seq).view(
                batch_size * max_visit_num, ccs_max_diag_num, self.emb_dim
            )
        )
        ccs_proc_seq = self.proc_linear_1(
            self.ccs_proc_emb(ccs_proc_seq).view(
                batch_size * max_visit_num, ccs_max_proc_num, self.emb_dim
            )
        )
        
        ...
        ...
        ...            
        
        # Integrate sequences
        diag_rep = self.diag_integ(diag_seq)
        proc_rep = self.proc_integ(proc_seq)
        icd_patient_rep = torch.concat([diag_rep, proc_rep], dim=-1)
        patient_rep = self.final_linear2(icd_patient_rep)
        
        # CCS branch
        ccs_diag_rep = self.diag_integ(ccs_diag_seq)
        ccs_proc_rep = self.proc_integ(ccs_proc_seq)
        ccs_patient_rep_or = torch.concat([ccs_diag_rep, ccs_proc_rep], dim=-1)
        ccs_patient_rep = self.final_linear2(ccs_patient_rep_or)
        
        return patient_rep, ccs_patient_rep


