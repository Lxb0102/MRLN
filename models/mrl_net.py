"""
MRLN: Multi-granularity Representation Learning Network
Main model combining all modules
"""
import torch
import torch.nn as nn
from .multi_granularity_encoder import MultiGranularityEncoder
from .drug_decoder import DrugDecoder


class MRLN(nn.Module):
    """
    MRLN Model
    
    At the t-th visit of patient:
    1. Multi-granularity input representation: encode diagnosis and procedure 
       from ICD and CCS branches to get N^t_d and N^t_s
    2. Contrastive alignment: align semantic space between different code granularities
    3. Drug representation: retrieval augmented prediction to get final drug combination M^t
    """
    
    def __init__(self, voc_size, emb_dim=64, nhead=2, device='cpu', ddi_graph=None, ehr_graph=None):
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.med_dim = emb_dim * 2
        self.voc_size = voc_size
        
        # Multi-granularity encoder
        self.encoder = MultiGranularityEncoder(voc_size, emb_dim, nhead, device)
        
        # Drug decoder
        self.decoder = DrugDecoder(voc_size, self.med_dim, device)
    
    def forward(self, input, diag_mask=None, proc_mask=None, ages=None, gender=None, drug_mem=None):
        """
        Forward pass
        
        Args:
            input: List containing [diag_seq, proc_seq, drug_seq, diag_mask, proc_mask,
                   ccs_diag_seq, ccs_proc_seq, ccs_diag_mask, ccs_proc_mask]
        
        Returns:
            decoder_out: Drug predictions [prob, prob*prob.T, vs_prob, vs_ccs_prob]
            patient_rep: ICD patient representation
            ccs_patient_rep: CCS patient representation
        """
        # Encode multi-granularity inputs
        patient_rep, ccs_patient_rep = self.encoder(
            input[0], input[1], input[3], input[4],
            input[5], input[6], input[7], input[8]
        )
        
        # Decode drugs with retrieval augmented prediction
        decoder_out = self.decoder(
            drug_mem=input[2],
            patient_rep=patient_rep,
            ccs_patient_rep=ccs_patient_rep
        )
        
        return decoder_out, patient_rep, ccs_patient_rep

