"""
Drug Representation Module
Retrieval augmented prediction mechanism to predict new drugs based on 
patient presentation and retrieve reusable drugs from historical medications
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DrugDecoder(nn.Module):
    """Drug representation module with retrieval augmented prediction"""
    
    def __init__(self, voc_size, med_dim, device='cpu'):
        super().__init__()
        self.device = device
        self.voc_size = voc_size
        self.med_dim = med_dim
        
        # Drug prediction blocks
        self.med_block = nn.Parameter(torch.randn([med_dim, voc_size[2]-1], device=device))
        
        # Patient-drug matching contactor (ME: Matching Enhancement)
        self.patient_mem_contact = nn.Sequential(
            nn.Linear(med_dim*2, med_dim*4, device=device),
            nn.Tanh(),
            nn.Linear(med_dim*4, voc_size[2]-1, device=device),
            nn.Tanh(),
            nn.Dropout(0.2)
        )
    
    def history_gate_unit(self, patient_rep, all_vst_drug, contacter):
        """
        History gate unit for retrieving reusable drugs from historical medications
        
        Args:
            patient_rep: Patient representation [V, D]
            all_vst_drug: Historical medications [B, V, num_drugs]
            contacter: Matching enhancement component
        
        Returns:
            his_seq_enhance: Enhanced drug predictions from history [V, num_drugs]
        """
        # Shift patient representation to align with previous visit
      ...
        # Apply gating and aggregate
        his_seq_enhance = his_seq_filter_mask * his_seq_container * all_vst_drug
        his_seq_enhance = his_seq_enhance.sum(dim=1)
        
        return his_seq_enhance.reshape(-1, self.voc_size[2] - 1)
    
    def forward(self, drug_mem, patient_rep, ccs_patient_rep):
        """
        Predict drugs using retrieval augmented mechanism
        
        Args:
            drug_mem: Historical medications [1, V, M]
            patient_rep: ICD patient representation [B, V, D]
            ccs_patient_rep: CCS patient representation [B, V, D]
        
        Returns:
            prob: Final drug prediction probabilities [V, num_drugs]
            vs_prob: Visible ICD probabilities [V, num_drugs]
            vs_ccs_prob: Visible CCS probabilities [V, num_drugs]
        """
        # Process historical medications
        drug_mem = torch.nn.functional.one_hot(
            drug_mem.squeeze(dim=0), num_classes=self.voc_size[2]
        ).sum(dim=-2)[:, 1:].to(torch.float32)
        
        drug_mem_pad = torch.zeros_like(drug_mem[0]).unsqueeze(dim=0)
        drug_mem = torch.cat([drug_mem_pad, drug_mem], dim=0)[:drug_mem.size()[0]].unsqueeze(dim=0).repeat(
            [patient_rep.size()[0], 1, 1]
        ).to(self.device)
        
        # Get patient representations
        patient_rep = patient_rep.squeeze(dim=1)
        ccs_patient_rep = ccs_patient_rep.squeeze(dim=1)
        
        # Retrieve reusable drugs from history (ME component)
        his_enhance = self.history_gate_unit(patient_rep, drug_mem, self.patient_mem_contact)
        ccs_his_enhance = self.history_gate_unit(ccs_patient_rep, drug_mem, self.patient_mem_contact)
        
        # Predict new drugs based on patient presentation
        final_prob_1 = patient_rep @ self.med_block
        ccs_final_prob_1 = ccs_patient_rep @ self.med_block
        
        # Combine prediction and retrieval
        prob = final_prob_1 + his_enhance
        ccs_prob = ccs_final_prob_1 + ccs_his_enhance
        
        prob = prob.reshape(-1, self.voc_size[2]-1)
        ccs_prob = ccs_prob.reshape(-1, self.voc_size[2] - 1)
        
        prob_padder = torch.full_like(prob.T[0], 0).unsqueeze(dim=0).T
        
        # Visible probabilities for visualization
        vs_prob = torch.cat([prob_padder, F.sigmoid(prob)], dim=-1)
        vs_ccs_prob = torch.cat([prob_padder, F.sigmoid(ccs_prob)], dim=-1)
        
        # Final combined prediction
        prob = prob + ccs_prob
        prob = F.sigmoid(prob)
        prob = torch.cat([prob_padder, prob], dim=-1)
        
        return prob, prob * prob.T.unsqueeze(dim=-1), vs_prob, vs_ccs_prob


