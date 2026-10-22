"""
Training script for MRLN (Multi-granularity Representation Learning Network)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import dill
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader

from config import *
from models import MRLN
from utils import (
    mimic_data, pad_batch_v2_train, pad_batch_v2_eval,
    multi_label_metric, ddi_rate_score, FocalLoss, MemoryBank
)
from outer_models import trd_encoder
from util import llprint

torch.manual_seed(1203)


def get_n_params(model):
    """Count model parameters"""
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def main():
    """Main training function"""
    device = DEVICE if torch.cuda.is_available() else 'cpu'
    
    # Load data
    print("Loading data...")
    voc = dill.load(open(VOCAB_FILE, 'rb'))
    ccs_voc = dill.load(open(CCS_VOCAB_FILE, 'rb'))
    data = dill.load(open(RECORDS_FILE, 'rb'))
    
    # Process CCS mappings
    for patient_num in range(len(data)):
        for vst_num in range(len(data[patient_num])):
            diag_ccs_list = [
                ccs_voc[0]['icd2ccs_idx'][i] 
                for i in data[patient_num][vst_num][0] 
                if i in ccs_voc[0]['icd2ccs_idx']
            ]
            proc_ccs_list = [
                ccs_voc[1]['icd2ccs_idx'][i] 
                for i in data[patient_num][vst_num][1] 
                if i in ccs_voc[1]['icd2ccs_idx']
            ]
            diag_ccs_list = list(set(diag_ccs_list)) if diag_ccs_list else [-1]
            proc_ccs_list = list(set(proc_ccs_list)) if proc_ccs_list else [-1]
            data[patient_num][vst_num].insert(3, diag_ccs_list)
            data[patient_num][vst_num].insert(4, proc_ccs_list)
    
    # Shift indices (add 1 for padding)
    for patient in range(len(data)):
        for vst in range(len(data[patient])):
            data[patient][vst][0] = [i+1 for i in data[patient][vst][0]]
            data[patient][vst][1] = [i+1 for i in data[patient][vst][1]]
            data[patient][vst][2] = [i+1 for i in data[patient][vst][2]]
            data[patient][vst][3] = [max(0, i + 1) for i in data[patient][vst][3]]
            data[patient][vst][4] = [max(0, i + 1) for i in data[patient][vst][4]]
    
    # Initialize memory bank for contrastive learning
    memory_bank = MemoryBank(size=MEMORY_BANK_SIZE, dim=MEMORY_BANK_DIM, device=device)
    
    # Get vocabulary sizes
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    diag_ccs_voc, proc_ccs_voc = ccs_voc[0], ccs_voc[1]
    
    voc_size = (
        len(diag_voc.idx2word) + 1,
        len(pro_voc.idx2word) + 1,
        len(med_voc.idx2word) + 1,
        len(diag_ccs_voc['idx2word']) + 1,
        len(proc_ccs_voc['idx2word']) + 1
    )
    
    # Split data
    split_point = int(len(data) * TRAIN_SPLIT)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]
    
    # Load DDI matrix
    ddi_matrix = dill.load(open(DDI_MATRIX_FILE, 'rb'))
    ddi_matrix = torch.tensor(ddi_matrix, device=device)
    ddi_matrix = torch.concat([torch.zeros_like(ddi_matrix[0]).unsqueeze(dim=-1), ddi_matrix], dim=-1)
    ddi_matrix = torch.concat([torch.zeros_like(ddi_matrix[0]).unsqueeze(dim=0), ddi_matrix], dim=0)
    
    # Create data loaders
    train_loader = DataLoader(
        [[j[:3] for j in i] for i in data_train],
        batch_size=BATCH_SIZE,
        collate_fn=pad_batch_v2_train,
        shuffle=False,
        pin_memory=False
    )
    ccs_train_loader = DataLoader(
        [[[j[3], j[4], j[2]] for j in i] for i in data_train],
        batch_size=BATCH_SIZE,
        collate_fn=pad_batch_v2_train,
        shuffle=False,
        pin_memory=False
    )
    
    train_loader = [patient[:5] for patient in train_loader]
    ccs_train_loader = [patient[:2] + patient[3:] for patient in ccs_train_loader]
    train_loader = [
        train_loader[patient] + ccs_train_loader[patient]
        for patient in range(len(train_loader))
    ]
    
    eval_loader = DataLoader(
        [[j[:3] for j in i] for i in data_eval],
        batch_size=BATCH_SIZE,
        collate_fn=pad_batch_v2_eval,
        shuffle=False,
        pin_memory=False
    )
    ccs_eval_loader = DataLoader(
        [[[j[3], j[4], j[2]] for j in i] for i in data_eval],
        batch_size=BATCH_SIZE,
        collate_fn=pad_batch_v2_eval,
        shuffle=False,
        pin_memory=False
    )
    eval_loader = [patient[:5] for patient in eval_loader]
    ccs_eval_loader = [patient[:2] + patient[3:] for patient in ccs_eval_loader]
    eval_loader = [
        eval_loader[patient] + ccs_eval_loader[patient]
        for patient in range(len(eval_loader))
    ]
    
    # Initialize model
    model = MRLN(
        emb_dim=EMB_DIM,
        voc_size=voc_size,
        device=device,
        ddi_graph=ddi_matrix
    ).to(device)
    
    print(f'Model parameters: {get_n_params(model)}')
    
    # Initialize optimizer and loss
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = FocalLoss(alpha=0.5, gamma=0.55)
    
    # Contrastive alignment module
    from models.contrastive_alignment import ContrastiveAlignment
    contrastive_alignment = ContrastiveAlignment(temperature=TEMPERATURE)
    
    # Training setup
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    best_prauc = 0.0
    best_jac = 0.0
    best_model_path = None
    
    # Training loop
    for epoch in range(EPOCHS):
        print(f'\n=== Epoch {epoch + 1}/{EPOCHS} ===')
        
        # Training phase
        model.train()
        epoch_loss_list = []
        
        for index, datas in enumerate(train_loader):
            datas = [i.to(device) for i in datas]
            output, patient_rep, ccs_patient_rep = model(datas)
            
            # Prepare ground truth
            gt_container = torch.zeros_like(output[0], device=device).reshape(-1, voc_size[2])
            loss3_target = np.full((output[0].size()), -1).reshape([-1, voc_size[2]])
            
            for batch_idx, batch in enumerate(datas[2][0]):
                for idx, seq in enumerate(batch):
                    gt_container[batch_idx][seq] = 1.0
            
            gt_container = gt_container.reshape(output[0].size())
            gt_container[:, 0] = 0
            
            # Contrastive alignment loss
            query = patient_rep.squeeze(1)  # [B, D]
            positive = ccs_patient_rep.squeeze(1)  # [B, D]
            negatives = memory_bank.get_hard_negatives(
                query, num_neg=NUM_NEGATIVES, candidate_pool=CANDIDATE_POOL
            )  # [B, num_neg, D]
            
            alignment_loss = contrastive_alignment(query, positive, negatives)
            
            # Prediction losses
            loss_1 = criterion(output[0], gt_container)
            ccs_loss_1 = criterion(output[2], gt_container)
            loss_2 = F.multilabel_margin_loss(
                output[0], torch.LongTensor(loss3_target).to(device)
            )
            ccs_loss_2 = F.multilabel_margin_loss(
                output[2], torch.LongTensor(loss3_target).to(device)
            )
            
            # Total loss
            loss = (LOSS_WEIGHT_ICD * (loss_1 + LOSS_WEIGHT_MARGIN * loss_2) +
                   LOSS_WEIGHT_CCS * (ccs_loss_1 + LOSS_WEIGHT_MARGIN * ccs_loss_2) +
                   LOSS_WEIGHT_ALIGNMENT * alignment_loss)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Update memory bank
            with torch.no_grad():
                ccs_patient_rep = ccs_patient_rep.squeeze(1)
                memory_bank.update(ccs_patient_rep)
            
            epoch_loss_list.append(loss.item())
            
            llprint(f'\r|{"#" * int(50 * (index / len(train_loader)))}{"-" * int(50 - 50 * (index / len(train_loader)))}|'
                   f'{100 * (index / len(train_loader)):.2f}%|train_step:{index}/{len(train_loader)}')
        
        avg_train_loss = np.mean(epoch_loss_list) if epoch_loss_list else 0.0
        print()
        
        # Evaluation phase
        model.eval()
        prob_container = []
        gt_container = []
        labels_container = []
        ddi_cnt = 0
        ddi_all_cnt = 0
        avg_med = 0
        avg_precise = 0
        avg_recall = 0
        avg_f1 = 0
        count = 1e-6
        
        with torch.no_grad():
            for index, datas in enumerate(eval_loader):
                datas = [i.to(device) for i in datas]
                output, patient_rep, ccs_patient_rep = model(datas)
                
                output = output[0]
                gt_data = datas[2][0]
                
                for idx, vst in enumerate(output.reshape(-1, voc_size[2])):
                    gt_temp = torch.zeros_like(vst, device=device)
                    gt_temp[gt_data[idx]] = 1
                    gt_temp[0] = 0
                    avg_med += vst.sum()
                    out_labels = torch.where(vst > 0.35, 1.0, 0.0)
                    
                    labels_container.append(out_labels)
                    prob_container.append(vst)
                    gt_container.append(gt_temp)
                    
                    if gt_temp.sum() != 0:
                        precise = (out_labels * gt_temp).sum() / (out_labels.sum() + 1e-9)
                        recall = (out_labels * gt_temp).sum() / (gt_temp.sum() + 1e-9)
                    else:
                        continue
                    
                    avg_precise += precise
                    avg_recall += recall
                    
                    if (precise + recall) == 0:
                        continue
                    else:
                        f1 = (2.0 * precise * recall) / (precise + recall)
                    avg_f1 += f1
                    
                    ddi_temp_container = out_labels * out_labels.T.unsqueeze(dim=-1)
                    ddi_cnt += ddi_rate_score([ddi_temp_container], ddi_matrix)[0]
                    ddi_all_cnt += ddi_rate_score([ddi_temp_container], ddi_matrix)[1]
                    
                    count += 1
                
                llprint(f'\r|{"@" * int(50 * (index / len(eval_loader)))}{"-" * int(50 - 50 * (index / len(eval_loader)))}|'
                       f'{100 * (index / len(eval_loader)):.2f}%|eval_step:{index}/{len(eval_loader)}')
        
        # Calculate metrics
        avg_precise = avg_precise / count
        avg_recall = avg_recall / count
        avg_f1 = avg_f1 / count
        
        jac, prauc, F_1 = multi_label_metric(gt_container, labels_container, prob_container, voc_size=voc_size)
        
        try:
            ddi_rate = ddi_cnt / ddi_all_cnt
        except:
            print('No drug interaction pairs')
            ddi_rate = 0
        
        print(f'\navg_prc = {avg_precise:.4f}')
        print(f'avg_rec = {avg_recall:.4f}')
        print(f'jac = {jac:.4f}')
        print(f'prauc = {prauc:.4f}')
        print(f'avg_f1 = {avg_f1:.4f}')
        print(f'ddi_rate = {ddi_rate:.4f}')
        print(f'avg_med = {avg_med/count:.4f}')
        
        # Save metrics
        log_file = os.path.join(LOG_DIR, 'epoch_metrics.csv')
        write_header = not os.path.exists(log_file)
        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_f1': avg_f1,
            'val_jaccard': jac,
            'val_prauc': prauc,
            'val_ddi_rate': ddi_rate,
            'val_precision': avg_precise,
            'val_recall': avg_recall,
            'avg_med_count': avg_med / count
        }
        
        df = pd.DataFrame([{k: float(v) if torch.is_tensor(v) else v for k, v in epoch_result.items()}])
        df.to_csv(log_file, mode='a', header=write_header, index=False)
        
        # Save best model
        if prauc > best_prauc or (prauc == best_prauc and jac > best_jac):
            if best_model_path is not None and os.path.exists(best_model_path):
                try:
                    os.remove(best_model_path)
                except:
                    pass
            
            best_prauc = prauc
            best_jac = jac
            best_model_path = os.path.join(
                SAVE_DIR,
                f'best_model_epoch_{epoch+1}_prauc_{prauc:.4f}_ja_{jac:.4f}.pth'
            )
            torch.save(model.state_dict(), best_model_path)
            print(f'\n*** Saved best model to: {best_model_path} ***')
            print(f'*** Best metrics - PRAUC: {best_prauc:.4f}, Jaccard: {best_jac:.4f} ***\n')
    
    print(f'\nTraining completed! Best model saved to: {best_model_path}')
    print(f'Best metrics - PRAUC: {best_prauc:.4f}, Jaccard: {best_jac:.4f}')


if __name__ == '__main__':
    main()

