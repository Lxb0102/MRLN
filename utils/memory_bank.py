"""
Memory Bank for Contrastive Learning
Stores and retrieves negative samples for contrastive alignment
"""
import torch
import torch.nn.functional as F


class MemoryBank:
    """Memory bank for storing patient representations for contrastive learning"""
    
    def __init__(self, size, dim, device):
        self.size = size
        self.ptr = 0
        self.device = device
        # Initialize memory bank with random normalized vectors
        self.bank = torch.randn(size, dim, device=device)
        self.bank = F.normalize(self.bank, dim=1)
    
    @torch.no_grad()
    def update(self, reps):
        """
        Update memory bank with new representations
        
        Args:
            reps: tensor [B, dim]
        """
        reps = F.normalize(reps, dim=1)
        B = reps.size(0)
        if B > self.size:
            reps = reps[:self.size]
        
        if self.ptr + B > self.size:
            overflow = (self.ptr + B) - self.size
            self.bank[self.ptr:] = reps[:B - overflow]
            self.bank[:overflow] = reps[B - overflow:]
            self.ptr = overflow
        else:
            self.bank[self.ptr:self.ptr + B] = reps
            self.ptr = (self.ptr + B) % self.size
    
    def get_negatives(self, num_neg=64):
        """
        Sample random negatives from memory bank
        
        Args:
            num_neg: Number of negatives to sample
        
        Returns:
            tensor [num_neg, dim]
        """
        if num_neg > self.size:
            num_neg = self.size
        idx = torch.randint(0, self.size, (num_neg,), device=self.device)
        return self.bank[idx]
    
    def sample(self, num_neg=64):
        """Alias for get_negatives"""
        return self.get_negatives(num_neg=num_neg)
    
    def get_hard_negatives(self, query, num_neg=5, candidate_pool=50):
        """
        Get hard negatives (most similar) from memory bank
        
        Args:
            query: [B, D]
            num_neg: Number of negatives per query
            candidate_pool: Number of candidates to sample before selecting
        
        Returns:
            [B, num_neg, D]
        """
        B, D = query.size()
        
        # Sample candidates
        idx = torch.randint(0, self.size, (candidate_pool,), device=self.device)
        candidates = self.bank[idx]  # [candidate_pool, D]
        
        # Normalize
        query_norm = F.normalize(query, dim=1)
        candidates_norm = F.normalize(candidates, dim=1)
        
        # Compute similarity [B, candidate_pool]
        sim = torch.matmul(query_norm, candidates_norm.T)
        
        # Select top-k most similar (hard negatives)
        topk_idx = torch.topk(sim, k=num_neg, dim=1).indices  # [B, num_neg]
        
        # Get corresponding vectors
        hard_negatives = candidates_norm[topk_idx]  # [B, num_neg, D]
        
        return hard_negatives

