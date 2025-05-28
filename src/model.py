import torch
import torch.nn as nn
from torchdiffeq import odeint
from src.module import IdealFilter, DegreeNorm, LinearFilter

class AllRankRec(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, observed_inter):
        """
        Predict preference scores for all items given the observed interaction.
        Args:
            observed_inter (torch.Tensor): A binary matrix of shape (batch_size, num_items),
                where 1 indicates an observed interaction.
        Returns:
            pred_score (torch.Tensor): A score matrix of shape (batch_size, num_items),
            where higher values indicate higher predicted preference.
        """
        pass
    
    def mask_observed(self, pred_score, observed_inter):
        # Mask out the scores for items that have been already interacted with.
        return pred_score * (1 - observed_inter) - 1e8 * observed_inter

    def full_predict(self, observed_inter):
        pred_score = self.forward(observed_inter)
        return self.mask_observed(pred_score, observed_inter)
    
class BSPM(AllRankRec):
    def __init__(self,
                 sharp_solv, sharp_step, sharp_time,
                 ideal_cutoff, ideal_weight,
                 early_merge, sharp_off, point_combi):
        super().__init__()
        self.sharp_solv = sharp_solv
        self.early_merge = early_merge
        self.sharp_off = sharp_off
        self.point_combi = point_combi
        
        if not sharp_off:
            sharp_ts = torch.linspace(0, sharp_time, sharp_step+1).float()
            self.register_buffer('sharp_ts', sharp_ts)
        
        self.linear = LinearFilter()
        self.ideal = None
        if ideal_cutoff > 0 and ideal_weight > 0:
            self.ideal = IdealFilter(ideal_cutoff, ideal_weight)
            self.norm = DegreeNorm(0.5)
        
    def fit(self, inter):
        self.linear.fit(inter)
        if self.ideal:
            self.ideal.fit(inter)
            self.norm.fit(inter)
            
    def sharp_func(self, t, r):
        return self.linear.sharpen(r)
            
    def forward(self, inter):
        # Blurring (Heat Equation)
        linear_out = self.linear.forward(inter)
        # Bluarring (Ideal Low-pass Filter)
        if self.ideal:
            ideal_out = self.norm.forward_pre(inter)
            ideal_out = self.ideal.forward(ideal_out)
            ideal_out = self.norm.forward_post(ideal_out)
        # Early Merge
        out = linear_out
        if self.early_merge and self.ideal:
            out += ideal_out
        # Sharpening                
        if not self.sharp_off:
            out = odeint(func=self.sharp_func, y0=out, t=self.sharp_ts, method=self.sharp_solv)      
            if self.point_combi:
                out = torch.cat([linear_out.unsqueeze(0), out[1:]]).mean(dim=0)
            else:
                out = out[-1]
        # Late Merge    
        if not self.early_merge and self.ideal:
            out += ideal_out
        return out