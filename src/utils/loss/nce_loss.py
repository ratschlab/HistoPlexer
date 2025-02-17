import numpy as np
import torch
from torch import nn

# checked

class PatchNCELoss(nn.Module):
    def __init__(self, 
                batch_size,
                total_step,
                n_step_decay,
                scheduler='lambda',
                lookup='linear',
                nce_includes_all_negatives_from_minibatch=False):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool
        self.batch_size = batch_size
        self.total_step = total_step
        self.n_step_decay = n_step_decay
        self.scheduler = scheduler
        self.lookup = lookup
        self.nce_T = 0.07
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch

    def forward(self, feat_q, feat_k, current_step=-1):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))  # [B*num_patches, 1, C] @ [B*num_patches, C, 1] --> [B*num_patches, 1]
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1)) # [B, num_patches, C] @ [B, C, num_patches] --> [B, num_patches, num_patches]

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        # weight loss based on current step and positive pairs' similarity (https://arxiv.org/abs/2303.06193)
        if current_step != -1:
            # Compute scheduling
            t = (current_step - 1) / self.total_step
            if self.scheduler == 'sigmoid':
                p = 1 / (1 + np.exp((t - 0.5) * 10))
            elif self.scheduler == 'linear':
                p = 1 - t
            elif self.scheduler == 'lambda':
                k = 1 - self.n_step_decay / self.total_step
                m = 1 / (1 - k)
                p = m - m * t if t >= k else 1.0
            elif self.scheduler == 'zero':
                p = 1.0
            else:
                raise ValueError(f"Unrecognized scheduler: {self.scheduler}")
            # Weight lookups
            w0 = 1.0
            x = l_pos.squeeze().detach()
            if self.lookup == 'top':
                x = torch.where(x > 0.0, x, torch.zeros_like(x))
                w1 = torch.sqrt(1 - (x - 1) ** 2)
            elif self.lookup == 'linear':
                w1 = torch.relu(x)
            elif self.lookup == 'bell':
                sigma, mu, sc = 1, 0, 4
                w1 = 1 / (sigma * np.sqrt(2 * torch.pi)) * torch.exp(-((x - 0.5) * sc - mu) ** 2 / (2 * sigma ** 2))
            elif self.lookup == 'uniform':
                w1 = torch.ones_like(x)
            else:
                raise ValueError(f"Unrecognized lookup: {self.lookup}")
            # Apply weights with schedule
            w = p * w0 + (1 - p) * w1
            # Normalize
            w = w / w.sum() * len(w)
            loss *= w
        return loss