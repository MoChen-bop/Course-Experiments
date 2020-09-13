import torch
import torch.nn.functional as F 

__all__ = ['Fvt', 'modality_loss', 'joint_embedding_loss']


def Fvt(x, y):
    return torch.matmul(x, y.transpose(-2, -1))


def modality_loss(comp: torch.Tensor, dim: int, batched=False, device='cuda:0'):
    batched = int(batched)
    Dy = 1 - torch.eye(comp.size(1), device=device)
    comp_diff = comp - comp.diagonal(0, batched, batched + 1).unsqueeze(dim + batched)
    return F.relu(Dy + comp_diff).mean(dim=-1).mean(dim=-1)


def joint_embedding_loss(im_enc, txt_enc, _lbls, batched=False, device='cuda:0'):
    assert im_enc.size() == txt_enc.size()

    comp = Fvt(im_enc, txt_enc)

    loss = modality_loss(comp, 0, batched, device) + \
        modality_loss(comp, 1, batched, device)

    return loss
