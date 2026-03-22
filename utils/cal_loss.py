import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
import numpy as np
from loguru import logger
import os

EMBED_PAD_NUM = -10000


def materialize_knn_prob(batch, logits, knn_prob):
    if knn_prob is not None:
        return knn_prob.reshape(-1, knn_prob.size(-1)) if knn_prob.dim() > 2 else knn_prob

    total_rows = batch.get("knn_num_rows")
    if total_rows is None:
        raise ValueError("Batch must contain either dense knn_probs or sparse KNN tensors.")

    if torch.is_tensor(total_rows):
        total_rows = int(total_rows.item())
    else:
        total_rows = int(total_rows)

    dense_knn_prob = torch.zeros(
        (total_rows, logits.size(-1)),
        dtype=torch.float32,
        device=logits.device,
    )

    row_ids = batch["knn_row_ids"]
    if row_ids.numel() == 0:
        return dense_knn_prob

    dense_knn_prob[row_ids.long(), batch["knn_col_ids"].long()] = batch["knn_values"].to(dtype=torch.float32)
    return dense_knn_prob

def interpolate(knn_log_probs, lm_log_probs, lmbda=0.25):
    interpolated = torch.logaddexp(
        lm_log_probs + np.log(1 - lmbda), 
        knn_log_probs + np.log(lmbda))

    return interpolated

def kl_loss_evaluate(logits, batch, tokenizer, args, knn_label, knn_prob):
    label_probs = materialize_knn_prob(batch, logits, knn_prob)
    
    shift_logits = logits[:, :-1].contiguous() # (batch, seq_len-1, vocab_size)
    shift_labels = batch['labels'][:, 1:].contiguous() # (batch, seq_len-1)
    
    nonpad_mask = shift_labels != -100
    shift_logits = shift_logits[nonpad_mask] # (nonpad b*t, vocab_size)
    shift_labels = shift_labels[nonpad_mask] # (nonpad b*t)
    label_probs = label_probs / (label_probs.sum(dim=-1, keepdim=True) + 1e-10) # Normalize label_probs
    
    # Ensure that the dimensions match
    assert shift_logits.shape == label_probs.shape, f"shift_logits.shape = {shift_logits.shape}, label_probs.shape = {label_probs.shape}"
    assert torch.all(shift_labels == knn_label), f"shift_labels and knn_label mismatch"
    assert torch.allclose(label_probs.sum(dim=-1), torch.ones_like(label_probs.sum(dim=-1))), f"label_probs does not sum to 1"
    
    # Compute the label_probs
    shift_probs = F.softmax(shift_logits, dim=-1)

    # Calculate PPL
    label_log_probs = label_probs.log()
    label_log_probs = torch.nan_to_num(label_log_probs, nan=None, neginf=-10000.0)
    lm_log_probs = F.log_softmax(shift_logits, dim=-1)
    interpolate_log_probs = interpolate(label_log_probs, lm_log_probs, lmbda=args.lmbda)
    nll_loss = F.nll_loss(interpolate_log_probs, shift_labels, reduction='sum')
    lm_loss = F.nll_loss(lm_log_probs, shift_labels, reduction='sum')
    token_num = shift_labels.shape[0]
    
    return nll_loss, lm_loss, token_num

def kl_loss_token(logits, batch, tokenizer, args, knn_label, knn_prob, alpha=0.5):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch['labels'][:, 1:].contiguous()

    # knn_label and knn_prob are 1D/2D concatenated from collate_fn
    # They contain only valid (non-pad) tokens, already aligned with shift_labels valid tokens
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)

    nonpad_mask = flat_labels != -100

    if not nonpad_mask.any():
        return None, None, None

    # Apply nonpad mask to get valid tokens from model outputs
    logits_f = flat_logits[nonpad_mask]
    labels_f = flat_labels[nonpad_mask]

    # KNN data is already filtered to valid tokens only (1D concatenated)
    flat_knn_label = knn_label.reshape(-1) if knn_label.dim() > 1 else knn_label
    flat_knn_prob = materialize_knn_prob(batch, logits, knn_prob)

    # Trim to matching length (safety net for any off-by-one)
    n = min(labels_f.size(0), flat_knn_label.size(0))
    labels_f = labels_f[:n]
    logits_f = logits_f[:n]
    knn_label_f = flat_knn_label[:n]
    knn_prob_f = flat_knn_prob[:n]

    # Verify alignment
    if not torch.all(labels_f == knn_label_f):
        mismatch_count = torch.sum(labels_f != knn_label_f).item()
        logger.error(f"Label mismatch: {mismatch_count}/{n} labels differ")
        logger.error(f"First 10 labels_f:    {labels_f[:10].tolist()}")
        logger.error(f"First 10 knn_label_f: {knn_label_f[:10].tolist()}")
        raise ValueError(f"Labels must match for correct training. {mismatch_count}/{n} mismatches found.")

    # Normalize the KNN probabilities
    knn_prob_f = knn_prob_f / (knn_prob_f.sum(dim=-1, keepdim=True) + 1e-10)

    kl_loss = F.kl_div(
        F.log_softmax(logits_f, dim=-1),
        knn_prob_f,
        reduction='batchmean'
    )
    
    loss_fct = nn.CrossEntropyLoss()
    lm_loss = loss_fct(logits_f, labels_f)
    
    total_loss = alpha * kl_loss + (1 - alpha) * lm_loss
    logger.debug(f"KL loss: {kl_loss} LM loss: {lm_loss} Total loss: {total_loss}")
    return total_loss, kl_loss, lm_loss
