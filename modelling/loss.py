import time

import torch
import numpy as np

def rmsse_torch(pred, gt, historical):
    batch_size, historical_period, num_ts = historical.shape
    device = historical.device

    historical_mask = (historical!=0).long()
    historical_first_valid_index_locator = torch.arange(historical_mask.size(1), 0, -1).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, num_ts).to(device)
    historical_first_valid_index_locator = torch.argmax(historical_first_valid_index_locator * historical_mask, dim=1, keepdim=True)
    denominator_denominator = (historical_period - historical_first_valid_index_locator - 1).squeeze(1)
    sum_diff_historical = torch.sum(torch.diff(historical, dim=1) ** 2, dim=1)

    denominator = sum_diff_historical/denominator_denominator
    denominator[denominator==0] = 1e6
    numerator = torch.mean((pred - gt) ** 2, dim=1)

    metrics_each_batch = (numerator/denominator).mean(-1)
    batch_level_metrics = metrics_each_batch.mean(0)
    return batch_level_metrics

def rmsse_loss(scaler):
    def loss(preds, labels, historical, threshold=None, gate_preds=None):
        if scaler:
            preds = scaler.inverse_transform(preds)
            if gate_preds is not None:
                if threshold is not None:
                    preds = torch.where(gate_preds > threshold, 1, 0) * preds
                else:
                    preds = torch.where(gate_preds > 0, 1, 0) * preds
            labels = scaler.inverse_transform(labels)
            historical = scaler.inverse_transform(historical)
        preds = torch.where(preds < 0, 0, preds)
        rmsse = rmsse_torch(preds, labels, historical)
        return rmsse
    return loss

def mse_torch(preds, labels):
    loss = ((preds-labels)**2).mean()
    return loss

def mse_loss(scaler):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mse = mse_torch(preds=preds, labels=labels)
        return mse
    return loss

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mse_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mse = masked_mse(preds=preds, labels=labels, null_val=null_val)
        return mse
    return loss

def rmse_loss(scaler):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        preds = torch.where(preds < 0, 0, preds)
        rmse = torch.sqrt(mse_torch(preds=preds, labels=labels))
        return rmse
    return loss

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_rmse_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        rmse = masked_rmse(preds=preds, labels=labels, null_val=null_val)
        return rmse
    return loss

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mae_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = masked_mae(preds=preds, labels=labels, null_val=null_val)
        return mae
    return loss

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mape = masked_mape(preds=preds, labels=labels, null_val=null_val)
        return mape
    return loss