import torch
from torch.nn import functional as F
from torch import nn


def train_single_epoch(epoch,
                       model,
                       train_loader,
                       optimizer,
                       device,
                       loss_instance=None,
                       **kwargs):
    log_interval = 10
    model.train()
    train_loss = 0
    num_batch = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        labels = labels.to(device)
        logits = model(data)
        loss = loss_instance(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        train_loss += loss.item()
        optimizer.step()
        num_batch += 1

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / num_batch))
    return train_loss / num_batch


def test_single_epoch(epoch,
                      model,
                      test_val_loader,
                      device,
                      loss_instance=None,
                      num_labels=None,
                      **kwargs):
    model.eval()
    loss = 0
    num_batch = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_val_loader):
            data = data.to(device)
            labels = labels.to(device)
            logits = model(data)

            loss += loss_instance(
                logits,
                labels,
            ).item()
            num_batch += 1

    print('======> Test set loss: {:.4f}'.format(
        loss / num_batch))
    return loss / num_batch
