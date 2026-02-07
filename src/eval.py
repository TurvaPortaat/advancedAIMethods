import torch

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
  model.eval()
  total = 0
  correct = 0
  running_loss = 0

  for x, y in dataloader:
    x, y = x.to(device), y.to(device)
    out = model(x)
    loss = criterion(out, y)

    running_loss += loss.item() * y.size(0)
    _, pred = out.max(1)
    correct += pred.eq(y).sum().item()
    total += y.size(0)

  avg_loss = running_loss / max(total, 1)
  acc = 100.0 * correct / max(total, 1)
  
  return avg_loss, acc
