import time
import torch
from src.eval import evaluate

def train_model(
    model, 
    trainloader, 
    epochs=10, 
    lr=1e-3,
    optimizer_name="adam"):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    
    if optimizer_name.lower() = "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "seconds": [],
    }
        
    for epoch in range(1, epochs+1):
        t0 = time.time()
        model.train()
        
        correct = 0
        total = 0
        running_loss = 0
        
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)

            #print("Batch x shape:", x.shape)        #smoketest
            #print("Batch y shape:", y.shape)        #smoketest
            #print("Model output shape:", out.shape) #smoketest
        train_loss = running_loss /max(total, 1)
        train_acc = 100.0 * correct /max(total, 1)

        val_loss, val_acc = evaluate(model, valloader, criterion, device)
        sec = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["seconds"].append(sec)

        print(f"Epoch {epoch:02d}/{epochs} | train acc {train_acc:.2f}% loss {train_loss:.4f} | val acc {val_acc:.2f}% loss {val_loss:.4f} | {sec:.1f}s")        

        #break   #smoketest

    return model, history
