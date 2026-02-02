import torch

def train_model(model, trainloader, epochs=5, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0

        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)

            print("Batch x shape:", x.shape)        #smoketest
            print("Batch y shape:", y.shape)        #smoketest
            print("Model output shape:", out.shape) #smoketest

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}: train accuracy {acc:.2f}%")

        break   #smoketest

    return model