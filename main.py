from torchview import draw_graph
from src.data import get_dataloaders
from src.train import train_model
from src.eval import evaluate
import json
import time
import argparse
import torch
from src.models.custom_inception_residual import CustomInceptionResidualNet
from src.models.custom_residual import CustomResidualNet
from src.models.small_residual import SmallResidualNet
from src.models.resnet18_pretrained import ResNet18Pretrained
from src.models.inception_style import InceptionCIFAR, Inception_module

MODE = "train" # "smoke" or "train"

def get_model(name: str):
    name = name.lower()
    if name == "inception":
        return InceptionCIFAR()
    if name == "small_residual":
        return SmallResidualNet()
    if name == "custom_residual":
        return CustomResidualNet()
    if name == "resnet18":
        return ResNet18Pretrained(pretrained=True)
    if name == "custom_inception_residual":
        return CustomInceptionResidualNet()
    raise ValueError(f"Unknown model name: {name}")

def append_results(path, record):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    data.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "train", "viz"], default="smoke")
    parser.add_argument("--model", choices=["inception","small_residual","custom_residual","resnet18","custom_inception_residual"], default="inception")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam","sgd"], default="adam")
    parser.add_argument("--val_size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--results", default="results.json")
    args = parser.parse_args()
    
    trainset, trainloader, valset, valloader, testset, testloader = get_dataloaders(
        batch_size = args.batch, 
        val_size = args.val_size,
        seed = args.seed,
        num_workers = args.num_workers,
        root="./data"
    )

    print("Sizes: ", len(trainset), len(valset), len(testset))

    # quick forward sanity
    x = torch.randn(2,3,32,32)
    m = get_model(args.model)
    y = m(x)
    print("Forward OK: ", args.model, y.shape)

    if args.mode == "smoke":
        return
        
    if args.mode == "viz":
        from torchview import draw_graph
        g = draw_graph(get_model(args.model), input_size(1,3,32,32), expand_nested= True,
                       save_graph = True, filename=f"{args.model}_graph", directory="runs")
        print(g.visual_graph)
        return
        
        #for m in [
        #   InceptionCIFAR(),
        #   SmallResidualNet(),
        #   CustomResidualNet(),
        #   ResNet18Pretrained(pretrained=False),
        #   CustomInceptionResidualNet(),
        #]:
        #   y = m(x)
        #   print(type(m).__name__, y.shape)
        #   print("SMOKE TEST DONE – no training executed")

    if MODE == "train":

    #   model = CustomInceptionResidualNet()    # OR ResNetPretrained(), OR InceptionCIFAR(), OR ETC.
    #   train_model(model, trainloader, epochs=1, lr=1e-3)  #lower epochs to 1 during smoketest

    #   print("Train size:", len(trainset)) #smoketest
    #   print("Val size:", len(valset))     #smoketest
    #   print("Test size:", len(testset))   #smoketest
    
    # TRAIN
    model = get_model(args.model)
    t0 = time.time()
    model, history = train_model(
        model,
        trainloader = trainloader,
        valloader = valloader,
        epochs = args.epochs,
        lr = args.lr,
        optimizer_name = args.optimizer
    )
    train_seconds = time.time() - t0

    # TEST EVAL (final evaluation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(device)
    testl_loss, test_acc = evaluate(model, testloader, criterion, device)

     record = {
        "model": args.model,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "val_size": args.val_size,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "train_seconds_total": train_seconds,
        "history": history,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "augmentations": "RandomHorizontalFlip + RandomCrop(32,pad=4) on train; val/test ToTensor only",
    }
    append_results(args.results, record)
    print(f"Saved results -> {args.results}")
    print(f"TEST acc: {test_acc:.2f}% loss {test_loss:.4f}")

if __name__ == "__main__":
    main()
