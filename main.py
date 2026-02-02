from torchview import draw_graph
from src.models.inception_style import InceptionCIFAR, Inception_module
from src.data import get_dataloaders
from src.train import train_model

def main():
    trainset, trainloader, valset, valloader, testset, testloader = get_dataloaders(batch_size=64, val_size=5000)

    print(len(trainset),len(valset), len(testset))

    model = InceptionCIFAR()
    train_model(model, trainloader, epochs=1, lr=1e-3)  #lower epochs to 1 during smoketest

    print("Train size:", len(trainset)) #smoketest
    print("Val size:", len(valset))     #smoketest
    print("Test size:", len(testset))   #smoketest

    # VISUALIZATION # Comment during smoketest

    #g = draw_graph(Inception_module(), input_size=(1,192,32,32), save_graph=True, filename="inception_block", directory="runs")

    #print(g.visual_graph)

if __name__ == "__main__":
    main()
