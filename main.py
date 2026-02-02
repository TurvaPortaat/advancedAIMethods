from torchview import draw_graph
from src.data import get_dataloaders
from src.train import train_model
import torch
from src.models.custom_inception_residual import CustomInceptionResidualNet
from src.models.custom_residual import CustomResidualNet
from src.models.small_residual import SmallResidualNet
from src.models.resnet18_pretrained import ResNet18Pretrained
from src.models.inception_style import InceptionCIFAR, Inception_module

MODE = "smoke" # "smoke" or "train"

def main():
    trainset, trainloader, valset, valloader, testset, testloader = get_dataloaders(batch_size=64, val_size=5000)

    print(len(trainset),len(valset), len(testset))

    if MODE == "smoke":
        x = torch.randn(2, 3, 32, 32)

        for m in [
            InceptionCIFAR(),
            SmallResidualNet(),
            CustomResidualNet(),
            ResNet18Pretrained(pretrained=False),
            CustomInceptionResidualNet(),
        ]:
            y = m(x)
            print(type(m).__name__, y.shape)
            print("SMOKE TEST DONE – no training executed")

    if MODE == "train":

        model = CustomInceptionResidualNet()    # OR ResNetPretrained(), OR InceptionCIFAR(), OR ETC.
        train_model(model, trainloader, epochs=1, lr=1e-3)  #lower epochs to 1 during smoketest

        print("Train size:", len(trainset)) #smoketest
        print("Val size:", len(valset))     #smoketest
        print("Test size:", len(testset))   #smoketest

        # VISUALIZATION

        #g = draw_graph(Inception_module(), input_size=(1,192,32,32), save_graph=True, filename="inception_block", directory="runs")

        #print(g.visual_graph)

if __name__ == "__main__":
    main()
