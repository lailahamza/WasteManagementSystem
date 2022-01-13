import os
import torch
import io
import json
import torchvision

import torchvision.models as tvmodels
import torchvision.transforms as totransforms
import torch.nn as tn
import torch.nn.functional as Ftn

from PIL import Image
from torchvision import models
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid


# load model
class ClassificationImageBase(tn.Module):

    def phase_training(self, batch):
        images, etiquettes = batch
        pred = self(images)  # Genère les prédictions
        perte = Ftn.cross_entropy(pred, etiquettes)  # Calcule la perte
        return perte

    def phase_validation(self, batch):
        images, etiquettes = batch
        pred = self(images)  # Genère les prédictions
        perte = Ftn.cross_entropy(pred, etiquettes)  # Calcule la perte
        acc = accur(pred, etiquettes)  # Calcule l'accuracy
        return {'perte_val': perte.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['perte_val'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'perte_val': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, data_directory):
        print("Epoch {}: perte_train: {:.4f}, perte_val: {:.4f}, val_acc: {:.4f}".format(
            epoch + 1, data_directory['perte_train'], data_directory['perte_val'], data_directory['val_acc']))


class ResNet(ClassificationImageBase):
    def __init__(self):
        super().__init__()

        # utiliser un modèle pré-entrainé
        self.network = tvmodels.resnet50(pretrained=True)

        # Remplacer la dernière couche
        features_numero = self.network.fc.in_features
        self.network.fc = tn.Linear(features_numero, len(data_directory.classes))

    def forward(self, xvar):
        return torch.sigmoid(self.network(xvar))


def to_device(data, device):
    """Déplacer Tensors vers l'appareil choisi"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# Hyper-parameters
b_size = 32  # Batch Size
num_epochs = 8
lr = 5.5e-5
mod = ResNet()
PATH = "app/wastemanage_ffn.pth"
mod.load_state_dict(torch.load(PATH))
mod.eval()


# image -> tensor
def image_transformation(image_bytes):
    transform = totransforms.Compose([totransforms.Grayscale(num_output_channels=1),
                                      totransforms.Resize((28, 28)),
                                      totransforms.ToTensor(),
                                      totransforms.Normalize((0.1307,), (0.3081,))])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)


# predict
def predict_image(image):
    var = image.reshape(-1, 28 * 28)
    yvar = mod(var)
    # Choisir l'indice avec la probabilité la plus élevée
    prob, predicted = torch.max(yvar.data, 1)
    return predicted
