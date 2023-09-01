import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
import os
import time
import requests


class LitMobileNet(pl.LightningModule):
    def __init__(self, base_model, num_target_classes, lr=1e-3):
        super().__init__()

        # num of target classes
        self.num_target_classes = num_target_classes

        # loss
        self.loss = nn.CrossEntropyLoss()

        # learning rate for optimizer
        self.lr = lr

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

        # network layer
        # init a pretrained weight from base model
        backbone = base_model
        num_filters = backbone.classifier[0].in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify num_target_classess
        self.classifier = nn.Linear(num_filters, self.num_target_classes)

    def forward(self, x: torch.Tensor):
        '''method used for inference input -> output'''
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        logs = {'train_loss': loss, 'train_accuracy': acc}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        logs = {'val_loss': loss, 'val_accuracy': acc}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # Let's return preds to use it in a custom callback
        return preds

    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        logs = {'test_loss': loss, 'test_accuracy': acc}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

    def configure_optimizers(self):
        '''defines model optimizer'''
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = torchmetrics.accuracy(
            preds, y, "multiclass", num_classes=self.num_target_classes)
        return preds, loss, acc


def download_image(url, save_dir="temp/"):
    try:
        # Ambil konten gambar dari URL
        response = requests.get(url)
        response.raise_for_status()

        # Generate nama file berdasarkan timestamp Unix
        timestamp = int(time.time())
        file_name = f"{timestamp}.jpg"
        file_path = os.path.join(save_dir, file_name)

        # Simpan gambar ke direktori yang diinginkan
        with open(file_path, "wb") as file:
            file.write(response.content)

        print(f"Image downloaded as {file_name}")

        return save_dir+"/"+file_name
    except Exception as e:
        print(f"Failed to download images: {e}")


def predict_image(model, image_input, device):
    # predict data
    model.eval()
    with torch.inference_mode():
        x = image_input.unsqueeze(0).to(device)

        logits = model(x)

        probability = torch.softmax(logits.squeeze(), dim=0)

        return probability
