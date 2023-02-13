from aiims_dataset import CTScanDatasetModule
import torch
import pytorch_lightning as pl
from models.modelUnet_v1 import UNet
import torchmetrics
import torch.nn as nn
import torch.optim as optim

class CTScanModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet(1,7)#in_channels=1, out_channels=7)
        self.loss_module = nn.CrossEntropyLoss(weight=torch.tensor([0.3/7] + [0.1] + [6/35]*5))
        self.MIoU = torchmetrics.JaccardIndex(task='multiclass', num_classes=7)


    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.03)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, masks)
        acc = self.MIoU(preds, masks)

        self.log("train_MIoU", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        preds = self.model(imgs).argmax(dim=-3)
        acc = self.MIoU(preds, masks)
        self.log("val_acc", acc)




torch.set_float32_matmul_precision('medium')
data_module = CTScanDatasetModule()

model = CTScanModule()
trainer = pl.Trainer(log_every_n_steps=1,accelerator='gpu', devices=1, max_epochs=100)

trainer.fit(model, data_module) 





