import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning  as pl
from torch.nn.modules.linear import Linear
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
class Conv4(nn.Module):
    def __init__(self,img_channels=1,hidden_size=256,K=2,latent_size=512):
        super().__init__()
        self.latent_size = latent_size
        self.feature = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.auto_regressive = nn.GRU(latent_size,hidden_size,2)
        # self.W = nn.ModuleList([nn.Linear(hidden_size,latent_size,bias=False) for i in range(K)] )
        self.W = nn.ModuleList([nn.Sequential(nn.Linear(in_features=hidden_size, out_features=hidden_size//2, bias=True),
        nn.Dropout(),
        nn.ReLU(),
        nn.Linear(in_features=hidden_size//2, out_features=latent_size, bias=True)) for i in range(K) ] )     
        self.K= K
        
    def forward(self, x):
        x = self.feature(x)
        return x

class Linear_Layer(nn.Module):
    def __init__(self,input_size = 25088, classes = 10):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(input_size, 1024, bias=True),
        nn.Dropout(),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 512, bias=True),
        nn.Dropout(),
        nn.ReLU(inplace=True),
        nn.Linear(512, classes, bias=True))
    
    def forward(self, x):
        x = self.feature(x)
        return x

class LinClassifier(pl.LightningModule):
    def __init__(self,pretrain_path) -> None:
        super().__init__()
        K=2
        latent_size = 512
        self.feat = Conv4(img_channels=3)
        ckpt = torch.load(pretrain_path)
        self.feat.load_state_dict(ckpt["model"])
        for param in self.feat.parameters():
            param.requires_grad=False
        
        if hasattr(self.feat,"latent_size"):
            self.latent_size = self.feat.latent_size
        else:
            self.latent_size = 512

        self.lin = nn.Sequential(nn.Linear(25088, 1024, bias=True),
        nn.Dropout(),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 512, bias=True),
        nn.Dropout(),
        nn.ReLU(inplace=True),
        nn.Linear(512, 10, bias=True))
        self.val_acc = 0
    
    def forward(self,x):
        # x = torch.unsqueeze(x,-3)
        batch_grid, img_shape = x.shape[:3],x.shape[3:]
        res = self.feat.feature(x.view(-1,*img_shape))
        res = res.view(batch_grid[0],-1)
        x = self.lin(res)
        return x

    def cross_entropy_loss(self,logits,labels):
        return torch.nn.CrossEntropyLoss()(logits,labels)
    def training_step(self,train_batch,batch_idx):
        data,target = train_batch
        logits = self.forward(data)
        loss = self.cross_entropy_loss(logits,target)
        accuracy = self.accuracy(logits, target)
        self.log("train_accuracy", accuracy)
        self.log("train_loss",loss)
        return loss
    def validation_step(self,val_batch,batch_idx):
        data, target = val_batch
        logits = self.forward(data)
        loss = self.cross_entropy_loss(logits,target)
        accuracy = self.accuracy(logits, target)
        self.log("val_loss",loss)
        self.log("val_accuracy",accuracy)
        
        return {"val_loss": loss, "val_accuracy": accuracy}
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        
        accuracy = torch.tensor([x["val_accuracy"] for x in outputs]).mean()
        if self.val_acc <accuracy:
            self.val_acc = accuracy
        return super().validation_epoch_end(outputs)
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(),lr=1e-3)
        return opt
    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)