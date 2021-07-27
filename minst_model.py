import torch
from torch import nn
from torch.nn import functional as F

from torch.nn.modules.linear import Linear
import pytorch_lightning  as pl
class mnist_Conv4(nn.Module):
    def __init__(self,img_channels=1,hidden_size=100,K=2,latent_size=1024):
        super().__init__()
        self.latent_size = latent_size
        self.feature = nn.Sequential(*[
            nn.Conv2d(in_channels=img_channels, out_channels=16, kernel_size=5, stride=2,padding=1),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2,padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.Conv2d(in_channels=128,out_channels=256, kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.Conv2d(in_channels=256,out_channels=512, kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.AdaptiveAvgPool2d(output_size=(1,1))
        ])
        
        self.auto_regressive = nn.GRU(latent_size,hidden_size,1)
        self.W = nn.ModuleList([nn.Linear(hidden_size,latent_size,bias=False) for i in range(K)] )
        self.K= K
        
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.feature(x)
        return x
       
class mnist_LinClassifier(pl.LightningModule):
    def __init__(self,pretrain_path) -> None:
        super().__init__()
        
        self.feat = mnist_Conv4(latent_size=512)
        model_state = {key[len("feature."):]:val for key,val in torch.load(pretrain_path)["model"].items() if "feature" in key}
        
        self.feat.feature.load_state_dict(model_state)
        for param in self.feat.parameters():
            param.requires_grad=False
        
        if hasattr(self.feat,"latent_size"):
            self.latent_size = self.feat.latent_size
        else:
            self.latent_size = 512
        self.lin = Linear(self.latent_size,10)
    
    def forward(self,x):
        x = torch.unsqueeze(x,-3)
        batch_grid, img_shape = x.shape[:3],x.shape[3:]
        res = self.feat.feature(x.view(-1,*img_shape))
        res = res.view(*batch_grid,-1)
        res = torch.mean(res,dim=[1,2])
        x = self.lin(res)
        return F.log_softmax(x,1)

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
        
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(),lr=5e-3)
        return opt
        
    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)
