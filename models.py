import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.nn.modules.linear import Linear
import pytorch_lightning  as pl

class Conv4Suggested(nn.Module):
    def __init__(self,img_channels=1,hidden_size=100,K=2,latent_size=1024):
        super().__init__()
        self.latent_size = latent_size
        
        self.feature = nn.Sequential(*[
            nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(4,4))
        ])
        
        # self.
        
        self.auto_regressive = nn.GRU(latent_size,hidden_size,1)
        self.W = nn.ModuleList([nn.Linear(hidden_size,latent_size,bias=False) for i in range(K)] )
        
        self.K= K
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.feature(x)
        return x
    
class Conv4(nn.Module):
    def __init__(self,img_channels=1,hidden_size=100,K=2,latent_size=1024):
        super().__init__()
        self.latent_size = latent_size
        self.feature = nn.Sequential(*[
            nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=8, kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(4,4))
        ])
        # self.feature = nn.Sequential(*[
        #     nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=3, stride=1,padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2,padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3, stride=1,padding=1),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d(output_size=(4,4))
        # ])
        
        # self.
        
        self.auto_regressive = nn.GRU(latent_size,hidden_size,1)
        self.W = nn.ModuleList([nn.Linear(hidden_size,latent_size,bias=False) for i in range(K)] )
        
        self.K= K
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.feature(x)
        return x
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # # x = self.dropout2(x)
        # x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        # return output
class LinClassifier(pl.LightningModule):
    def __init__(self,pretrain_path,no_mean=False,grid_shape=None) -> None:
        super().__init__()
        assert  not (no_mean and grid_shape is None), "grid shape can not be None for no_mean=True"
        self.no_mean = no_mean
        self.feat = Conv4(latent_size=128)
        model_state = {key[len("feature."):]:val for key,val in torch.load(pretrain_path)["model"].items() if "feature" in key}
        
        self.feat.feature.load_state_dict(model_state)
        for param in self.feat.parameters():
            param.requires_grad=False
        
        if hasattr(self.feat,"latent_size"):
            self.latent_size = self.feat.latent_size
        else:
            self.latent_size = 128
        if no_mean:
            self.lin = Linear(self.latent_size*grid_shape[0]*grid_shape[1],10)
        else:
            self.lin = Linear(self.latent_size,10)
        self.val_acc = 0
    
    def forward(self,x):
        x = torch.unsqueeze(x,-3)
        batch_grid, img_shape = x.shape[:3],x.shape[3:]
        res = self.feat.feature(x.view(-1,*img_shape))
        res = res.view(*batch_grid,-1)
        if self.no_mean:
            res = res.view(batch_grid[0],-1)
        else:
            res = torch.mean(res,dim=[1,2])
        x = self.lin(res)
        x = F.sigmoid(x)

        return F.log_softmax(x,1)
    def cross_entropy_loss(self,logits,labels):
        return F.nll_loss(logits,labels)
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
