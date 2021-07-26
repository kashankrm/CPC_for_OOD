import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
from torch.nn.modules.linear import Linear
import pytorch_lightning  as pl
class Conv4(nn.Module):
    def __init__(self, name="resnet18",hidden_size=100,K=2):
        super().__init__()
        if name == "resnet18":
            self.latent_size = 512
            self.feature = models.resnet18()
        elif name == "resnet50":
            self.latent_size = 2048
            self.feature = models.resnet50()
        self.feature = torch.nn.Sequential(*(list(self.feature.children())[:-1]))
        self.auto_regressive = nn.GRU(self.latent_size, hidden_size, 3)
        self.W = nn.ModuleList([nn.Linear(hidden_size,self.latent_size, bias=True) for i in range(K)] )
        self.K= K

    def forward(self, x):
        x = self.feature(x)
        return x
       
class LinClassifier(pl.LightningModule):
    def __init__(self,pretrain_path, n_classes = 10) -> None:
        super().__init__()
        ckpt = torch.load(pretrain_path)
        if "hidden_size" in ckpt:
            hidden_size = ckpt["hidden_size"]
            gru = ckpt["gru"]
        else:
            hidden_size = 100
            gru = 1
        self.feat = Conv4(hidden_size = hidden_size)
        new_state_dict = {}
        for k, v in ckpt["model"].items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        self.feat.load_state_dict(state_dict, strict= False)
        self.feat.feature = torch.nn.DataParallel(self.feat.feature)
        self.feat.auto_regressive = torch.nn.DataParallel(self.feat.auto_regressive)
        for param in self.feat.parameters():
            param.requires_grad=False
        if hasattr(self.feat,"latent_size"):
            self.latent_size = self.feat.latent_size

        self.lin = Linear(self.latent_size * 7 * 7,n_classes)
        self.lin = torch.nn.DataParallel(self.lin)

    def forward(self,x):
        batch_grid, img_shape = x.shape[:3],x.shape[3:]
        res = self.feat.feature(x.view(-1,*img_shape))
        res = res.view(batch_grid[0],-1)
        x = self.lin(res)
        return  F.log_softmax(x,1)

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
        opt = torch.optim.Adam(self.parameters(), lr = 1e-4)
        return opt
        
    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)
