import random
class FeatureBank:
    def __init__(self,max_len=1000) -> None:
        self.max_len = max_len
        self.data = []
        self.neg_per_pic = 2
    def append(self,feat_vec,batch_idx):
        self.data.append({"batch_idx":batch_idx,"feat_vec":feat_vec})
        if len(self.data)>self.max_len:
            self.russian_roulette()
    def get_samples(self,num_batches,batch_idx):
        def get_diff_batch(num_batch):
            for _ in range(num_batch):
                batch = random.choice(self.data)
                while batch["batch_idx"] == batch_idx:
                    batch = random.choice(self.data)
                yield batch["feat_vec"]
        return [self.data[0]["feat_vec"]] if len(self.data) ==1 else list(get_diff_batch(num_batches))
        
    def russian_roulette(self):
        del self.data[0]
        # unlucky_idx = random.rand(0,len(self.data))
        # del self.data[unlucky_idx]
