from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
import time
import random

class TrainingManager():
    def __init__(self,dataloader:DataLoader,num_epochs:int,save_every_n_epochs:int = None):
        self.dataloader = dataloader
        self.dataset_len = len(self.dataloader)

        self.num_epochs = num_epochs
        self.save_every_n_epochs = save_every_n_epochs
        self.epochs = [self.current_epoch + step for step in range(self.num_epochs)]

        self.total_step = self.dataset_len*self.num_epochs
        self.current_iter = 0
        self.current_epoch = 1
        self.epoch_loss = 0
        self.total_loss = 0
        self.avg_loss = 0
        self.progress_bar = tqdm(range(self.total_step),desc=f"Epoch {1}/{num_epochs}")

        self.current_lr = None

        self.log={
            "total_step":self.total_step,
            "epoch_loss":[],
            "epoch_lr":[]
        }

    def step(self,loss,**kwargs):
        self.epoch_loss += loss
        self.current_iter += 1
        self.total_loss += loss
        self.avg_loss = self.total_loss / self.current_iter
        
        self.progress_bar.update(1)
        self.progress_bar.set_postfix(loss=f"{loss:.4f}",**kwargs)

        
    def epoch_step(self):
        avg_epoch_loss = self.epoch_loss/self.dataset_len
        self.log["epoch_loss"].append(avg_epoch_loss)
        self.log["epoch_lr"].append(self.current_lr)
        tqdm.write(f"Epoch {self.current_epoch}/{self.num_epochs} | Epoch Loss: {avg_epoch_loss:.4f}")
        self.current_epoch += 1 
        self.epoch_loss = 0
        if self.current_epoch <= self.num_epochs:
            self.progress_bar.set_description(f"Epoch {self.current_epoch}/{self.num_epochs}")


    
    
    def is_checkpoint(self)->bool:
        if self.current_epoch == self.num_epochs:
            return True
        
        if self.save_every_n_epochs is not None and (self.current_epoch) % self.save_every_n_epochs == 0:
             return True
            
        return False
    
    
class Dataset(Dataset):
        def  __init__(self,repeat):
            self.dataset = [i for i in range(10)]
            self.repeat=repeat

        def __len__(self):
            return len(self.dataset)*self.repeat
        
        def __getitem__(self,idx):
            true_idx = idx%len(self.dataset)
            return self.dataset[true_idx]


def main():
    num_epochs = 10
    save_every_n_epochs = 10

    batch_size = 2
    repeat = 10
    dataset = Dataset(repeat=repeat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    tm = TrainingManager(dataloader,num_epochs,save_every_n_epochs)

    for epoch in tm.epochs:
        for data in tm.dataloader:
            time.sleep(0.01)
            loss = random.randint(0,10)
            tm.step(loss)
            
        if tm.is_checkpoint():
            print("checkponit")

        tm.epoch_step()


if __name__=="__main__":
    main()
        
