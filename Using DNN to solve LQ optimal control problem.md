## Using DNN to solve LQ optimal control problem

A price-taking firm is trying to solve an investment problem with adjustment cost as follows:



$\max_{u_t}  \sum_{t=0}^\infty \beta^t\big[(\alpha_0 - \alpha_1 Y_t)y_t-\frac{\gamma}{2}u_t^2\big]$

$s.t.~ Y_{t+1} = h_0 + h_1 Y_t $

$s.t.~ y_{t+1} = y_t+u_t$

$y_0,Y_0~\text{Given}$



The Euler equation can be written as:

$\gamma u_t = \beta \big[\gamma u_{t+1}+(\alpha_0 -\alpha_1 Y_{t+1})\big]$

where

$Y_{t+1} = h_0 + h_1 Y_t$



The recursive form of the Euler equation can be written as follows:

$\gamma u(Y) = \beta \big[\gamma u(Y')+(\alpha_0 -\alpha_1 Y')\big]$

where

$Y' = h_0 + h_1 Y$​

Here we are looking a function $u$ (a neural net here) that solves the Euler equation. By solving I mean a function that minizes $L_2$ norm of the Euler residuals

$\varepsilon (Y;u)^2 \equiv \bigg(\gamma u(Y)-\beta\big[\gamma u\big(Y'(Y)\big)+ \big(\alpha_0 -\alpha_1 Y'(Y)\big)\big]\bigg)^2 $

over some points of interest in state space $\Gamma(Y) \subset \mathbb{R}$



```python
import torch
import torch.nn as nn
import torch.nn.functional as F
##############DNN####################################################
class NN(nn.Module):
    def __init__(self, 
                 dim_hidden = 128,
                layers = 4,
                hidden_bias = True):
        super().__init__()
        self.dim_hidden= dim_hidden
        self.layers = layers
        self.hidden_bias = hidden_bias
        
        torch.manual_seed(1234)
        module = []
        module.append(nn.Linear(1,self.dim_hidden, bias = self.hidden_bias))
        module.append(nn.ReLU())
        
        for i in range(self.layers-1):
            module.append(nn.Linear(self.dim_hidden,self.dim_hidden, bias = self.hidden_bias))
            module.append(nn.ReLU())  
            
        module.append(nn.Linear(self.dim_hidden,1))
        
        self.u = nn.Sequential(*module)


    def forward(self, x):
        u_out = self.u(x)
        return  u_out
    
    
################DATA################################################
    
 class Data:
    def __init__(self,
                 beta = 0.95,
                 alpha_0 = 1.0,
                 alpha_1 = 2.0,
                 gamma = 90.0,
                 h_0 =  0.03,
                 h_1 = 0.94,
                 time = 64,
                 Y_0 = 0.2,
                 batch_size = 4
                ):
        self.beta = beta
        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1
        self.gamma = gamma
        self.h_0 = h_0
        self.h_1= h_1
        self.time = time
        self.Y_0 = Y_0
        self.batch_size = batch_size
        
        self.Y_t = torch.zeros([self.time])
        self.Y_t[0] = self.Y_0
        for t in range(self.time-1):
            self.Y_t[t+1] = self.h_0 + self.h_1*self.Y_t[t]
            
        self.Y_prime_t = self.h_0 + self.h_1*self.Y_t
        
        self.train_data = torch.stack((self.Y_t,self.Y_prime_t),1)
        

```

DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class Data_loader(Dataset):
    
    def __init__(self,data):
        self.data = data
        self.Y = self.data[:,[0]]
        self.Y_prime = self.data[:,1:]
        self.n_samples = self.data.shape[0]

    def __getitem__(self,index):
            return self.Y[index], self.Y_prime[index] # order: Y first, then Y_prime 
        
    def __len__(self):
        return self.n_samples
    
data_set = Data().train_data
data_label = Data_loader(data = data_set)
batch_size = Data().batch_size
train = DataLoader(dataset = data_label, batch_size = batch_size, shuffle = True)

```

Train

```python
α_0 = Data().alpha_0
α_1 = Data().alpha_1
γ = Data().gamma
β = Data().beta
max_epochs = 1001
u_hat = NN()
learning_rate = 1e-2

optimizer = torch.optim.Adam(u_hat.parameters(), lr=learning_rate, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

for epoch in range(max_epochs):
    for index, (Y,Y_prime) in enumerate(train):
        
        euler_res = γ*u_hat(Y) - β*( γ*u_hat(Y_prime) + α_0 - α_1* Y_prime )
        loss = euler_res.pow(2).mean()
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
    scheduler.step()
    if epoch % 100 == 0:
        #print("epoch:",",",epoch,',',"{:.2e}".format(loss.item()),',',"{:.2e}".format(get_lr(optimizer))) 
        print("epoch:",epoch, ",","MSE Euler Residuals:","{:.2e}".format(loss.item()))    

```

