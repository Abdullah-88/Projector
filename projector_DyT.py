import torch
from torch import nn

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5):
    
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value      
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
       

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x
       

class Proj(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.p =  nn.Linear(dim,dim,bias = False)
            
       
       
             	   
    def forward(self, x):
        u, v = x, x 
        u = self.p(u)
                                  
        g = u * v
        
      
        return g



class ProjectorBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
       
        self.dyt = DynamicTanh(d_model)       
        self.proj = Proj(d_model)
      
    def forward(self, x):
        residual = x
        x = self.dyt(x)
        x = self.proj(x)           
        x = x + residual            
        out = x
        return out



class Projector(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()
        
        self.model = nn.Sequential(
            *[ProjectorBlock(d_model) for _ in range(num_layers)]
        )

    def forward(self, x):
       
        return self.model(x)








