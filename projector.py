import torch
from torch import nn



 

class Proj(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.p =  nn.Linear(dim,dim)
            
       
       
             	   
    def forward(self, x):
        u, v = x, x 
        u = self.p(u)
                                  
        g = u * v
        
      
        return g



class ProjectorBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
       
        self.norm = nn.LayerNorm(d_model)       
        self.proj = Proj(d_model)
      
    def forward(self, x):
        residual = x
        x = self.norm(x)
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








