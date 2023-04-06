import torch 

class Recht(torch.nn.Module):
    def __init__(self, params=None) -> None:
        super().__init__()
        if params == None:
            self.A = torch.tensor([[1.01, 0.01, 0],[0.01, 1.01, 0.01],[0, 0.01, 1.01]])
            self.B, self.C = torch.eye(3), torch.eye(3)
            self.D = torch.zeros((3,3))
        else:
            self.A, self.B, self.C, self.D = params[0], params[1], params[2], params[3]
            
    def forward(self, state : torch.tensor, action : torch.tensor) -> torch.tensor:
        x = state @ self.A.T + action @ self.B.T
        obs = x @ self.C.T + action @ self.D.T
        return obs