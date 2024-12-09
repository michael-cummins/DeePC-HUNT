import torch
import torch.nn as nn 
import torch.optim as optim
from tqdm import tqdm
from deepc_hunt.utils import sample_initial_signal, episode_loss, Projection
from typing import Dict

class Trainer:

    def __init__(self, controller : nn.Module, env : nn.Module) -> None:
        self.controller = controller
        self.env = env
        self.opt = optim.Rprop(self.controller.parameters(), lr=0.01, step_sizes=(1e-3,1e2))
        # Box constraints for numerical stability
        self.projection = Projection(lower=1e-5, upper=1e5)

    def run(self, epochs: int, time_steps: int, uref=None, yref=None) -> Dict[str, torch.Tensor]:

        pbar = tqdm(range(epochs), ncols=100)
        
        # If uref and yref haven't beend passed, assume 0
        if uref is None: 
            uref = torch.zeros(self.controller.m)
            uref = uref.repeat(self.controller.n_batch, self.controller.N)
        if yref is None: 
            yref = torch.zeros(self.controller.p)
            yref = yref.repeat(self.controller.n_batch, self.controller.N)

        for _ in pbar:
            
            # Get random initial signal from data
            u_ini, y_ini = sample_initial_signal(
                Tini=self.controller.Tini, 
                m=self.controller.m, p=self.controller.p, 
                batch=self.controller.n_batch, 
                ud=self.controller.ud, 
                yd=self.controller.yd
            )
            u_ini = u_ini.to(self.controller.device)
            y_ini = y_ini.to(self.controller.device)
            uT, yT = u_ini, y_ini

            # For collecting closed-loop cost
            Y = torch.Tensor().to(self.controller.device)
            U = torch.Tensor().to(self.controller.device)

            # Begin simulation 
            for _ in range(time_steps):
                
                # Solve for input
                decision_vars = self.controller(uref=uref, yref=yref, u_ini=u_ini, y_ini=y_ini)
                u_pred = decision_vars[0]
                action = u_pred[:,:self.controller.m]
            
                # Apply input to surrogate model
                obs = self.env(y_ini[:,-self.controller.p:], action)

                # Collect closed-loop cost
                real_y = yref[:,:self.controller.p].unsqueeze(1)
                real_u = uref[:,:self.controller.m].unsqueeze(1)
                Y = torch.cat((Y, obs.unsqueeze(1) - real_y.to(self.controller.device)), axis=1)
                U = torch.cat((U, action.unsqueeze(1) - real_u.to(self.controller.device)), axis=1)

                # Update initial condition
                uT = torch.cat((uT, action), 1)
                yT = torch.cat((yT, obs), 1)
                y_ini = yT[:,-self.controller.p*self.controller.Tini:]
                u_ini = uT[:,-self.controller.m*self.controller.Tini:]
            
            # Compute loss and take gradient step
            loss = episode_loss(Y=Y, U=U, controller=self.controller)
            self.opt.zero_grad()
            loss.backward(retain_graph=True)
            self.opt.step()
            self.controller.apply(self.projection)
            
            description = ''
            for name, param in self.controller.named_parameters():
                description += f'{name} : {param.data.item():.3f}, '
            pbar.set_description(description)
        
        for name, param in self.controller.named_parameters():
            print(f'Name : {name}, Value : {param.data}')

        return {k: param for k, param in self.controller.named_parameters()}