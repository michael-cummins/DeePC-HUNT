import torch
import torch.nn as nn 
import torch.optim as optim
from tqdm import tqdm
from deepc_hunt.utils import sample_initial_signal, episode_loss, Projection

class Trainer:

    def __init__(self, controller : nn.Module, env : nn.Module) -> None:
        self.controller = controller
        self.env = env
        self.opt = optim.Rprop(self.controller.parameters(), lr=0.01, step_sizes=(1e-3,1e2))
        self.projection = Projection(lower=1e-5, upper=1e5)

    def run(self, epochs, time_steps):
        pbar = tqdm(range(epochs), ncols=100)
        
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

            I, PI = self.controller.get_PI()

            G = torch.Tensor().to(self.controller.device)
            Ey = torch.Tensor().to(self.controller.device) if self.controller.stochastic_y else None
            Eu = torch.Tensor().to(self.controller.device) if self.controller.stochastic_u else None
            Y = torch.Tensor().to(self.controller.device)
            U = torch.Tensor().to(self.controller.device)

            # Begin simulation 
            for _ in range(time_steps):
                
                # Solve for input
                decision_vars = self.controller(ref=None, uref=None, u_ini=u_ini, y_ini=y_ini)
                [g, u_pred] = decision_vars[:2]
                
                sig_y = decision_vars[3] if Ey is not None else None
                sig_u = decision_vars[4] if Eu is not None else None

                # Apply input to simulation
                action = u_pred[:,:self.controller.m] 
                obs = self.env.forward(y_ini[:,-self.controller.p:], action)

                G = torch.cat((G, g.unsqueeze(1)), axis=1)
                U = torch.cat((U, action.unsqueeze(1)), axis=1)
                Y = torch.cat((Y, obs.unsqueeze(1)), axis=1)
                
                if Eu is not None:
                    Eu = torch.cat((Eu, sig_u.unsqueeze(1)), axis=1)
                if Ey is not None:
                    Ey = torch.cat((Ey, sig_y.unsqueeze(1)), axis=1)

                # Update initial condition
                uT = torch.cat((uT, action), 1)
                yT = torch.cat((yT, obs), 1)
                y_ini = yT[:,-self.controller.p*self.controller.Tini:]
                u_ini = uT[:,-self.controller.m*self.controller.Tini:]
            
            loss = episode_loss(Y=Y, G=G, U=U, Ey=Ey, Eu=Eu, controller=self.controller, PI=(I-PI))
            self.opt.zero_grad()
            loss.backward(retain_graph=True)
            self.opt.step()
            self.controller.apply(self.projection)
            
            description = f'Loss = {loss.item():.4f}, '
            for name, param in self.controller.named_parameters():
                description += f'{name} : {param.data.item():.3f}, '
            pbar.set_description(description)
        
        for name, param in self.controller.named_parameters():
            print(f'Name : {name}, Value : {param.data}')

        return [param for param in self.controller.parameters()]