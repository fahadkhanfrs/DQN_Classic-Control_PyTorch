import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, action_dim)
#init defines the layers
#and forward does the calculations
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Initialize the DQN network
    state_dim = 12
    action_dim = 2
    model = DQN(state_dim, action_dim)
    
    # Example forward pass
    sample_state = torch.randn(10, state_dim)
    q_values = model(sample_state)
    print(f"Q-values: {q_values}")