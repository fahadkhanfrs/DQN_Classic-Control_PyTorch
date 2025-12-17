import torch
from torch import nn
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import os

DATE_FORMAT = "%m-%d_%H-%M-%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# Generate plots and save them to files
matplotlib.use('Agg')

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"  # Force CPU for compatibility

#DQN Agent

class Agent:
    
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set
        
        # Hyperparameters
        self.env_id = hyperparameters['env_id']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_params', {}) # Get optional env params

        # Neural Network
        self.loss_fn = nn.MSELoss() 
        self.optimizer = None # Will be initialized later with policy DQN parameters

        # Path to run info
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.png")

    def run(self, is_training = True, render = False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"Training started at {start_time.strftime(DATE_FORMAT)}\n"
            print(log_message)
            with open(self.LOG_FILE, "w") as file:
                file.write(log_message + "\n")
                           
        # Create instance of the environment
        # Using "self.env_make_params" to pass optional parameters
        env = gymnasium.make(self.env_id, render_mode="human" if render else None, **self.env_make_params)
        
        # Observation space dimensions
        num_states = env.observation_space.shape[0] # Expecting type: Box(low, high, (shape0,), float64)
        # Number of possible actions
        num_actions = env.action_space.n

        rewards_per_episode = []

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            
            # Target DQN initialization, which will be periodically updated with policy DQN weights
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict()) # Copies weights and biases from policy to target
            
            step_count = 0
            
            #Policy network optimizer, "Adam" optimizer can be swapped with others like SGD
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            epsilon_history = []

            # Track best reward
            best_reward = -9999999
        
        else:
            # Load learned policy
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            # Set DQN to evaluation mode
            policy_dqn.eval()

        # Train indefinitely
        for episode in itertools.count():

            states, _ = env.reset()
            states = torch.tensor(states, dtype=torch.float, device=device)

            terminated = False # True when agent reaches goal or fails
            episode_reward = 0.0
            
            # Perform actions until episode terminates or reaches max reward
            while (not terminated and episode_reward < self.stop_on_reward):

                # Select action using epsilon-greedy
                if is_training and random.random() < epsilon:
                    # Random action
                    action = env.action_space.sample()
                    action = torch.tensor(action, device=device)
                else:
                    # Greedy action
                    with torch.no_grad():
                        # tensor([1, 2, 3]) -> tensor([[1, 2, 3]])
                        # argmax finds the index of the maximum value along a dimension
                        # policy_dqn returns tensor([[q_value1, q_value2]]), so squeeze to remove extra dim
                        action = policy_dqn(states.unsqueeze(dim=0)).squeeze().argmax() #greater value action of the two

                # Execute action in environment. Truncated and info are not used.
                new_state, reward, terminated, truncated, info = env.step(action.item())
                
                # Accumulated reward
                episode_reward += reward

                # Convert new_state and reward to tensor
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    # Save experience in memory
                    memory.append((states, new_state, action, reward, terminated))
                    # Increment step count
                    step_count += 1

                # Move to new states
                states = new_state

            # Keep track of rewards per episode
            rewards_per_episode.append(episode_reward)

            # Save best model
            if is_training:

                if episode_reward > best_reward:
                
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)

                    with open(self.LOG_FILE, "a") as file:
                        file.write(log_message + "\n")

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                # Update graph every x seconds
                current_time = datetime.now()
                if (current_time - last_graph_update_time) > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # If enough experience has been collected, start training
                if len(memory) > self.mini_batch_size:

                    # Sample mini-batch from memory
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Copy policy DQN to target DQN
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

    def save_graph(self, rewards_per_episode, epsilon_history):

            # Save plots
            fig = plt.figure(1)

            # Average rewards vs episodes
            mean_rewards = np.zeros(len(rewards_per_episode))
            for x in range(len(mean_rewards)):
                mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):x+1]) # Moving average over last 100 episodes
            plt.subplot(121) # 1 row, 2 columns, first plot
            plt.xlabel("Episode")
            plt.ylabel("Average Reward (last 100 eps)")
            plt.plot(mean_rewards)

            # Plot epsilon decay vs episodes
            plt.subplot(122) # 1 row, 2 columns, second plot
            plt.xlabel("Time Step")
            plt.ylabel("Epsilon Decay")
            plt.plot(epsilon_history)

            plt.subplots_adjust(wspace=1.0, hspace=1.0)

            # Save the figure
            plt.savefig(self.GRAPH_FILE)
            plt.close(fig)
    
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        #Transpose the list of experiences and separate them
        states, new_states, actions, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([tensor1, tensor2, tensor3]) -> tensor([[...], [...], [...]])
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)
        new_states = torch.stack(new_states).float().to(device)

        with torch.no_grad():
            # Compute target Q-values (expected Q-values)
            target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
                                 # if termination = 1, then everything becomes zero from up here

        # Compute current Q-values using policy DQN
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Compute loss for the whole mini-batch
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad() # Clears the gradients of all optimized tensors
        loss.backward()            # Backpropagates the loss (computes gradients towards vertex)
        self.optimizer.step()      # Update network parameters, i.e. weights and biases
       
if __name__ == "__main__":

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True) # training mode
    else:
        dql.run(is_training=False, render=True) # test mode with rendering