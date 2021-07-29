import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import defaultdict

class ActorCritic(nn.Module):
    def __init__(self, granularity):
        super(ActorCritic, self).__init__()
        self.action_translation_map = self.build_tranlation_map2(granularity)
        self.affine = nn.Linear(8, 128)
        #self.hidden = nn.Linear(256, 128)

        self.action_layer = nn.Linear(128, len(self.action_translation_map))
        self.value_layer = nn.Linear(128, 1)

        self.logprobs = []
        self.state_values = []
        self.rewards = []
        
        

    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = F.relu(self.affine(state))
        #state = F.relu(self.hidden(state))
        
        state_value = self.value_layer(state)

        action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()

        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)

        return action.item()

    def calculateLoss(self, gamma=0.99):

        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)

        # normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std())

        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)
        return loss

    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
        

    def build_tranlation_map(self, granularity=4):
        g = granularity - 1 
        result = {}
        
        helper = defaultdict(list)
        index = 0
        for i, rng in enumerate([(0,1), (-1, -0.5), (0.5, 1)]):
            for j in range(granularity):
                a, b = rng
                delta = abs( (a-b)/g)
                helper[i].append(a + (delta*j))
                
        
        for i in range(granularity): 
            for j in range(1, len(helper)):
                for k in range(granularity):
                    result[index] = (helper[0][i], helper[j][k])
                    index += 1
        
        return result
    
    def build_tranlation_map2(self, granularity=4):
        result= {}
        delta = 1/granularity
        result[0] = (0,0)
        for i in range(1,granularity+1):
             result[i] = ((delta*i)-0.01, 0)
        index=granularity+1     
        for i, rng in enumerate([(-1, -0.5), (0.5, 1)]):
             for j in range(granularity):
                a,b = rng
                delta = abs((a-b)/(granularity))
                result[index] = (0,(a+delta*(j+1))-0.01) 
                index += 1
        
        return result                  
             
                
    def map_state_to_action(self, state):
        return self.action_translation_map[state]
