from model import ActorCritic
import torch
import gym
from PIL import Image

def test(n_episodes=5, name='LunarLander_0.02_0.9_0.999.pth'):
    env = gym.make('LunarLanderContinuous-v2')
    policy = ActorCritic(3)
    
    policy.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = True
    save_gif = False

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            raw_action = policy(state)
            action = policy.map_state_to_action(raw_action)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
                 if save_gif:
                     img = env.render(mode = 'rgb_array')
                     img = Image.fromarray(img)
                     img.save('./gif/{}.jpg'.format(t))
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
            
if __name__ == '__main__':
    test()
