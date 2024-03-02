import gym
import env.envs


env = gym.make('Chess-v1')
done = False
while not done:
    env.render()
    action = env.action_space.sample() # This is a line to sample a move from the legal moves
    next_state, reward, done, info = env.step(action) # This is the next position the reward that the agent and if the game is terminated and some aditional infostate = next_state