from gym.envs.registration import register

register(
    id='Chess-v1',
    entry_point='env.envs:ChessEnv',
    kwargs={'chess960': False}
)