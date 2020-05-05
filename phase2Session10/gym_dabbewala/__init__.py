from gym.envs.registration import register

register(
    id='DabbeWala-v0',
    entry_point='gym_dabbewala.envs:DabbeWalaEnv',
    max_episode_steps=2000
)
