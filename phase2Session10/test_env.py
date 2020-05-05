"""
to test the changes made in environment real quick
"""


import gym
import gym_dabbewala


def random_agent(episodes=10000):
	env = gym.make("DabbeWala-v0")
	env.reset()
	env.render()
	for _ in range(episodes):
		action = env.action_space.sample()
		_, reward, done, _ = env.step(action)
		env.render()
		# print(f"reward{reward}, car_position{(env.car.centerx, env.car.centery)}, pickup: {env.x1,env.y1}, drop: {env.x2,env.y2}")
		# print(f'{len(env.x), len(env.y)}')
		
		if done:
			break
	env.close()


if __name__ == "__main__":
    random_agent()
