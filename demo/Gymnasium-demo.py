import gymnasium as gym

# 创建一个环境，这里用经典的 CartPole
env = gym.make("CartPole-v1", render_mode="human")

# 重置环境，返回初始观测值和info
observation, info = env.reset(seed=42)

for _ in range(1000):
    # 随机选择一个动作
    action = env.action_space.sample()

    # 执行动作，得到下一步信息
    observation, reward, terminated, truncated, info = env.step(action)

    # 如果游戏结束（杆倒下了），重置
    if terminated or truncated:
        observation, info = env.reset()

env.close()
