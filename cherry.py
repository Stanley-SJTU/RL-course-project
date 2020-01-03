import cherry as ch

# Wrap environments
env = gym.make('CartPole-v0')
env = ch.envs.Logger(env, interval=1000)
env = ch.envs.Torch(env)

policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
replay = ch.ExperienceReplay()  # Manage transitions

for step in range(1000):
    state = env.reset()
    while True:
        mass = Categorical(policy(state))
        action = mass.sample()
        log_prob = mass.log_prob(action)
        next_state, reward, done, _ = env.step(action)

        # Build the ExperienceReplay
        replay.append(state, action, reward, next_state, done, log_prob=log_prob)
        if done:
            break
        else:
            state = next_state

    # Discounting and normalizing rewards
    rewards = ch.td.discount(0.99, replay.reward(), replay.done())
    rewards = ch.normalize(rewards)

    loss = -th.sum(replay.log_prob() * rewards)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    replay.empty()
