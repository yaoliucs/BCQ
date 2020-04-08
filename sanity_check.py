import argparse
import gym
import numpy as np
import os
import torch

from main import eval_policy

import BCQ
import DDPG
from utils import ExtendedReplayBuffer


# Handles interactions with the environment, i.e. train behavioral or generate buffer
def interact_with_environment(env, state_dim, action_dim, max_action, device, args):
    # For saving files
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # Initialize and load policy
    policy = DDPG.DDPG(state_dim, action_dim, max_action, device)  # , args.discount, args.tau)
    if args.generate_buffer: policy.load(f"./models/behavioral_{setting}")

    # Initialize buffer
    replay_buffer = ExtendedReplayBuffer(state_dim, action_dim, env.init_qpos.shape[0], env.init_qvel.shape[0], device)

    evaluations = []
    episode_values = []

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # Interact with the environment for max_timesteps
    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action with noise
        if (
                    (args.generate_buffer and np.random.uniform(0, 1) < args.rand_action_p) or
                    (args.train_behavioral and t < args.start_timesteps)
        ):
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.gaussian_std, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        qpos = env.sim.data.qpos.flat.copy()
        qvel = env.sim.data.qvel.flat.copy()
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool, qpos, qvel)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if args.train_behavioral and t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            episode_values.append(episode_reward)
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    # Save final buffer and performance
    evaluations.append(eval_policy(policy, args.env, args.seed, 100))
    np.save(f"./results/buffer_policy_performance_{buffer_name}", evaluations)
    np.save(f"./results/buffer_average_performance_{buffer_name}", np.mean(episode_values))
    replay_buffer.save(f"./buffers/{buffer_name}")


def check_state_filter(state_dim, action_dim, max_state, max_action, device, args):
    # For saving files
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"
    hp_setting = f"N{args.load_buffer_size}_k{str(args.sigmoid_k)}_betac{str(args.beta_c)}_betaa{str(args.beta_a)}"

    # Initialize policy
    env = gym.make(args.env)
    policy = BCQ.BCQ_state(state_dim, action_dim, max_state, max_action, device,
                           args.discount, args.tau, args.lmbda, args.phi,
                           beta_a=args.beta_a, beta_c=args.beta_c, sigmoid_k=args.sigmoid_k)

    # Load buffer
    replay_buffer = ExtendedReplayBuffer(state_dim, action_dim, env.init_qpos.shape[0], env.init_qvel.shape[0], device)
    replay_buffer.load(f"./buffers/{buffer_name}")

    evaluations = []
    filter_scores = []
    episode_num = 0
    done = True
    training_iters = 0

    # state, action, next_state, reward, not_done, qpos, qvel = replay_buffer.sample_more(100)
    # score, value, critic = evaluate_filter_and_critic(policy, state, qpos, qvel, args)
    # print(score, value, critic)

    while training_iters < int(args.max_timesteps / 5):
        vae_loss = policy.train_vae(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
        print(f"Training iterations: {training_iters}")
        print("VAE loss", vae_loss)
        training_iters += args.eval_freq

    training_iters = 0
    while training_iters < args.max_timesteps:
        score = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
        evaluations.append(eval_policy(policy, args.env, args.seed))
        np.save(f"./results/SCheck_{hp_setting}_{buffer_name}", evaluations)

        filter_scores = np.append(filter_scores, score)
        np.save(f"./results/SCheck_{hp_setting}_{buffer_name}_filter", filter_scores)

        state, action, next_state, reward, not_done, qpos, qvel = replay_buffer.sample_more(100)
        score, value, critic = evaluate_filter_and_critic(policy, state, qpos, qvel, args)
        np.save(f"./results/SCheck_{hp_setting}_{buffer_name}_{training_iters}_score", score)
        np.save(f"./results/SCheck_{hp_setting}_{buffer_name}_{training_iters}_value", value)
        np.save(f"./results/SCheck_{hp_setting}_{buffer_name}_{training_iters}_critic", critic)
        np.save(f"./results/SCheck_{hp_setting}_{buffer_name}_{training_iters}_qpos", qpos.cpu().numpy())
        np.save(f"./results/SCheck_{hp_setting}_{buffer_name}_{training_iters}_qvel", qvel.cpu().numpy())

        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}")

def evaluate_filter_and_critic(policy, state, qpos, qvel, args):
    # Compute score
    recon, mean, std = policy.vae2(state)
    recon_loss = ((recon - state) ** 2).mean(dim=1)
    KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(dim=1)
    vae_loss = recon_loss + 0.5 * KL_loss
    score = -vae_loss.detach().cpu().numpy().flatten()

    # Evaluate
    evaluates = evaluate_from_sa(policy, args.env, args.seed, qpos, qvel, args.discount)

    # Compute critic
    with torch.no_grad():
        sampled_actions = policy.vae.decode(state)
        perturbed_actions = policy.actor(state, sampled_actions)
        critic_values = policy.critic.q1(state, perturbed_actions).cpu().numpy().flatten()

    return (score, evaluates, critic_values)


def evaluate_from_sa(policy, env_name, seed, qpos_tensor, qvel_tensor, gamma, num_trajectory=1):
    env = gym.make(env_name)
    env.seed(seed + 100)

    n = qpos_tensor.shape[0]
    policy_values = []
    for i in range(n):
        episode_values = []
        qpos = qpos_tensor.cpu().numpy()[i, :]
        qvel = qvel_tensor.cpu().numpy()[i, :]
        for k in range(num_trajectory):
            episode_reward = 0
            env.reset()
            env.set_state(qpos, qvel)
            state = env.env._get_obs()
            gamma_n = 1
            for t in range(int(args.max_timesteps)):
                action = policy.select_action(np.array(state))
                state, reward, done, _ = env.step(action)
                episode_reward += gamma_n*reward
                gamma_n *= gamma
                if done:
                    break
            episode_values.append(episode_reward)
        policy_values.append(np.mean(episode_values))
    return np.array(policy_values)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Hopper-v3")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name",
                        default="Extended-Imperfect")  # Prepends name to filename "Final/Imitation/Imperfect"
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6,
                        type=int)  # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--start_timesteps", default=25e3,
                        type=int)  # Time steps initial random policy is used before training behavioral
    parser.add_argument("--rand_action_p", default=0.3,
                        type=float)  # Probability of selecting random action during batch generation
    parser.add_argument("--gaussian_std", default=0.3,
                        type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    parser.add_argument("--batch_size", default=100, type=int)  # Mini batch size for networks
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--lmbda", default=0.75)  # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--phi", default=0.05)  # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral (DDPG)
    parser.add_argument("--generate_buffer", action="store_true")  # If true, generate buffer
    parser.add_argument("--state_vae", action="store_true")  # If true, use an vae to fit state distribution
    parser.add_argument("--test_state_vae", action="store_true")  # If true, only test vae
    parser.add_argument("--score_activation", default="sigmoid")  # "sigmoid", "sigmoid_exp", "hard"
    parser.add_argument("--beta_a", default=0.0, type=float)  # state filter hyperparameter (actor)
    parser.add_argument("--beta_c", default=-2.0, type=float)  # state filter hyperparameter (critic)
    parser.add_argument("--sigmoid_k", default=100, type=float)
    parser.add_argument("--load_buffer_size", default=1e6, type=int)  # number of samples to load into the buffer
    args = parser.parse_args()

    print("---------------------------------------")
    if args.train_behavioral:
        print(f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
    elif args.generate_buffer:
        print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
    elif args.state_vae:
        print(f"Setting: Training BCQ with state vae, Env: {args.env}, Seed: {args.seed}")
    else:
        print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if args.train_behavioral and args.generate_buffer:
        print("Train_behavioral and generate_buffer cannot both be true.")
        exit()

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    if not os.path.exists("./buffers"):
        os.makedirs("./buffers")

    env = gym.make(args.env)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_state = float(env.observation_space.high[0])
    if max_state == np.inf:
        max_state = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train_behavioral or args.generate_buffer:
        interact_with_environment(env, state_dim, action_dim, max_action, device, args)
    else:
        check_state_filter(state_dim, action_dim, None, max_action, device, args)
