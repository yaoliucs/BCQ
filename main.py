import argparse
import gym
import numpy as np
import os
import torch
import time

import BCQ
import DDPG
import BEAR
import utils


# Handles interactions with the environment, i.e. train behavioral or generate buffer
def interact_with_environment(env, state_dim, action_dim, max_action, device, args):
    # For saving files
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # Initialize and load policy
    policy = DDPG.DDPG(state_dim, action_dim, max_action, device)  # , args.discount, args.tau)
    if args.generate_buffer:
        if args.policy_seed != args.seed:
            policy_setting = f"{args.env}_{args.policy_seed}"
        else:
            policy_setting = setting
        policy.load(f"./models/behavioral_{policy_setting}")
    if args.load_multiple_policy:
        chk = 1
        policy.load(f"./models/behavioral_{setting}_chk{chk}")

    # Initialize buffer
    # replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer = utils.ExtendedReplayBuffer(state_dim, action_dim, env.init_qpos.shape[0], env.init_qvel.shape[0], device)

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
        # replay_buffer.add(state, action, next_state, reward, done_bool)
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

        # Evaluate episode
        if args.train_behavioral and (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))

            if args.n_checkpoint > 0 and (t+1) % int(args.max_timesteps/args.n_checkpoint) == 0:
                k = int((t+1)/int(args.max_timesteps/args.n_checkpoint))
                policy.save(f"./models/behavioral_{setting}_chk{k}")
                np.save(f"./results/behavioral_{setting}_chk{k}", evaluations)
            else:
                policy.save(f"./models/behavioral_{setting}")
                np.save(f"./results/behavioral_{setting}", evaluations)
        elif args.generate_buffer and args.load_multiple_policy:
            if (t+1) % int(args.max_timesteps/args.n_checkpoint) == 0:
                chk = min(chk+1,args.n_checkpoint)
                policy.load(f"./models/behavioral_{setting}_chk{chk}")

    # Save final policy
    if args.train_behavioral:
        if args.n_checkpoint > 0:
            k = int((t + 1) / int(args.max_timesteps / args.n_checkpoint))
            policy.save(f"./models/behavioral_{setting}_chk{k}")
        else:
            policy.save(f"./models/behavioral_{setting}")
        replay_buffer.save(f"./buffers/{buffer_name}")
        noisy_evaluation = np.mean(episode_values)
        np.save(f"./results/buffer_average_performance_{buffer_name}", noisy_evaluation)

    # Save final buffer and performance
    else:
        evaluations.append(eval_policy(policy, args.env, args.seed, 100))
        np.save(f"./results/buffer_policy_performance_{buffer_name}", evaluations)
        noisy_evaluation = eval_noisy_policy(policy, args.env, args.seed,
                                             args.rand_action_p, args.gaussian_std, action_dim, max_action)
        np.save(f"./results/buffer_average_performance_{buffer_name}", [noisy_evaluation, np.mean(episode_values)])
        replay_buffer.save(f"./buffers/{buffer_name}")


# Train BEAR QL
def train_BEAR(state_dim, action_dim, max_action, device, args):
    print("Training BEAR\n")
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"
    hp_setting = f"N{args.load_buffer_size}_phi{args.phi}_n{args.n_action}_ne{args.n_action_execute}" \
                 f"_{args.score_activation}_k{str(args.sigmoid_k)}_betac{str(args.beta_c)}_betaa{str(args.beta_a)}"

    # Initialize policy
    policy = BEAR.BEAR(2, state_dim, action_dim, max_action, delta_conf=0.1, use_bootstrap=False,
                       version=args.version,
                       lambda_=0.0,
                       threshold=0.05,
                       mode=args.mode,
                       num_samples_match=args.num_samples_match,
                       mmd_sigma=args.mmd_sigma,
                       lagrange_thresh=args.lagrange_thresh,
                       use_kl=(True if args.distance_type == "KL" else False),
                       use_ensemble=(False if args.use_ensemble_variance == "False" else True),
                       kernel_type=args.kernel_type,
                       actor_lr=args.actor_lr)

    # Load buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer.load(f"./buffers/Extended-{buffer_name}", args.load_buffer_size, bootstrap_dim=4)

    if args.actor_lr != 1e-3:
        hp_setting += f"_lr{args.actor_lr}"
    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0

    while training_iters < args.max_timesteps:
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)

        evaluations.append(eval_policy(policy, args.env, args.seed))
        np.save(f"./results/BEAR3_{hp_setting}_{buffer_name}", evaluations)

        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}")


def train_BEAR_state(state_dim, action_dim, max_action, device, args):
    print("Training BEARState\n")
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"
    hp_setting = f"N{args.load_buffer_size}_phi{args.phi}_n{args.n_action}_ne{args.n_action_execute}" \
                 f"_{args.score_activation}_k{str(args.sigmoid_k)}_betac{str(args.beta_c)}_betaa{str(args.beta_a)}"
    # Initialize policy
    policy = BEAR.BEAR(2, state_dim, action_dim, max_action, delta_conf=0.1, use_bootstrap=False,
                       version=args.version,
                       lambda_=0.0,
                       threshold=0.05,
                       mode=args.mode,
                       num_samples_match=args.num_samples_match,
                       mmd_sigma=args.mmd_sigma,
                       lagrange_thresh=args.lagrange_thresh,
                       use_kl=(True if args.distance_type == "KL" else False),
                       use_ensemble=(False if args.use_ensemble_variance == "False" else True),
                       kernel_type=args.kernel_type,
                       use_state_vae=args.state_vae,
                       actor_lr=args.actor_lr,
                       beta_a=args.beta_a, beta_c=args.beta_c, sigmoid_k=args.sigmoid_k,
                       n_action=args.n_action, n_action_execute=args.n_action_execute,
                       qbackup=args.qbackup, qbackup_noise=args.qbackup_noise,
                       vmin=args.vmin,
                       )

    # Load buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer.load(f"./buffers/Extended-{buffer_name}", args.load_buffer_size, bootstrap_dim=4)

    evaluations = []
    training_iters = 0

    while training_iters < int(200000):
        vae_loss = policy.train_vae(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
        print(f"Training iterations: {training_iters}")
        print("VAE loss", vae_loss)
        training_iters += args.eval_freq

    if args.vae_type == "gumbel":
        hp_setting += "_gsvae_2"
    if args.pretrain_vae:
        hp_setting += "_fixvae"
    if args.qbackup:
        hp_setting += f"_qbackup{args.qbackup_noise}"
    if args.actor_lr != 1e-3:
        hp_setting += f"_lr{args.actor_lr}"

    training_iters = 0

    while training_iters < args.max_timesteps:
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)

        evaluations.append(eval_policy(policy, args.env, args.seed))
        np.save(f"./results/BEARState3_{hp_setting}_{buffer_name}", evaluations)

        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}")


# Trains BCQ offline
def train_BCQ(state_dim, action_dim, max_action, device, args):
    # For saving files
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # Initialize policy
    policy = BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)

    # Load buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer.load(f"./buffers/{buffer_name}", args.load_buffer_size)

    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0

    while training_iters < args.max_timesteps:
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)

        evaluations.append(eval_policy(policy, args.env, args.seed))
        np.save(f"./results/BCQ_N{args.load_buffer_size}_phi{args.phi}_{buffer_name}", evaluations)

        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}")


def train_BCQ_state(state_dim, action_dim, max_state, max_action, device, args):
    # For saving files
    setting = f"{args.env}_{args.train_seed}"
    log_name = f"{args.buffer_name}_{setting}"

    print("=== Start Train ===\n")
    print("Args:\n",args)

    # Initialize policy
    policy = BCQ.BCQ_state(state_dim, action_dim, max_state, max_action, device,
                           args.discount, args.tau, args.lmbda, args.phi,
                           n_action=args.n_action, n_action_execute=args.n_action_execute,
                           qbackup=args.qbackup, qbackup_noise=args.qbackup_noise,
                           score_activation=args.score_activation,
                           actor_lr=args.actor_lr,
                           vae_type=args.vae_type,
                           beta_a=args.beta_a, beta_c=args.beta_c, sigmoid_k=args.sigmoid_k,
                           pretrain_vae=args.pretrain_vae,
                           vmin=args.vmin,)

    # Load buffer
    # replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    # replay_buffer.load(f"./buffers/{buffer_name}", args.load_buffer_size)
    replay_buffer = utils.ExtendedReplayBuffer(state_dim, action_dim, env.init_qpos.shape[0],
                                               env.init_qvel.shape[0], device)
    replay_buffer.load(f"./buffers/Extended-{args.buffer_name}_{args.env}_{args.seed}", args.load_buffer_size)

    evaluations = []
    filter_scores = []
    episode_num = 0
    done = True

    training_iters = 0
    if args.pretrain_vae:
        while training_iters < 200000:
            vae_loss = policy.train_action_vae(replay_buffer, iterations=int(args.eval_freq),
                                               batch_size=args.batch_size)
            print(f"Training iterations: {training_iters}")
            print("Action VAE loss", vae_loss)
            training_iters += args.eval_freq

    training_iters = 0
    while training_iters < 200000:
        vae_loss = policy.train_vae(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
        print(f"Training iterations: {training_iters}. State VAE loss: {vae_loss:.3f}.") # Recon MSE: {recon_mse:.3f}"
        training_iters += args.eval_freq
    if args.vae_type == "gumbel":
        state, action, next_state, reward, not_done = replay_buffer.sample(100000)
        policy.vae2.compute_frequency(next_state)
        policy.vae2.save(f"./models/gsvae_{setting}")
    else:
        policy.vae2.save(f"./models/vae_{setting}")

    if args.automatic_beta:  # args.automatic_beta:
        test_loss = policy.test_vae(replay_buffer, batch_size=100000)
        beta_c = np.percentile(test_loss, args.beta_percentile)
        policy.beta_c = beta_c
        hp_setting = f"N{args.load_buffer_size}_phi{args.phi}_n{args.n_action}_ne{args.n_action_execute}" \
                     f"_{args.score_activation}_k{str(args.sigmoid_k)}_cpercentile{args.beta_percentile}"
        np.save(f"./results/BCQState_{hp_setting}_{log_name}_vaescore", test_loss)
        print("Test vae",args.beta_percentile,"percentile:", beta_c)
    else:
        hp_setting = f"N{args.load_buffer_size}_phi{args.phi}_n{args.n_action}_ne{args.n_action_execute}" \
                     f"_{args.score_activation}_k{str(args.sigmoid_k)}_betac{str(args.beta_c)}_betaa{str(args.beta_a)}"

    if args.vae_type == "gumbel":
        hp_setting += "_gsvae_2"
    if args.pretrain_vae:
        hp_setting += "_fixvae"
    if args.qbackup:
        hp_setting += f"_qbackup{args.qbackup_noise}"
    if args.actor_lr != 1e-3:
        hp_setting += f"_lr{args.actor_lr}"

    if args.test_critic_elbo:
        if not os.path.exists(f"./results/SCheck_{hp_setting}_{log_name}"):
            os.mkdir(f"./results/SCheck_{hp_setting}_{log_name}")

    # Start training
    print("Log files at:", f"./results/BCQState_{hp_setting}_{log_name}")
    training_iters = 0
    while training_iters < args.max_timesteps:
        score = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
        evaluations.append(eval_policy(policy, args.env, args.seed, eval_episodes=20))
        np.save(f"./results/BCQState_{hp_setting}_{log_name}", evaluations)

        filter_scores = np.append(filter_scores, score)
        np.save(f"./results/BCQState_{hp_setting}_{log_name}_filter", filter_scores)

        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}")

        # if args.test_critic_elbo and (training_iters % 100000 == 0):
        #     init_state, init_action, init_qp, init_qv = sample_initial_state_action(policy, 100, args, policy.device)
        #     score, value, bsl_value, critic, bsl_critic \
        #         = evaluate_filter_and_critic(policy, init_state, init_action, init_qp, init_qv, args)
        #     np.save(f"./results/SCheck_{hp_setting}_{log_name}/{training_iters}_init_score", score)
        #     np.save(f"./results/SCheck_{hp_setting}_{log_name}/{training_iters}_init_value", value)
        #     np.save(f"./results/SCheck_{hp_setting}_{log_name}/{training_iters}_init_bsl_value", bsl_value)
        #     np.save(f"./results/SCheck_{hp_setting}_{log_name}/{training_iters}_init_critic", critic)
        #     np.save(f"./results/SCheck_{hp_setting}_{log_name}/{training_iters}_init_bsl_critic", bsl_critic)
        #     np.save(f"./results/SCheck_{hp_setting}_{log_name}/{training_iters}_init_qpos", init_qp.cpu().numpy())
        #     np.save(f"./results/SCheck_{hp_setting}_{log_name}/{training_iters}_init_qvel", init_qv.cpu().numpy())
        #
        #     state, action, next_state, reward, not_done, qpos, qvel = replay_buffer.sample_more(100)
        #     score, value, bsl_value, critic, bsl_critic \
        #         = evaluate_filter_and_critic(policy, state, action, qpos, qvel, args)
        #     np.save(f"./results/SCheck_{hp_setting}_{log_name}/{training_iters}_score", score)
        #     np.save(f"./results/SCheck_{hp_setting}_{log_name}/{training_iters}_value", value)
        #     np.save(f"./results/SCheck_{hp_setting}_{log_name}/{training_iters}_bsl_value", bsl_value)
        #     np.save(f"./results/SCheck_{hp_setting}_{log_name}/{training_iters}_critic", critic)
        #     np.save(f"./results/SCheck_{hp_setting}_{log_name}/{training_iters}_bsl_critic", bsl_critic)
        #     np.save(f"./results/SCheck_{hp_setting}_{log_name}/{training_iters}_qpos", qpos.cpu().numpy())
        #     np.save(f"./results/SCheck_{hp_setting}_{log_name}/{training_iters}_qvel", qvel.cpu().numpy())


def train_BC(state_dim, action_dim, max_action, device, args):
    # For saving files
    setting = f"{args.env}_{args.train_seed}"
    log_name = f"{args.buffer_name}_{setting}"

    # Initialize policy
    policy = BCQ.BCQ_state(state_dim, action_dim, None, max_action, device, args.discount, args.tau, args.lmbda, phi=0)

    # Load buffer
    replay_buffer = utils.ExtendedReplayBuffer(state_dim, action_dim, env.init_qpos.shape[0],
                                               env.init_qvel.shape[0], device)
    replay_buffer.load(f"./buffers/Extended-{args.buffer_name}_{args.env}_{args.seed}", args.load_buffer_size)

    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0

    while training_iters < args.max_timesteps:
        vae_loss = policy.train_action_vae(replay_buffer, iterations=int(args.eval_freq),
                                           batch_size=args.batch_size)

        evaluations.append(eval_policy(policy, args.env, args.seed))
        np.save(f"./results/BC_N{args.load_buffer_size}_{log_name}", evaluations)

        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}")


def train_offline_DDPG(state_dim, action_dim, max_action, device, args):
    # For saving files
    setting = f"{args.env}_{args.train_seed}"
    log_name = f"{args.buffer_name}_{setting}"

    # Initialize policy
    policy = DDPG.DDPG(state_dim, action_dim, max_action, device)  # , args.discount, args.tau)

    # Load buffer
    replay_buffer = utils.ExtendedReplayBuffer(state_dim, action_dim, env.init_qpos.shape[0],
                                               env.init_qvel.shape[0], device)
    replay_buffer.load(f"./buffers/Extended-{args.buffer_name}_{args.env}_{args.seed}", args.load_buffer_size)

    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0

    while training_iters < args.max_timesteps:
        pol_vals = policy.train(replay_buffer, batch_size=args.batch_size)
        training_iters += 1
        if training_iters % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/DDPG_N{args.load_buffer_size}_{log_name}", evaluations)
            print(f"Training iterations: {training_iters}")


def test_vae_state(state_dim, action_dim, max_state, max_action, device, args):
    # For saving files
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"
    hp_setting = f"{args.score_activation}_k{str(args.sigmoid_k)}_betac{str(args.beta_c)}_betaa{str(args.beta_a)}"

    # Initialize policy
    policy = BCQ.BCQ_state(state_dim, action_dim, max_state, max_action, device,
                           args.discount, args.tau, args.lmbda, args.phi,
                           beta_a=args.beta_a, beta_c=args.beta_c, sigmoid_k=args.sigmoid_k)

    # Load buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer.load(f"./buffers/{buffer_name}", args.load_buffer_size)

    training_iters = 0

    while training_iters < int(args.max_timesteps / 5):
        vae_loss = policy.train_vae(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
        print(f"Training iterations: {training_iters}")
        print("VAE loss", vae_loss)
        training_iters += args.eval_freq
    policy.vae2.save(f"./models/vae_{setting}")
    test_loss = policy.test_vae(replay_buffer, batch_size=100000)
    print(test_loss)
    np.save(f"./results/vae_pretrain/elbo_{args.seed}", test_loss)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def eval_noisy_policy(policy, env_name, seed, rand_action_p, gaussian_std,
                      action_dim, max_action, eval_episodes=100):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            if (np.random.uniform(0, 1) < rand_action_p):
                action = env.action_space.sample()
            else:
                action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * gaussian_std, size=action_dim)
                ).clip(-max_action, max_action)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation policy with noise over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def evaluate_filter_and_critic(policy, state, action, qpos, qvel, args):
    # Compute score
    if policy.discrete_type == "vanilla":
        recon, mean, std = policy.vae2(state)
        recon_loss = ((recon - state) ** 2).mean(dim=1)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(dim=1)
        if policy.score_activation == "KL":
            vae_loss = 0.5 * KL_loss
        else:
            vae_loss = recon_loss + 0.5 * KL_loss
    elif policy.discrete_type == "gumbel":
        if policy.score_activation == "KL":
            vae_loss = policy.vae2.latent_KL(state)
        elif policy.score_activation == "frequency":
            vae_loss = -policy.vae2.frequency_score(state)
        else:
            vae_loss = policy.vae2.elbo_loss(state)
    score = -vae_loss.detach().cpu().numpy().flatten()

    # Evaluate
    evaluates = evaluate_from_sa(policy, args.env, args.seed, qpos, qvel, args.discount, num_trajectory=1)
    bsl_evaluates = evaluate_from_sa(policy, args.env, args.seed, qpos, qvel, args.discount, action, num_trajectory=1)

    # Compute critic
    batch_size = state.shape[0]
    with torch.no_grad():
        repeated_state = torch.repeat_interleave(state, args.n_action_execute, 0)
        sampled_actions = policy.vae.decode(repeated_state)
        perturbed_actions = policy.actor(repeated_state, sampled_actions)
        critic_values = policy.critic.q1(repeated_state, perturbed_actions).reshape(batch_size, -1).max(1)[0].reshape(
            -1, 1)
        critic_values = critic_values.cpu().numpy().flatten()
    with torch.no_grad():
        bsl_critic_values = policy.critic.q1(state, action).cpu().numpy().flatten()

    return (score, evaluates, bsl_evaluates, critic_values, bsl_critic_values)

def sample_initial_state_action(policy, num_state, args, device):
    env = gym.make(args.env)
    env.seed(args.seed+100)

    states = np.zeros((num_state, env.observation_space.shape[0]))
    qpos = np.zeros((num_state, env.init_qpos.shape[0]))
    qvel = np.zeros((num_state, env.init_qvel.shape[0]))

    for i in range(num_state):
        s = env.reset()
        qp = env.sim.data.qpos.flat.copy()
        qv = env.sim.data.qvel.flat.copy()
        states[i,:] = s
        qpos[i,:] = qp
        qvel[i,:] = qv

    states = torch.FloatTensor(states).to(device)
    with torch.no_grad():
        actions = policy.vae.decode(states)
    qpos = torch.FloatTensor(qpos).to(device)
    qvel = torch.FloatTensor(qvel).to(device)
    return (states, actions, qpos, qvel)


def evaluate_from_sa(policy, env_name, seed, qpos_tensor, qvel_tensor, gamma, action_tensor=None, num_trajectory=1):
    env = gym.make(env_name)
    env.seed(seed + 100)

    n = qpos_tensor.shape[0]
    policy_values = []
    for i in range(n):
        episode_values = []
        qpos = qpos_tensor.cpu().numpy()[i, :]
        qvel = qvel_tensor.cpu().numpy()[i, :]
        if action_tensor is not None:
            init_action = action_tensor.cpu().numpy()[i, :]
        else:
            init_action = None
        for k in range(num_trajectory):
            episode_reward = 0
            env.reset()
            env.set_state(qpos, qvel)
            state = env.env._get_obs()
            gamma_n = 1
            for t in range(int(args.max_timesteps)):
                if t == 0 and (init_action is not None):
                    action = init_action
                else:
                    action = policy.select_action(np.array(state))
                state, reward, done, _ = env.step(action)
                episode_reward += gamma_n * reward
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
    parser.add_argument("--policy_seed", default=-1, type=int) # seed of behavior policy, if we are using a different seed with behavior policy
    parser.add_argument("--train_seed", default=-1, type=int)
    parser.add_argument("--buffer_name", default="Imperfect")  # Prepends name to filename "Final/Imitation/Imperfect"
    parser.add_argument("--eval_freq", default=1e4, type=float)  # How often (time steps) we evaluate
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
    parser.add_argument("--phi", default=0.05, type=float)  # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral (DDPG)
    parser.add_argument("--generate_buffer", action="store_true")  # If true, generate buffer
    parser.add_argument("--state_vae", action="store_true")  # If true, use an vae to fit state distribution
    parser.add_argument("--vae_type", default="vanilla")  # "vanilla": vae, "gumbel": gumbel, discrete
    parser.add_argument("--test_state_vae", action="store_true")  # If true, only test vae
    parser.add_argument("--score_activation", default="sigmoid")  # "sigmoid", "KL", "frequency"
    parser.add_argument("--beta_a", default=0.0, type=float)  # state filter hyperparameter (actor)
    parser.add_argument("--beta_c", default=-0.4, type=float)  # state filter hyperparameter (critic)
    parser.add_argument("--sigmoid_k", default=100, type=float)
    parser.add_argument("--load_buffer_size", default=100000, type=int)  # number of samples to load into the buffer
    parser.add_argument("--actor_lr", default=1e-3, type=float) # learning rate of actor
    parser.add_argument("--vmin", default=0, type=float)  # learning rate of actor
    # BEAR parameter
    parser.add_argument("--bear", action="store_true")  # If true, use BEAR
    parser.add_argument("--version", default='0',
                        type=str)  # Basically whether to do min(Q), max(Q), mean(Q)
    parser.add_argument('--mode', default='hardcoded', #hardcoded
                        type=str)  # Whether to do automatic lagrange dual descent or manually tune coefficient of the MMD loss (prefered "auto")
    parser.add_argument('--num_samples_match', default=5, type=int)  # number of samples to do matching in MMD
    parser.add_argument('--mmd_sigma', default=20.0, type=float)  # The bandwidth of the MMD kernel parameter default 10
    parser.add_argument('--kernel_type', default='laplacian',
                        type=str)  # kernel type for MMD ("laplacian" or "gaussian")
    parser.add_argument('--lagrange_thresh', default=10.0,
                        type=float)  # What is the threshold for the lagrange multiplier
    parser.add_argument('--distance_type', default="MMD", type=str)  # Distance type ("KL" or "MMD")
    parser.add_argument('--use_ensemble_variance', default='False', type=str)  # Whether to use ensemble variance or not

    parser.add_argument("--qbackup", action="store_true")  # If true, use q learning backup instead of actor critic algorithm
    parser.add_argument("--qbackup_noise", type=float, default=0.0)
    parser.add_argument("--test_critic_elbo", default=True)  # If true, only test vae
    parser.add_argument("--automatic_beta", action="store_true")  # If true, use percentile for beta
    parser.add_argument("--beta_percentile", type=float, default=1.5)  #
    parser.add_argument("--pretrain_vae", action="store_true")  # If true, pre train action vae
    parser.add_argument("--n_action", default=10, type=int)
    parser.add_argument("--n_action_execute", default=100, type=int)

    parser.add_argument("--bc", action="store_true")  # If true, run BC
    parser.add_argument("--ddpg", action="store_true")  # If true, run DDPG

    parser.add_argument("--n_checkpoint", default=0, type=int) # Number of checkpoints to save
    parser.add_argument("--load_multiple_policy", action="store_true") # Load multiple policy

    args = parser.parse_args()

    print("---------------------------------------")
    if args.train_behavioral:
        print(f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
    elif args.generate_buffer:
        print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
    elif args.bear:
        if args.state_vae:
            print(f"Setting: Training BEAR with state vae, Env: {args.env}, Seed: {args.seed}")
        else:
            print(f"Setting: Training BEAR, Env: {args.env}, Seed: {args.seed}")
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
    if args.env == "HalfCheetah-v2":
        args.vmin = -591
    if args.env == "Walker2d-v2":
        args.vmin = -16

    if args.policy_seed == -1:
        args.policy_seed = args.seed
    if args.train_seed == -1:
        args.train_seed = args.seed

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.train_seed != args.seed:
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
    elif args.test_state_vae:
        test_vae_state(state_dim, action_dim, max_state, max_action, device, args)
    elif args.bear:
        if args.state_vae:
            train_BEAR_state(state_dim, action_dim, max_action, device, args)
        else:
            train_BEAR(state_dim, action_dim, max_action, device, args)
    elif args.state_vae:
        train_BCQ_state(state_dim, action_dim, max_state, max_action, device, args)
    elif args.bc:
        train_BC(state_dim, action_dim, max_action, device, args)
    elif args.ddpg:
        train_offline_DDPG(state_dim, action_dim, max_action, device, args)
    else:
        train_BCQ(state_dim, action_dim, max_action, device, args)
