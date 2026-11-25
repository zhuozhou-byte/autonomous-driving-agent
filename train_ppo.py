from collections import deque
from datetime import datetime
from functools import partial
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

PROJECT_DIR = Path(__file__).resolve().parent

# Run mode
RUN_MODE = 'evaluate'
RESUME_MODEL = PROJECT_DIR / 'models' / 'ppo_curve.zip'
RESULTS_PATH = PROJECT_DIR / 'results' / 'ppo_curve.json'

# Training
TOTAL_TIMESTEPS = 100_000
TRAIN_SPEED_ONLY = True
RESUME_STEERING_NOISE_SCALE = 0.01
RESUME_THROTTLE_NOISE_SCALE = 0.25

# Maps and traffic
MAP_LAYOUT = 'C'
TRAFFIC_DENSITY = 0.0
RANDOM_SPAWN_LANE = False
MAX_EPISODE_STEPS = 1_000
RANDOM_SEED = 42

# Training maps
TRAIN_START_SEED = 2_000
TRAIN_MAP_COUNT = 1_000
TRAIN_FOCUS_SEEDS = (2016, 2028, 2155, 2339, 2544, 2603, 2663, 2725, 2783, 2832, 2864, 2876)
TRAIN_FOCUS_PROBABILITY = 0.50

# Validation and testing
VALIDATION_INTERVAL = 5_000
STOP_ON_PERFECT_VALIDATION = True
VALIDATION_START_SEED = 100
VALIDATION_MAP_COUNT = 200
TEST_START_SEED = 1_000
TEST_MAP_COUNT = 100

# PPO
LEARNING_RATE = 3e-5
ROLLOUT_STEPS = 2_048
BATCH_SIZE = 256
UPDATE_EPOCHS = 10
DISCOUNT_FACTOR = 0.999
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENTROPY_COEFFICIENT = 0.0
VALUE_COEFFICIENT = 0.5
MAX_GRAD_NORM = 0.5
POLICY_LAYERS = (128, 128)
VALUE_LAYERS = (128, 128)
STEERING_LAYERS = (128, 128)
SPEED_LAYERS = (128, 128)
HIDDEN_INITIALIZATION_GAIN = 2 ** 0.5
ACTION_INITIALIZATION_GAIN = 0.01
USE_SDE = True
SDE_SAMPLE_FREQ = 4
LOG_STD_INIT = -2.0
SQUASH_ACTIONS = True
TARGET_KL = 0.02

# Runtime
DEVICE = 'cpu'
CPU_THREADS = 1
PROGRESS_INTERVAL = 10_000
RECENT_EPISODES = 100
PPO_VERBOSE = 0
MIN_ELAPSED_SECONDS = 0.001
SEED_UPPER_BOUND = 2 ** 32
PERCENT_SCALE = 100
OUTPUT_DIR = PROJECT_DIR / 'demo_outputs' / 'ppo_training'
COLLISION_KEYS = ('crash_vehicle', 'crash_object', 'crash_building', 'crash_human')


# Package imports
_original_path = sys.path.copy()
sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != PROJECT_DIR]
try:
    from metadrive import MetaDriveEnv
    from metadrive.policy.env_input_policy import EnvInputPolicy
finally:
    sys.path[:] = _original_path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv

# Policy networks
class DrivingActionNetwork(torch.nn.Module):
    def __init__(self, input_size, steering_layers, speed_layers, activation_fn, speed_input_size=None):
        super().__init__()

        self.speed_input_size = input_size if speed_input_size is None else speed_input_size

        def make_head(widths, feature_count):
            layers = []
            previous = feature_count
            for width in widths:
                layers.extend([torch.nn.Linear(previous, width), activation_fn()])
                previous = width
            layers.append(torch.nn.Linear(previous, 1))
            return torch.nn.Sequential(*layers)

        self.steering = make_head(steering_layers, input_size)
        self.speed = make_head(speed_layers, self.speed_input_size)
        self.steering.requires_grad_(False)
        for layer in self.speed:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=HIDDEN_INITIALIZATION_GAIN)
                torch.nn.init.zeros_(layer.bias)
        torch.nn.init.orthogonal_(self.speed[-1].weight, gain=ACTION_INITIALIZATION_GAIN)

    def forward(self, features):
        return torch.cat((self.steering(features), self.speed(features[:, :self.speed_input_size])), dim=1)


class IndependentSpeedPolicy(ActorCriticPolicy):
    def __init__(self, *args, steering_layers=STEERING_LAYERS, speed_layers=SPEED_LAYERS,
                 speed_input_size=None, **kwargs):
        self.steering_layers = tuple(steering_layers)
        self.speed_layers = tuple(speed_layers)
        self.speed_input_size = speed_input_size
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        if self.net_arch['pi'] or self.action_space.shape != (2,):
            raise ValueError('Independent speed control requires raw actor features and two actions.')
        self.action_net = DrivingActionNetwork(
            self.features_dim, self.steering_layers, self.speed_layers, self.activation_fn, self.speed_input_size,
        )
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs,
        )


# Configuration checks
def validate_settings():
    if RUN_MODE not in ('evaluate', 'train'):
        raise ValueError("RUN_MODE must be 'evaluate' or 'train'.")
    if RESUME_MODEL is not None and not Path(RESUME_MODEL).is_file():
        raise FileNotFoundError(f'Model not found: {RESUME_MODEL}')
    if RESUME_MODEL is None and (RUN_MODE == 'evaluate' or TRAIN_SPEED_ONLY):
        raise ValueError('A saved model is required for evaluation or speed-only training.')
    if not 0 <= TRAFFIC_DENSITY <= 1:
        raise ValueError('TRAFFIC_DENSITY must be between 0 and 1.')
    if not isinstance(MAP_LAYOUT, str) or not MAP_LAYOUT:
        raise ValueError('MAP_LAYOUT must be a nonempty road sequence.')
    for name in (
        'TOTAL_TIMESTEPS', 'TRAIN_MAP_COUNT', 'VALIDATION_MAP_COUNT', 'VALIDATION_INTERVAL',
        'TEST_MAP_COUNT', 'MAX_EPISODE_STEPS', 'ROLLOUT_STEPS', 'BATCH_SIZE',
        'UPDATE_EPOCHS', 'CPU_THREADS', 'PROGRESS_INTERVAL', 'RECENT_EPISODES',
    ):
        value = globals()[name]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer.')
    for value in (TRAIN_START_SEED, VALIDATION_START_SEED, TEST_START_SEED, RANDOM_SEED):
        if not isinstance(value, int) or not 0 <= value < SEED_UPPER_BOUND:
            raise ValueError('Seeds must be integers in [0, 2**32).')
    if not 0 <= TRAIN_FOCUS_PROBABILITY <= 1:
        raise ValueError('TRAIN_FOCUS_PROBABILITY must be between 0 and 1.')
    if len(set(TRAIN_FOCUS_SEEDS)) != len(TRAIN_FOCUS_SEEDS) or len(TRAIN_FOCUS_SEEDS) >= TRAIN_MAP_COUNT:
        raise ValueError('Focus seeds must be unique and leave other training maps available.')
    if any(not isinstance(seed, int) or not TRAIN_START_SEED <= seed < TRAIN_START_SEED + TRAIN_MAP_COUNT
           for seed in TRAIN_FOCUS_SEEDS):
        raise ValueError('Focus seeds must belong to the training map pool.')
    for name in ('LEARNING_RATE', 'RESUME_STEERING_NOISE_SCALE', 'RESUME_THROTTLE_NOISE_SCALE'):
        if not np.isfinite(globals()[name]) or globals()[name] <= 0:
            raise ValueError(f'{name} must be finite and positive.')
    if BATCH_SIZE < 2 or ROLLOUT_STEPS % BATCH_SIZE:
        raise ValueError('BATCH_SIZE must be at least 2 and divide ROLLOUT_STEPS.')
    if SQUASH_ACTIONS and not USE_SDE:
        raise ValueError('SQUASH_ACTIONS requires USE_SDE.')
    if not isinstance(SDE_SAMPLE_FREQ, int) or (SDE_SAMPLE_FREQ != -1 and SDE_SAMPLE_FREQ <= 0):
        raise ValueError('SDE_SAMPLE_FREQ must be -1 or a positive integer.')
    if not np.isfinite(LOG_STD_INIT):
        raise ValueError('LOG_STD_INIT must be finite.')
    if TARGET_KL is not None and (not np.isfinite(TARGET_KL) or TARGET_KL <= 0):
        raise ValueError('TARGET_KL must be positive or None.')
    ranges = (
        ('Training', TRAIN_START_SEED, TRAIN_MAP_COUNT),
        ('Validation', VALIDATION_START_SEED, VALIDATION_MAP_COUNT),
        ('Test', TEST_START_SEED, TEST_MAP_COUNT),
    )
    for index, (name, first, count) in enumerate(ranges):
        if first + count > SEED_UPPER_BOUND:
            raise ValueError(f'{name} seeds must be below 2**32.')
        for other, other_first, other_count in ranges[index + 1:]:
            if max(first, other_first) < min(first + count, other_first + other_count):
                raise ValueError(f'{name} and {other.lower()} maps must not overlap.')


# Map sampling
def training_map_probabilities():
    if not TRAIN_FOCUS_SEEDS:
        return None
    probabilities = np.full(
        TRAIN_MAP_COUNT, (1 - TRAIN_FOCUS_PROBABILITY) / (TRAIN_MAP_COUNT - len(TRAIN_FOCUS_SEEDS)),
    )
    for seed in TRAIN_FOCUS_SEEDS:
        probabilities[seed - TRAIN_START_SEED] = TRAIN_FOCUS_PROBABILITY / len(TRAIN_FOCUS_SEEDS)
    return probabilities


# Environment
def environment_config(first_seed, map_count):
    return dict(
        use_render=False,
        image_observation=False,
        agent_policy=EnvInputPolicy,
        manual_control=False,
        discrete_action=False,
        map=MAP_LAYOUT,
        start_seed=first_seed,
        num_scenarios=map_count,
        random_spawn_lane_index=RANDOM_SPAWN_LANE,
        traffic_density=TRAFFIC_DENSITY,
        horizon=MAX_EPISODE_STEPS,
        store_map=False,
        log_level=logging.ERROR,
    )


def has_collision(info):
    return any(bool(info.get(key, False)) for key in COLLISION_KEYS)


class MapPool(gym.Wrapper):
    def __init__(self, env, random_maps, random_seed, map_probabilities=None):
        super().__init__(env)
        self.first_seed = env.config['start_seed']
        self.map_count = env.config['num_scenarios']
        self.random_maps = random_maps
        self.map_probabilities = map_probabilities
        self.rng = np.random.default_rng(random_seed)
        self.next_index = 0
        self.map_seed = self.first_seed
        self.collision_seen = False

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError('This environment does not support reset options.')
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.next_index = 0
        if self.random_maps:
            if self.map_probabilities is None:
                index = int(self.rng.integers(self.map_count))
            else:
                index = int(self.rng.choice(self.map_count, p=self.map_probabilities))
        else:
            index = self.next_index
            self.next_index = (self.next_index + 1) % self.map_count
        self.map_seed = self.first_seed + index
        self.collision_seen = False
        observation, info = self.env.reset(seed=self.map_seed)
        info['map_seed'] = self.map_seed
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.collision_seen |= has_collision(info)
        info['map_seed'] = self.map_seed
        info['episode_collision'] = self.collision_seen
        return observation, reward, terminated, truncated, info


def make_env(config, random_maps, random_seed, map_probabilities=None):
    env = MapPool(MetaDriveEnv(config.copy()), random_maps, random_seed, map_probabilities)
    return Monitor(env, info_keywords=('arrive_dest', 'route_completion', 'map_seed', 'episode_collision'))


# Training progress
class TrainingProgress(BaseCallback):
    def __init__(self, validation_env, best_model_path):
        super().__init__()
        self.validation_env = validation_env
        self.best_model_path = best_model_path
        self.best_score = None
        self.recent = deque(maxlen=RECENT_EPISODES)
        self.episodes = 0
        self.started_at = time.perf_counter()
        self.start_step = 0

    def _on_training_start(self):
        self.start_step = self.model.num_timesteps
        self.started_at = time.perf_counter()
        if RESUME_MODEL is not None:
            self.validate_and_save()

    def validate_and_save(self, candidate=None):
        candidate = self.model if candidate is None else candidate
        # Repeatable validation sequence
        self.validation_env.seed(RANDOM_SEED)
        observation = self.validation_env.reset()
        episodes = []
        while len(episodes) < VALIDATION_MAP_COUNT:
            action, _ = candidate.predict(observation, deterministic=True)
            observation, _, done, infos = self.validation_env.step(action)
            if done[0]:
                episodes.append(infos[0])

        seeds = [int(info['map_seed']) for info in episodes]
        expected = list(range(VALIDATION_START_SEED, VALIDATION_START_SEED + VALIDATION_MAP_COUNT))
        if seeds != expected:
            raise RuntimeError(f'Unexpected validation map sequence: {seeds}')
        success_rate = float(np.mean([info['arrive_dest'] for info in episodes]))
        route = float(np.mean([info['route_completion'] for info in episodes]))
        reward = float(np.mean([info['episode']['r'] for info in episodes]))
        out_of_road = float(np.mean([info.get('out_of_road', False) for info in episodes]))
        score = (success_rate, route, reward)
        improved = self.best_score is None or score > self.best_score
        if improved:
            self.best_score = score
            candidate.best_validation = dict(
                step=self.num_timesteps, additional_steps=self.num_timesteps - self.start_step,
                seeds=seeds, success_rate=success_rate,
                route=route, reward=reward, out_of_road_rate=out_of_road,
            )
            self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
            candidate.save(self.best_model_path)
        print(
            f'Validation at +{self.num_timesteps - self.start_step:,} steps '
            f'(total {self.num_timesteps:,}) | {len(episodes)} maps | '
            f'success {sum(bool(row["arrive_dest"]) for row in episodes)}/{len(episodes)} '
            f'({success_rate:.1%}) | mean route {route:.1%} | mean return {reward:+.1f} | '
            f'out of road {out_of_road:.0%}'
            + (' | NEW BEST' if improved else ''),
            flush=True,
        )

    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info:
                self.recent.append(info['episode'])
                self.episodes += 1
        added_steps = self.num_timesteps - self.start_step
        if added_steps % PROGRESS_INTERVAL == 0:
            elapsed = time.perf_counter() - self.started_at
            success = np.mean([row['arrive_dest'] for row in self.recent]) if self.recent else 0.0
            mean_return = np.mean([row['r'] for row in self.recent]) if self.recent else 0.0
            print(
                f'Train +{added_steps:>7,}/{TOTAL_TIMESTEPS:,} steps | total {self.num_timesteps:,} | '
                f'episodes {self.episodes} | recent success {success:.0%} | '
                f'recent training return {mean_return:+.1f} | '
                f'{added_steps / max(elapsed, MIN_ELAPSED_SECONDS):.0f} steps/s',
                flush=True,
            )
        if added_steps % VALIDATION_INTERVAL == 0:
            self.validate_and_save()
            if STOP_ON_PERFECT_VALIDATION and self.best_score[0] == 1.0:
                print('All validation maps passed. Stopping updates and starting final testing.', flush=True)
                return False
        return True

    def _on_training_end(self):
        self.validate_and_save()


# Model loading
def has_independent_speed(policy):
    return (
        hasattr(policy, 'steering_layers') and hasattr(policy, 'speed_layers')
        and hasattr(policy.action_net, 'steering') and hasattr(policy.action_net, 'speed')
    )


def freeze_steering(model):
    policy = model.policy
    if not has_independent_speed(policy):
        raise ValueError('This checkpoint needs TRAIN_SPEED_ONLY=False for further training.')
    if any(group.get('weight_decay', 0) for group in policy.optimizer.param_groups):
        raise ValueError('Frozen steering requires zero optimizer weight decay.')
    policy.pi_features_extractor.requires_grad_(False)
    policy.mlp_extractor.policy_net.requires_grad_(False)
    policy.action_net.steering.requires_grad_(False)
    mask = torch.ones_like(policy.log_std)
    mask[:, 0] = 0
    policy.log_std.register_hook(lambda gradient: gradient * mask)
    # Frozen noise momentum
    with torch.no_grad():
        for state in policy.optimizer.state.get(policy.log_std, {}).values():
            if torch.is_tensor(state) and state.shape == mask.shape:
                state.mul_(mask)


def check_model_inputs(model):
    if has_independent_speed(model.policy):
        count = model.policy.action_net.speed[0].in_features
        if count < model.observation_space.shape[0] and TRAFFIC_DENSITY > 0:
            raise ValueError('This speed policy omits lidar. Use zero traffic or train a full-observation policy.')


def ppo_settings():
    return dict(
        learning_rate=LEARNING_RATE, n_steps=ROLLOUT_STEPS, batch_size=BATCH_SIZE,
        n_epochs=UPDATE_EPOCHS, gamma=DISCOUNT_FACTOR, gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE, ent_coef=ENTROPY_COEFFICIENT,
        vf_coef=VALUE_COEFFICIENT, max_grad_norm=MAX_GRAD_NORM,
        use_sde=USE_SDE, sde_sample_freq=SDE_SAMPLE_FREQ, target_kl=TARGET_KL,
        seed=RANDOM_SEED, verbose=PPO_VERBOSE,
    )


def create_model(train_env):
    architecture = dict(pi=list(POLICY_LAYERS), vf=list(VALUE_LAYERS))
    if RESUME_MODEL is None:
        return PPO(
            'MlpPolicy', train_env, device=DEVICE, **ppo_settings(),
            policy_kwargs=dict(net_arch=architecture, log_std_init=LOG_STD_INIT, squash_output=SQUASH_ACTIONS),
        )
    model = PPO.load(RESUME_MODEL, env=train_env, device=DEVICE, force_reset=True, **ppo_settings())
    policy = model.policy
    check_model_inputs(model)
    if policy.squash_output != SQUASH_ACTIONS:
        raise ValueError('Saved action bounds do not match SQUASH_ACTIONS.')
    if not model.use_sde or policy.action_dist.use_expln:
        raise ValueError('Resume exploration requires gSDE with exponential standard deviations.')
    scales = (RESUME_STEERING_NOISE_SCALE, RESUME_THROTTLE_NOISE_SCALE)
    if any(scale != 1 for scale in scales):
        if not policy.action_dist.full_std or policy.log_std.shape[-1] != 2:
            raise ValueError('Resume exploration requires separate noise columns.')
        with torch.no_grad():
            for column, scale in enumerate(scales):
                policy.log_std[:, column].add_(float(np.log(scale)))
        policy.optimizer.state.pop(policy.log_std, None)
    for group in policy.optimizer.param_groups:
        group['lr'] = LEARNING_RATE
    if TRAIN_SPEED_ONLY:
        freeze_steering(model)
    elif has_independent_speed(policy):
        policy.action_net.steering.requires_grad_(True)
    policy.reset_noise(model.n_envs)
    return model


# Training loop
def train():
    print(f'PPO source: {RESUME_MODEL or "new model"}')
    print(f'Training maps: {TRAIN_START_SEED}..{TRAIN_START_SEED + TRAIN_MAP_COUNT - 1}')
    print(f'Validation maps: {VALIDATION_START_SEED}..{VALIDATION_START_SEED + VALIDATION_MAP_COUNT - 1}')
    print(f'Test maps: {TEST_START_SEED}..{TEST_START_SEED + TEST_MAP_COUNT - 1}')
    print(f'Focus seeds: {TRAIN_FOCUS_SEEDS} | Sampling share: {TRAIN_FOCUS_PROBABILITY:.0%}')
    print(f'Map: {MAP_LAYOUT} | Traffic: {TRAFFIC_DENSITY} | Speed-only training: {TRAIN_SPEED_ONLY}')
    print('Training and evaluation reward: MetaDrive defaults.', flush=True)

    run_dir = OUTPUT_DIR.resolve() / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    final_path = run_dir / 'ppo_agent.zip'
    best_path = run_dir / 'best_model.zip'
    train_env = make_env(
        environment_config(TRAIN_START_SEED, TRAIN_MAP_COUNT), True, RANDOM_SEED,
        map_probabilities=training_map_probabilities(),
    )
    validation_env = None
    interrupted = False
    try:
        # Separate simulation process
        validation_env = SubprocVecEnv(
            [partial(make_env, environment_config(VALIDATION_START_SEED, VALIDATION_MAP_COUNT),
                     random_maps=False, random_seed=RANDOM_SEED)],
            start_method='spawn',
        )
        model = create_model(train_env)
        starting_steps = model.num_timesteps
        model.training_settings = dict(
            **ppo_settings(),
            timesteps=TOTAL_TIMESTEPS, starting_timesteps=starting_steps,
            resume_model=None if RESUME_MODEL is None else str(Path(RESUME_MODEL).resolve()),
            train_speed_only=TRAIN_SPEED_ONLY,
            resume_steering_noise_scale=RESUME_STEERING_NOISE_SCALE,
            resume_throttle_noise_scale=RESUME_THROTTLE_NOISE_SCALE,
            training_reward='metadrive_default', evaluation_reward='metadrive_default',
            policy_kwargs=model.policy_kwargs,
            map_layout=MAP_LAYOUT, train_start_seed=TRAIN_START_SEED, train_map_count=TRAIN_MAP_COUNT,
            train_focus_seeds=list(TRAIN_FOCUS_SEEDS), train_focus_probability=TRAIN_FOCUS_PROBABILITY,
            validation_interval=VALIDATION_INTERVAL, stop_on_perfect_validation=STOP_ON_PERFECT_VALIDATION,
            validation_start_seed=VALIDATION_START_SEED, validation_map_count=VALIDATION_MAP_COUNT,
            test_start_seed=TEST_START_SEED, test_map_count=TEST_MAP_COUNT,
            traffic=TRAFFIC_DENSITY, random_spawn_lane=RANDOM_SPAWN_LANE, horizon=MAX_EPISODE_STEPS,
        )
        try:
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=TrainingProgress(validation_env, best_path),
                reset_num_timesteps=False,
            )
        except KeyboardInterrupt:
            interrupted = True
            print('\nTraining interrupted. Saving current PPO weights.', flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        model.save(final_path)
        print(f'Additional steps: {model.num_timesteps - starting_steps:,}')
        print(f'Cumulative steps: {model.num_timesteps:,}')
        print(f'Final model: {final_path}')
        print(f'Best model: {best_path}', flush=True)
    finally:
        try:
            if validation_env is not None:
                try:
                    validation_env.close()
                except (BrokenPipeError, EOFError):
                    if not interrupted:
                        raise
        finally:
            train_env.close()
    return None if interrupted else best_path


# Evaluation
def evaluate(model, title='Saved PPO evaluation'):
    check_model_inputs(model)
    env = MetaDriveEnv(environment_config(TEST_START_SEED, TEST_MAP_COUNT))
    results = []
    print(f'\n{title} | Deterministic actions | Default reward')
    print(f"{'Seed':>6} {'Success':>9} {'Collision':>10} {'Off-road':>10} {'Route':>9} {'Reward':>11} {'Time(s)':>9}")
    try:
        dt = env.config['physics_world_step_size'] * env.config['decision_repeat']
        for seed in range(TEST_START_SEED, TEST_START_SEED + TEST_MAP_COUNT):
            observation, _ = env.reset(seed=seed)
            total_reward = 0.0
            collision_seen = False
            for steps in range(1, MAX_EPISODE_STEPS + 1):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                collision_seen |= has_collision(info)
                if terminated or truncated:
                    break
            row = dict(
                seed=seed, success=bool(info.get('arrive_dest', False)),
                collision=collision_seen, out_of_road=bool(info.get('out_of_road', False)),
                route=float(info.get('route_completion', 0.0)),
                reward=total_reward, seconds=steps * dt, steps=steps,
            )
            results.append(row)
            print(
                f"{seed:>6} {str(row['success']):>9} {str(row['collision']):>10} "
                f"{str(row['out_of_road']):>10} {row['route']:>8.1%} "
                f"{row['reward']:>+11.2f} {row['seconds']:>9.1f}", flush=True,
            )
    finally:
        env.close()
    summary = evaluation_summary(results)
    count = summary['episodes']
    print(f"\nSuccess: {summary['successes']}/{count} ({summary['success_rate']:.1%})")
    print(f"Collision rate: {summary['collisions'] / count:.1%}")
    print(f"Out-of-road rate: {summary['out_of_road'] / count:.1%}")
    print(f"Mean route completion: {summary['mean_route_completion']:.1%}")
    print(f"Mean total reward: {summary['mean_reward']:+.2f}")
    seconds = summary['mean_success_seconds']
    average_time = 'N/A' if seconds is None else f'{seconds:.1f} s'
    print(f'Mean travel time, successful episodes: {average_time}')
    return results


def evaluation_summary(rows):
    if not rows:
        raise ValueError('Evaluation results must not be empty.')
    successes = sum(bool(row['success']) for row in rows)
    times = [row['seconds'] for row in rows if row['success']]
    return dict(
        episodes=len(rows), successes=successes, success_rate=successes / len(rows),
        collisions=sum(bool(row['collision']) for row in rows),
        out_of_road=sum(bool(row['out_of_road']) for row in rows),
        mean_route_completion=float(np.mean([row['route'] for row in rows])),
        mean_reward=float(np.mean([row['reward'] for row in rows])),
        mean_success_seconds=float(np.mean(times)) if times else None,
    )


# Result export
def save_results(model, model_path, rows):
    model_path = Path(model_path).resolve()
    try:
        model_file = model_path.relative_to(PROJECT_DIR.resolve()).as_posix()
    except ValueError:
        model_file = model_path.as_posix()
    episodes = [
        {key: value.item() if isinstance(value, np.generic) else value for key, value in row.items()}
        for row in rows
    ]
    result = dict(
        model=dict(file=model_file, sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                   num_timesteps=int(model.num_timesteps)),
        evaluation=dict(map_layout=MAP_LAYOUT, traffic_density=TRAFFIC_DENSITY,
                        seeds=[int(row['seed']) for row in episodes], deterministic=True,
                        reward='metadrive_default', max_episode_steps=MAX_EPISODE_STEPS,
                        random_spawn_lane=RANDOM_SPAWN_LANE),
        summary=evaluation_summary(episodes), episodes=episodes,
    )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Results: {RESULTS_PATH}')


def compare_results(before, after):
    if not before or [row['seed'] for row in before] != [row['seed'] for row in after]:
        raise ValueError('Both models must be evaluated on the same nonempty test seed sequence.')
    print(f'\nPPO comparison on the same {len(after)} test maps')
    print(f"{'Metric':<25} {'Before':>12} {'After':>12} {'Change':>12}")
    for label, key, scale, unit in (
        ('Success rate', 'success', PERCENT_SCALE, '%'),
        ('Out-of-road rate', 'out_of_road', PERCENT_SCALE, '%'),
        ('Mean route completion', 'route', PERCENT_SCALE, '%'),
        ('Mean total reward', 'reward', 1, ''),
    ):
        old_value = float(np.mean([row[key] for row in before])) * scale
        new_value = float(np.mean([row[key] for row in after])) * scale
        change_unit = ' pp' if unit else ''
        print(
            f'{label:<25} {old_value:>11.2f}{unit:<1} {new_value:>11.2f}{unit:<1} '
            f'{new_value - old_value:>+9.2f}{change_unit}',
            flush=True,
        )


# Entry point
def main():
    validate_settings()
    torch.set_num_threads(CPU_THREADS)
    if RUN_MODE == 'evaluate':
        model = PPO.load(RESUME_MODEL, device=DEVICE)
        rows = evaluate(model)
        save_results(model, RESUME_MODEL, rows)
        return 0
    model_path = train()
    if model_path is None:
        return 130
    before = None
    if RESUME_MODEL is not None:
        baseline = PPO.load(RESUME_MODEL, device=DEVICE)
        before = evaluate(baseline, 'Starting PPO')
        del baseline
    model = PPO.load(model_path, device=DEVICE)
    after = evaluate(model, 'Best PPO')
    save_results(model, model_path, after)
    if before is not None:
        compare_results(before, after)
    return 0


if __name__ == '__main__':
    sys.exit(main())
