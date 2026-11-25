"""Train stage-one SAC driving on varied roads with speed and road-rule rewards."""

from collections import deque
from datetime import datetime
import logging
from pathlib import Path
import subprocess
import sys
import time


PROJECT_DIR = Path(__file__).resolve().parent

# Run directly in PyCharm. Edit these settings before running.
RUN_MODE = 'train'  # 'train' or 'evaluate'
RESUME_MODEL = None  # Path to sac_model.zip for evaluation or continued training.
TOTAL_TIMESTEPS = 100_000  # Additional environment steps when resuming.

# Stage one: straight/curve combinations, no traffic or traffic lights.
# S = straight; C = a curve with a seeded direction, radius, and angle.
MAP_LAYOUTS = ('S', 'SS', 'SC', 'CS', 'SCS', 'CSC')
SPEED_LIMIT_KM_H = 40.0
MAX_EPISODE_STEPS = 1_000
TRAIN_START_SEED = 2_000
TRAIN_MAP_COUNT = 1_000
TEST_START_SEED = 1_000
TEST_MAP_COUNT = 24  # Four held-out maps per layout.
RANDOM_SEED = 42

# Reward settings. The speed limit is fixed across all stage-one roads.
PROGRESS_REWARD = 10.0  # Total progress reward for one complete route.
ARRIVAL_REWARD = 20.0
ROAD_FAILURE_PENALTY = 40.0
WRONG_WAY_PENALTY = 30.0
OVERSPEED_PENALTY_PER_SECOND = 5.0
TIME_PENALTY_PER_SECOND = 0.05
WRONG_WAY_SPEED_M_S = 0.5

# SAC hyperparameters: both steering and throttle/brake are learned.
LEARNING_RATE = 3e-4
BUFFER_SIZE = 100_000
LEARNING_STARTS = 2_000
BATCH_SIZE = 256
DISCOUNT_FACTOR = 0.999
TAU = 0.005
POLICY_LAYERS = (256, 256)
DEVICE = 'cpu'
CPU_THREADS = 1
PROGRESS_INTERVAL = 5_000
RENDER_EVALUATION = False
OUTPUT_DIR = PROJECT_DIR / 'demo_outputs' / 'sac_training'

# Avoid shadowing the installed package with the local MetaDrive checkout.
_original_path = sys.path.copy()
sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != PROJECT_DIR]
try:
    import gymnasium as gym
    import numpy as np
    import torch
    from metadrive import MetaDriveEnv
    from metadrive.component.map.pg_map import MapGenerateMethod
    from metadrive.component.road_network import Road
    from metadrive.policy.env_input_policy import EnvInputPolicy
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    # Reuse the working interpreter already used by metadrive_demo.py.
    python = Path(r'D:\conda-envs\metadrive\python.exe')
    if __name__ == '__main__' and python.is_file() and python.resolve() != Path(sys.executable).resolve():
        print(f'Using the existing MetaDrive environment: {python}', flush=True)
        try:
            raise SystemExit(subprocess.call([str(python), '-B', str(Path(__file__).resolve()), *sys.argv[1:]]))
        except KeyboardInterrupt:
            raise SystemExit(130)
    raise
finally:
    sys.path[:] = _original_path


def stage_one_settings():
    """Save reward semantics alongside the model and replay buffer."""
    return dict(
        version=1,  # Increment when changing the reward formula or rule definitions.
        layouts=list(MAP_LAYOUTS),
        speed_limit_km_h=SPEED_LIMIT_KM_H,
        progress_reward=PROGRESS_REWARD,
        arrival_reward=ARRIVAL_REWARD,
        road_failure_penalty=ROAD_FAILURE_PENALTY,
        wrong_way_penalty=WRONG_WAY_PENALTY,
        overspeed_penalty_per_second=OVERSPEED_PENALTY_PER_SECOND,
        time_penalty_per_second=TIME_PENALTY_PER_SECOND,
        wrong_way_speed_m_s=WRONG_WAY_SPEED_M_S,
    )


def environment_config():
    """Use default state observations and continuous controls with stage-one rules."""
    return dict(
        use_render=False,
        image_observation=False,
        agent_policy=EnvInputPolicy,
        manual_control=False,
        discrete_action=False,
        map='S',  # The seeded reset hook selects the actual sequence.
        stage_one=stage_one_settings(),
        traffic_density=0.0,
        accident_prob=0.0,
        random_lane_width=False,
        random_lane_num=False,
        random_spawn_lane_index=False,
        start_seed=TRAIN_START_SEED,
        num_scenarios=TRAIN_MAP_COUNT,
        horizon=MAX_EPISODE_STEPS,
        truncate_as_terminate=False,
        out_of_road_done=True,
        out_of_route_done=False,
        on_continuous_line_done=True,
        on_broken_line_done=False,
        store_map=False,
        log_level=logging.ERROR,
    )


class StageOneEnv(MetaDriveEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update(dict(stage_one=stage_one_settings()))
        return config

    def reset(self, seed=None):
        # MetaDrive calls reward_function once during reset; initialize its baseline there.
        self.previous_progress = None
        self.overspeed_steps = 0
        return super().reset(seed=seed)

    def _reset_global_seed(self, force_seed=None):
        super()._reset_global_seed(force_seed)
        layouts = self.config['stage_one']['layouts']
        self.active_layout = layouts[self.current_seed % len(layouts)]
        self.config['map_config']['type'] = MapGenerateMethod.BIG_BLOCK_SEQUENCE
        self.config['map_config']['config'] = self.active_layout

    def rule_status(self, vehicle):
        settings = self.config['stage_one']
        reference = vehicle.navigation.current_ref_lanes[0]
        longitude, _ = reference.local_coordinates(vehicle.position)
        angle = reference.heading_theta_at(longitude)
        tangent = np.array([np.cos(angle), np.sin(angle)])
        forward_speed = float(np.dot(vehicle.velocity, tangent))
        opposing_lane = vehicle.on_lane and Road(*vehicle.lane.index[:2]).is_negative_road()
        wrong_way = bool(opposing_lane or forward_speed < -settings['wrong_way_speed_m_s'])
        out_of_road = bool(self._is_out_of_road(vehicle))
        collision = any(bool(getattr(vehicle, key)) for key in (
            'crash_vehicle', 'crash_object', 'crash_building', 'crash_human', 'crash_sidewalk',
        ))
        arrived = bool(self._is_arrive_destination(vehicle)) and not (out_of_road or wrong_way or collision)
        speed = float(vehicle.speed_km_h)
        return dict(
            arrive_dest=arrived,
            out_of_road=out_of_road,
            wrong_way=wrong_way,
            collision=collision,
            speed_km_h=speed,
            speed_limit_km_h=float(settings['speed_limit_km_h']),
            overspeed=speed > settings['speed_limit_km_h'],
            forward_speed_m_s=forward_speed,
            route_completion=float(np.clip(vehicle.navigation.route_completion, 0.0, 1.0)),
            map_layout=self.active_layout,
        )

    def reward_function(self, vehicle_id):
        settings = self.config['stage_one']
        info = self.rule_status(self.agents[vehicle_id])
        progress = info['route_completion']
        is_reset = self.previous_progress is None
        delta_progress = 0.0 if is_reset else progress - self.previous_progress
        self.previous_progress = progress
        dt = 0.0 if is_reset else self.config['physics_world_step_size'] * self.config['decision_repeat']

        progress_reward = settings['progress_reward'] * delta_progress
        speed_cost = settings['overspeed_penalty_per_second'] * dt * max(
            0.0, info['speed_km_h'] / settings['speed_limit_km_h'] - 1.0,
        )
        time_cost = settings['time_penalty_per_second'] * dt
        terminal_reward = 0.0
        if info['out_of_road'] or info['collision']:
            terminal_reward = -settings['road_failure_penalty']
        elif info['wrong_way']:
            terminal_reward = -settings['wrong_way_penalty']
        elif info['arrive_dest']:
            terminal_reward = settings['arrival_reward']

        # A road-rule failure replaces the step reward; arriving adds the goal bonus.
        if terminal_reward < 0:
            progress_reward = speed_cost = time_cost = 0.0
        reward = progress_reward - speed_cost - time_cost + terminal_reward
        if not is_reset:
            self.overspeed_steps += int(info['overspeed'])
        steps = self.episode_lengths[vehicle_id]
        info.update(
            reward_progress=progress_reward,
            reward_overspeed=-speed_cost,
            reward_time=-time_cost,
            reward_terminal=terminal_reward,
            step_reward=reward,
            episode_overspeed_fraction=self.overspeed_steps / max(steps, 1),
            episode_overspeed=bool(self.overspeed_steps),
        )
        return float(reward), info

    def done_function(self, vehicle_id):
        done, info = super().done_function(vehicle_id)
        status = self.rule_status(self.agents[vehicle_id])
        info.update(status)
        return bool(done or status['wrong_way'] or status['collision'] or status['out_of_road']), info


class TrainingMaps(gym.Wrapper):
    """Translate an RL random seed into sampling within the training map pool."""

    def __init__(self, env):
        super().__init__(env)
        self.rng = np.random.default_rng(RANDOM_SEED)

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError('MetaDrive does not support reset options.')
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        first = self.env.config['start_seed']
        count = self.env.config['num_scenarios']
        scenario = int(self.rng.integers(first, first + count))
        return self.env.reset(seed=scenario)


class TrainingProgress(BaseCallback):
    def __init__(self):
        super().__init__()
        self.recent = deque(maxlen=100)

    def _on_training_start(self):
        self.started_at = time.perf_counter()
        self.start_step = self.model.num_timesteps

    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info:
                self.recent.append(info['episode'])
        steps = self.num_timesteps - self.start_step
        if steps % PROGRESS_INTERVAL == 0:
            elapsed = time.perf_counter() - self.started_at
            mean_return = np.mean([row['r'] for row in self.recent]) if self.recent else 0.0
            success = np.mean([row['arrive_dest'] for row in self.recent]) if self.recent else 0.0
            overspeed = np.mean([row['episode_overspeed_fraction'] for row in self.recent]) if self.recent else 0.0
            print(
                f'Train {steps:,}/{TOTAL_TIMESTEPS:,} | '
                f'recent return {mean_return:+.2f} | recent success {success:.1%} | '
                f'mean episode overspeed {overspeed:.1%} | '
                f'{steps / max(elapsed, 0.001):.0f} steps/s',
                flush=True,
            )
        return True


def validate_settings():
    if RUN_MODE not in ('train', 'evaluate'):
        raise ValueError("RUN_MODE must be 'train' or 'evaluate'.")
    if RUN_MODE == 'evaluate' and RESUME_MODEL is None:
        raise ValueError('Set RESUME_MODEL to a saved sac_model.zip before evaluation.')
    if RESUME_MODEL is not None and not Path(RESUME_MODEL).is_file():
        raise FileNotFoundError(f'Model not found: {RESUME_MODEL}')
    for name in (
        'TOTAL_TIMESTEPS', 'BUFFER_SIZE', 'BATCH_SIZE', 'CPU_THREADS',
        'TRAIN_MAP_COUNT', 'TEST_MAP_COUNT', 'MAX_EPISODE_STEPS', 'PROGRESS_INTERVAL',
    ):
        if not isinstance(globals()[name], int) or globals()[name] <= 0:
            raise ValueError(f'{name} must be a positive integer.')
    if not MAP_LAYOUTS or len(set(MAP_LAYOUTS)) != len(MAP_LAYOUTS):
        raise ValueError('MAP_LAYOUTS must contain distinct, nonempty S/C sequences.')
    if any(not layout or set(layout) - {'S', 'C'} for layout in MAP_LAYOUTS):
        raise ValueError('Each map layout must contain only S and C.')
    if min(TRAIN_MAP_COUNT, TEST_MAP_COUNT) < len(MAP_LAYOUTS):
        raise ValueError('Both map pools must cover every map layout.')
    for name in (
        'SPEED_LIMIT_KM_H', 'PROGRESS_REWARD', 'ARRIVAL_REWARD', 'ROAD_FAILURE_PENALTY',
        'WRONG_WAY_PENALTY', 'OVERSPEED_PENALTY_PER_SECOND', 'WRONG_WAY_SPEED_M_S',
    ):
        if not np.isfinite(globals()[name]) or globals()[name] <= 0:
            raise ValueError(f'{name} must be finite and positive.')
    if not np.isfinite(TIME_PENALTY_PER_SECOND) or TIME_PENALTY_PER_SECOND < 0:
        raise ValueError('TIME_PENALTY_PER_SECOND must be finite and nonnegative.')
    if not 0 <= LEARNING_STARTS < BUFFER_SIZE:
        raise ValueError('LEARNING_STARTS must be nonnegative and smaller than BUFFER_SIZE.')
    if RESUME_MODEL is None and RUN_MODE == 'train' and TOTAL_TIMESTEPS <= LEARNING_STARTS:
        raise ValueError('TOTAL_TIMESTEPS must exceed LEARNING_STARTS to update the policy.')
    if max(TRAIN_START_SEED, TEST_START_SEED) < min(
        TRAIN_START_SEED + TRAIN_MAP_COUNT, TEST_START_SEED + TEST_MAP_COUNT,
    ):
        raise ValueError('Training and test map pools must not overlap.')


def train():
    config = environment_config()
    model = None
    if RESUME_MODEL is not None:
        model = SAC.load(RESUME_MODEL, device=DEVICE)
        if getattr(model, 'training_config', None) != config:
            raise ValueError(
                'This checkpoint uses different roads or rewards. Restore its settings to resume, '
                'or set RESUME_MODEL=None to train the current task with a fresh replay buffer.'
            )
        replay_path = Path(RESUME_MODEL).with_name('replay_buffer.pkl')
        if not replay_path.is_file():
            raise FileNotFoundError(f'Resuming SAC requires its replay buffer: {replay_path}')
        model.load_replay_buffer(replay_path)

    run_dir = OUTPUT_DIR / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    model_path = run_dir / 'sac_model.zip'
    interrupted = False
    env = Monitor(TrainingMaps(StageOneEnv(config)), info_keywords=(
        'arrive_dest', 'out_of_road', 'wrong_way', 'episode_overspeed_fraction',
    ))
    try:
        if model is None:
            model = SAC(
                'MlpPolicy', env,
                learning_rate=LEARNING_RATE,
                buffer_size=BUFFER_SIZE,
                learning_starts=LEARNING_STARTS,
                batch_size=BATCH_SIZE,
                tau=TAU,
                gamma=DISCOUNT_FACTOR,
                train_freq=1,
                gradient_steps=1,
                ent_coef='auto',
                policy_kwargs=dict(net_arch=list(POLICY_LAYERS)),
                seed=RANDOM_SEED,
                device=DEVICE,
                verbose=0,
            )
        else:
            model.set_env(env)
        model.training_config = config
        start_step = model.num_timesteps
        print(f'SAC stage one | layouts={MAP_LAYOUTS} | limit={SPEED_LIMIT_KM_H:g} km/h | device={DEVICE}')
        print(f'Additional steps: {TOTAL_TIMESTEPS:,} | starting step: {start_step:,}')
        print(f'Output directory: {run_dir}', flush=True)
        try:
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=TrainingProgress(),
                reset_num_timesteps=RESUME_MODEL is None,
            )
        except KeyboardInterrupt:
            interrupted = True
            print('\nTraining interrupted. Saving the current model and replay buffer.', flush=True)
        run_dir.mkdir(parents=True, exist_ok=False)
        model.save(model_path)
        model.save_replay_buffer(run_dir / 'replay_buffer.pkl')
        print(f'Completed steps: {model.num_timesteps - start_step:,} | total: {model.num_timesteps:,}')
        print(f'Gradient updates: {model._n_updates:,}')
        print(f'Saved model: {model_path}', flush=True)
    finally:
        env.close()
    return None if interrupted else model_path


def evaluate(model_path):
    model = SAC.load(model_path, device=DEVICE)
    if 'stage_one' not in getattr(model, 'training_config', {}):
        raise ValueError('Select a stage-one checkpoint trained with the new road-rule reward.')
    config = model.training_config.copy()
    first = config['start_seed']
    if max(first, TEST_START_SEED) < min(first + config['num_scenarios'], TEST_START_SEED + TEST_MAP_COUNT):
        raise ValueError('The selected test maps overlap this model\'s training maps.')
    config.update(start_seed=TEST_START_SEED, num_scenarios=TEST_MAP_COUNT, use_render=RENDER_EVALUATION)
    env = StageOneEnv(config)
    results = []
    print('\nHeld-out evaluation | deterministic actions | stage-one road-rule reward', flush=True)
    try:
        for seed in range(TEST_START_SEED, TEST_START_SEED + TEST_MAP_COUNT):
            observation, _ = env.reset(seed=seed)
            total_reward = 0.0
            overspeed_steps = steps = 0
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                overspeed_steps += int(info['overspeed'])
                if terminated or truncated:
                    break
            row = dict(
                seed=seed, layout=info['map_layout'], success=bool(info['arrive_dest']),
                out_of_road=info['out_of_road'], wrong_way=info['wrong_way'], collision=info['collision'],
                route=info['route_completion'], reward=total_reward, steps=steps,
                overspeed_steps=overspeed_steps,
                compliant_success=bool(info['arrive_dest']) and overspeed_steps == 0,
                timeout=bool(truncated and not terminated),
            )
            results.append(row)
            print(
                f'Seed {seed} | {row["layout"]} | success={row["success"]} | '
                f'off-road={row["out_of_road"]} | wrong-way={row["wrong_way"]} | '
                f'overspeed={overspeed_steps / steps:.1%} | route={row["route"]:.1%} | '
                f'return={total_reward:+.2f}', flush=True,
            )
    finally:
        env.close()
    for label, key in (
        ('Arrival rate', 'success'), ('Compliant arrival rate', 'compliant_success'),
        ('Out-of-road rate', 'out_of_road'), ('Wrong-way rate', 'wrong_way'),
        ('Collision rate', 'collision'), ('Timeout rate', 'timeout'),
    ):
        print(f'{label}: {np.mean([row[key] for row in results]):.1%}')
    overspeed_fraction = sum(row['overspeed_steps'] for row in results) / sum(row['steps'] for row in results)
    print(f'Overspeed time fraction: {overspeed_fraction:.1%}')
    print(f'Mean route completion: {np.mean([row["route"] for row in results]):.1%}')
    print(f'Mean return: {np.mean([row["reward"] for row in results]):+.2f}')
    return results


def main():
    validate_settings()
    torch.set_num_threads(CPU_THREADS)
    model_path = train() if RUN_MODE == 'train' else Path(RESUME_MODEL)
    if model_path is None:
        return 130
    evaluate(model_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
