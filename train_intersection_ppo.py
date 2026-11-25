from collections import deque
import logging
from pathlib import Path
import sys
import tempfile

PROJECT_DIR = Path(__file__).resolve().parent

# Run settings
RUN_MODE = 'evaluate'
RESUME_TRAINING = True
MODEL_PATH = PROJECT_DIR / 'models' / 'ppo_yellow.zip'
RESUME_MODEL = None
TRAINING_STAGES = (('green', 60_000), ('red_start', 100_000), ('signals', 102_400))
VALIDATION_INTERVAL = 20_480
VALIDATION_START_SEED = 2_000
VALIDATION_EPISODES = 200
STAGE_SUCCESS_TARGET = 1.0
RENDER_EVALUATION = False
RANDOM_SEED = 42

# Intersection and episodes
MAP_SEED = 0
APPROACH_LENGTH_M = 70.0
LANE_WIDTH_M = 3.5
STOP_LINE_MARGIN_M = 2.0
START_DISTANCE_M = (20.0, 45.0)  # Vehicle center to stop line
START_SPEED_KMH = (0.0, 20.0)
MAX_SPEED_KMH = 40.0
RED_SECONDS = 8.0
GREEN_SECONDS = 8.0
YELLOW_SECONDS = 3.0
EPISODE_SECONDS = 60.0
PHYSICS_DT = 0.02
DECISION_REPEAT = 5
WAIT_ZONE_M = 6.0
STOPPED_SPEED_KMH = 1.0
TRAIN_START_SEED = 0
TRAIN_SCENARIOS = 1_000
TRAIN_FOCUS_SEEDS = ()
TRAIN_FOCUS_PROBABILITY = 0.0
TEST_START_SEED = 10_000
TEST_EPISODES = 50

# Rewards
PROGRESS_REWARD = 10.0
SUCCESS_REWARD = 20.0
RED_LIGHT_PENALTY = 20.0
COLLISION_PENALTY = 50.0
ROAD_DEPARTURE_PENALTY = 20.0
TIMEOUT_PENALTY = 20.0
TIME_COST_PER_SECOND = 0.2
APPROACH_DISTANCE_M = 20.0
APPROACH_SPEED_KMH = 20.0
APPROACH_SPEED_PENALTY = 2.0
BRAKING_RISK_PENALTY = 2.0
BRAKING_DECELERATION = 3.0
BRAKING_MARGIN_M = 1.0
SPEED_TOLERANCE_KMH = 2.0
MAX_APPROACH_OVERSPEED_FRACTION = 0.05

# PPO
LEARNING_RATE = 3e-4
SIGNAL_LEARNING_RATE = 1e-4
ROLLOUT_STEPS = 2_048
BATCH_SIZE = 256
UPDATE_EPOCHS = 10
DISCOUNT_FACTOR = 0.999
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENTROPY_COEFFICIENT = 0.0
TARGET_KL = 0.02
SDE_SAMPLE_FREQ = 8
STEERING_LOG_STD = -4.0
THROTTLE_LOG_STD = -2.5
INITIAL_THROTTLE_MEAN = 0.2
POLICY_LAYERS = (128, 128)
DEVICE = 'cpu'
CPU_THREADS = 1
PROGRESS_INTERVAL = 10_000

# Package imports
_original_path = sys.path.copy()
sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != PROJECT_DIR]
try:
    import gymnasium as gym
    import numpy as np
    import torch
    from metadrive import MetaDriveEnv
    from metadrive.component.pgblock.first_block import FirstPGBlock
    from metadrive.component.pgblock.intersection import InterSection
    from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
    from metadrive.obs.state_obs import StateObservation
    from metadrive.policy.env_input_policy import EnvInputPolicy
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
finally:
    sys.path[:] = _original_path

OBSERVATION_NAMES = (
    'forward_speed', 'lateral_speed', 'lateral_offset', 'heading_sin', 'heading_cos',
    'goal_distance', 'stop_line_distance', 'red_light', 'phase_remaining',
    'route_progress', 'episode_remaining', 'last_steering', 'last_throttle', 'yellow_light',
)
COLLISION_KEYS = ('crash_vehicle', 'crash_object', 'crash_building', 'crash_human', 'crash_sidewalk')
TASK_VERSION = 'signal_intersection_v3'


# Configuration checks
def validate_settings():
    if RUN_MODE not in ('train', 'evaluate'):
        raise ValueError("RUN_MODE must be 'train' or 'evaluate'.")
    dt = PHYSICS_DT * DECISION_REPEAT
    if dt <= 0 or DECISION_REPEAT < 1:
        raise ValueError('Simulation time steps must be positive.')
    for duration in (RED_SECONDS, GREEN_SECONDS, YELLOW_SECONDS, EPISODE_SECONDS):
        if duration <= 0 or not np.isclose(duration / dt, round(duration / dt)):
            raise ValueError('Signal and episode durations must be positive multiples of the decision interval.')
    if EPISODE_SECONDS < 2 * (RED_SECONDS + GREEN_SECONDS + YELLOW_SECONDS):
        raise ValueError('Allow at least two signal cycles per episode.')
    approach = APPROACH_LENGTH_M - FirstPGBlock.ENTRANCE_LENGTH - STOP_LINE_MARGIN_M
    if not 5 < START_DISTANCE_M[0] <= START_DISTANCE_M[1] < approach:
        raise ValueError('Starting distances must fit before the intersection stop line.')
    if not 0 <= START_SPEED_KMH[0] <= START_SPEED_KMH[1] <= MAX_SPEED_KMH:
        raise ValueError('Invalid starting speed range.')
    if min(TRAIN_SCENARIOS, TEST_EPISODES, VALIDATION_EPISODES, VALIDATION_INTERVAL, BATCH_SIZE) <= 0:
        raise ValueError('Training and evaluation counts must be positive.')
    if any(stage not in ('green', 'red_start', 'signals') or steps <= 0 for stage, steps in TRAINING_STAGES):
        raise ValueError('Invalid curriculum stages or training budgets.')
    if VALIDATION_INTERVAL % ROLLOUT_STEPS:
        raise ValueError('VALIDATION_INTERVAL must contain complete rollouts.')
    if ROLLOUT_STEPS % BATCH_SIZE:
        raise ValueError('BATCH_SIZE must divide ROLLOUT_STEPS.')
    train_seeds = set(range(TRAIN_START_SEED, TRAIN_START_SEED + TRAIN_SCENARIOS))
    test_seeds = set(range(TEST_START_SEED, TEST_START_SEED + TEST_EPISODES))
    validation_seeds = set(range(VALIDATION_START_SEED, VALIDATION_START_SEED + VALIDATION_EPISODES))
    if train_seeds & test_seeds or train_seeds & validation_seeds or test_seeds & validation_seeds:
        raise ValueError('Training, validation and test scenario seeds must be disjoint.')
    if not 0 <= TRAIN_FOCUS_PROBABILITY <= 1 or not set(TRAIN_FOCUS_SEEDS) <= train_seeds:
        raise ValueError('Focus probability must be in [0, 1], and focus seeds must belong to the training pool.')
    if not 0 < APPROACH_SPEED_KMH <= MAX_SPEED_KMH or APPROACH_DISTANCE_M <= 0:
        raise ValueError('Invalid approach speed or distance.')
    if BRAKING_DECELERATION <= 0 or BRAKING_MARGIN_M < 0:
        raise ValueError('Invalid braking parameters.')


def environment_config(render=False):
    return dict(
        use_render=render,
        image_observation=False,
        agent_policy=EnvInputPolicy,
        agent_observation=StateObservation,
        manual_control=False,
        traffic_density=0.0,
        accident_prob=0.0,
        random_spawn_lane_index=False,
        start_seed=MAP_SEED,
        num_scenarios=1,
        map_config=dict(
            type='block_sequence', config='X', lane_num=1,
            lane_width=LANE_WIDTH_M, exit_length=APPROACH_LENGTH_M,
        ),
        agent_configs={'default_agent': dict(
            spawn_lane_index=(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 0),
            destination=InterSection.node(1, 1, 1),
            spawn_velocity_car_frame=True,
            max_speed_km_h=MAX_SPEED_KMH,
        )},
        vehicle_config=dict(
            enable_reverse=False,
            lidar=dict(num_lasers=0), show_navi_mark=False, show_dest_mark=True,
        ),
        physics_world_step_size=PHYSICS_DT,
        decision_repeat=DECISION_REPEAT,
        horizon=None,
        store_map=True,
        log_level=logging.ERROR,
    )


# Transition rewards
def transition_reward(delta_progress, dt, waiting, collision, red_violation, off_road, success, timeout,
                      approach_risk=0.0, braking_risk=0.0):
    events = (
        (collision, -COLLISION_PENALTY, 'collision'),
        (red_violation, -RED_LIGHT_PENALTY, 'red_light'),
        (off_road, -ROAD_DEPARTURE_PENALTY, 'off_road'),
        (success, SUCCESS_REWARD, 'success'),
        (timeout, -TIMEOUT_PENALTY, 'timeout'),
    )
    for active, value, reason in events:
        if active:
            return float(value), reason
    reward = PROGRESS_REWARD * delta_progress - (0.0 if waiting else TIME_COST_PER_SECOND * dt)
    reward -= dt * (APPROACH_SPEED_PENALTY * approach_risk + BRAKING_RISK_PENALTY * braking_risk)
    return float(reward), 'running'


def driving_risks(distance, speed, color, remaining, dt):
    if distance < 0:
        return 0.0, 0.0
    speed = max(0.0, speed)
    target = APPROACH_SPEED_KMH / 3.6
    approach = (max(0.0, speed - target) / target) ** 2 if distance <= APPROACH_DISTANCE_M else 0.0
    must_stop = color == 'red' or (color == 'yellow' and distance >= speed * remaining)
    required = BRAKING_MARGIN_M + speed * dt + speed ** 2 / (2 * BRAKING_DECELERATION)
    braking = float(np.clip((required - distance) / required, 0, 1)) if must_stop and speed > 0.1 else 0.0
    return approach, braking


# Signal-controlled driving
class SignalIntersectionEnv(gym.Wrapper):
    def __init__(self, render=False, training=False, stage='signals'):
        if stage not in ('green', 'red_start', 'signals'):
            raise ValueError('Unknown curriculum stage.')
        super().__init__(MetaDriveEnv(environment_config(render)))
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(len(OBSERVATION_NAMES),), dtype=np.float32)
        self.dt = PHYSICS_DT * DECISION_REPEAT
        self.red_steps = round(RED_SECONDS / self.dt)
        self.green_steps = round(GREEN_SECONDS / self.dt)
        self.yellow_steps = round(YELLOW_SECONDS / self.dt)
        self.max_steps = round(EPISODE_SECONDS / self.dt)
        self.sampler = np.random.default_rng(RANDOM_SEED)
        self.training = training
        self.stage = stage
        self.light = None
        self.stop_line_node = None
        self.finished = True

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError('Use a scenario seed to choose the initial conditions.')
        self._clear_signal()
        if seed is not None:
            self.sampler = np.random.default_rng(seed)
        if seed is not None and not self.training:
            self.scenario_seed = int(seed)
        elif self.training and TRAIN_FOCUS_SEEDS and self.sampler.random() < TRAIN_FOCUS_PROBABILITY:
            self.scenario_seed = int(self.sampler.choice(TRAIN_FOCUS_SEEDS))
        else:
            self.scenario_seed = int(self.sampler.integers(TRAIN_START_SEED, TRAIN_START_SEED + TRAIN_SCENARIOS))
        rng = np.random.default_rng(self.scenario_seed)
        distance = rng.uniform(*START_DISTANCE_M)
        speed = rng.uniform(*START_SPEED_KMH)
        self.phase_offset = int(rng.integers(self.red_steps + self.green_steps + self.yellow_steps))
        agent = self.env.config['agent_configs']['default_agent']
        agent['spawn_longitude'] = (
            APPROACH_LENGTH_M - FirstPGBlock.ENTRANCE_LENGTH - STOP_LINE_MARGIN_M - distance
        )
        agent['spawn_velocity'] = (speed / 3.6, 0.0)
        self.env.reset(seed=MAP_SEED)
        self.approach_lane = self.env.current_map.road_network.get_lane(agent['spawn_lane_index'])
        self.origin = np.asarray(self.approach_lane.position(0, 0))
        self.direction = np.asarray(self.approach_lane.direction)
        self.side = np.array([-self.direction[1], self.direction[0]])
        self.stop_s = self.approach_lane.length - STOP_LINE_MARGIN_M
        final_lane = self.env.vehicle.navigation.final_lane
        goal = np.asarray(final_lane.position(final_lane.length - 5.0, 0))
        self.goal_s = float((goal - self.origin) @ self.direction)
        self.start_s = self._position()[0]
        self.route_length = self.goal_s - self.start_s
        self.steps = 0
        self.total_reward = 0.0
        self.wait_seconds = 0.0
        self.approach_steps = 0
        self.approach_overspeed_steps = 0
        self.approach_peak_speed = 0.0
        self.entry_speed = None
        self.last_action = np.zeros(2)
        self.finished = False
        self.light = self.env.engine.spawn_object(
            BaseTrafficLight, lane=self.approach_lane,
            position=self.approach_lane.position(self.stop_s, 0),
            random_seed=self.scenario_seed,
        )
        self._update_light()
        if self.env.config['use_render']:
            self._draw_stop_line()
        return self._observe(), self._info('running', False, False, False, False, False, 0.0)

    def _position(self):
        relative = np.asarray(self.env.vehicle.position) - self.origin
        return float(relative @ self.direction), float(relative @ self.side)

    def _footprint(self):
        heading = np.asarray(self.env.vehicle.heading)
        cosine, sine = float(heading @ self.direction), float(heading @ self.side)
        vehicle = self.env.vehicle
        forward = 0.5 * (vehicle.LENGTH * abs(cosine) + vehicle.WIDTH * abs(sine))
        lateral = 0.5 * (vehicle.WIDTH * abs(cosine) + vehicle.LENGTH * abs(sine))
        return forward, lateral

    def stop_distance(self):
        return self.stop_s - self._position()[0] - self._footprint()[0]

    def route_progress(self):
        return float(np.clip((self._position()[0] - self.start_s) / self.route_length, 0.0, 1.0))

    def signal_state(self):
        if self.stage == 'green' or (self.stage == 'red_start' and self.steps >= self.red_steps):
            return 'green', GREEN_SECONDS
        if self.stage == 'red_start':
            return 'red', (self.red_steps - self.steps) * self.dt
        phase = (self.phase_offset + self.steps) % (self.red_steps + self.green_steps + self.yellow_steps)
        if phase < self.red_steps:
            return 'red', (self.red_steps - phase) * self.dt
        if phase < self.red_steps + self.green_steps:
            return 'green', (self.red_steps + self.green_steps - phase) * self.dt
        return 'yellow', (self.red_steps + self.green_steps + self.yellow_steps - phase) * self.dt

    def _update_light(self):
        color, _ = self.signal_state()
        {'red': self.light.set_red, 'green': self.light.set_green, 'yellow': self.light.set_yellow}[color]()

    def necessary_wait(self):
        color, _ = self.signal_state()
        return bool(color in ('red', 'yellow') and 0 <= self.stop_distance() <= WAIT_ZONE_M
                    and self.env.vehicle.speed_km_h <= STOPPED_SPEED_KMH)

    def _observe(self):
        s, lateral = self._position()
        velocity = np.asarray(self.env.vehicle.velocity)
        heading = np.asarray(self.env.vehicle.heading)
        color, remaining = self.signal_state()
        scale = MAX_SPEED_KMH / 3.6
        observation = [
            velocity @ self.direction / scale, velocity @ self.side / scale,
            lateral / (LANE_WIDTH_M / 2), heading @ self.side, heading @ self.direction,
            (self.goal_s - s) / (2 * APPROACH_LENGTH_M), self.stop_distance() / APPROACH_LENGTH_M,
            float(color == 'red'), remaining / max(RED_SECONDS, GREEN_SECONDS, YELLOW_SECONDS), self.route_progress(),
            (self.max_steps - self.steps) / self.max_steps, *self.last_action, float(color == 'yellow'),
        ]
        return np.clip(observation, -1, 1).astype(np.float32)

    def step(self, action):
        if self.finished:
            raise RuntimeError('Reset the environment before stepping a finished episode.')
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError('Actions must contain finite steering and throttle/brake values.')
        self.last_action = np.clip(action, -1, 1)
        color_during_step, remaining = self.signal_state()
        previous_distance = self.stop_distance()
        previous_progress = self.route_progress()
        waiting = self.necessary_wait()
        speed = max(0.0, float(np.asarray(self.env.vehicle.velocity) @ self.direction))
        approach_risk, braking_risk = driving_risks(previous_distance, speed, color_during_step, remaining, self.dt)
        if 0 <= previous_distance <= APPROACH_DISTANCE_M:
            if speed * 3.6 > STOPPED_SPEED_KMH:
                self.approach_steps += 1
                self.approach_overspeed_steps += int(speed * 3.6 > APPROACH_SPEED_KMH + SPEED_TOLERANCE_KMH)
            self.approach_peak_speed = max(self.approach_peak_speed, speed * 3.6)
        _, _, _, _, native_info = self.env.step(self.last_action)
        self.steps += 1
        current_distance = self.stop_distance()
        crossed = previous_distance >= 0 and current_distance < 0
        if crossed:
            self.entry_speed = float(self.env.vehicle.speed_km_h)
        red_violation = bool(color_during_step == 'red' and crossed)
        collision = any(bool(native_info.get(key, False)) for key in COLLISION_KEYS)
        s, lateral = self._position()
        # Assigned straight lane corridor
        off_road = bool(native_info.get('out_of_road', False)
                        or abs(lateral) + self._footprint()[1] > LANE_WIDTH_M / 2 + 0.05)
        success = bool(s >= self.goal_s and not (collision or red_violation or off_road))
        timeout = self.steps >= self.max_steps and not (success or collision or red_violation or off_road)
        reward, reason = transition_reward(
            self.route_progress() - previous_progress, self.dt, waiting,
            collision, red_violation, off_road, success, timeout, approach_risk, braking_risk,
        )
        self.total_reward += reward
        self.wait_seconds += self.dt if waiting else 0.0
        self.finished = reason != 'running'
        self._update_light()
        info = self._info(reason, success, red_violation, collision, off_road, timeout, reward)
        info['signal_red_during_step'] = color_during_step == 'red'
        info['signal_during_step'] = color_during_step
        info['approach_risk'] = approach_risk
        info['braking_risk'] = braking_risk
        info['necessary_wait'] = waiting
        if self.env.config['use_render']:
            self.render()
        # Observed task deadline
        return self._observe(), reward, self.finished, False, info

    def _info(self, reason, success, red_violation, collision, off_road, timeout, reward):
        color, remaining = self.signal_state()
        return dict(
            scenario_seed=self.scenario_seed, map_seed=MAP_SEED,
            curriculum_stage=self.stage,
            signal_red=color == 'red', signal_color=color, phase_remaining=remaining, stop_line_distance=self.stop_distance(),
            route_completion=self.route_progress(), is_success=success, arrive_dest=success,
            red_light_violation=red_violation, collision=collision, out_of_road=off_road,
            timeout=timeout, outcome=reason, step_reward=reward, episode_reward=self.total_reward,
            episode_length=self.steps, seconds=self.steps * self.dt, waiting_seconds=self.wait_seconds,
            approach_overspeed_fraction=self.approach_overspeed_steps / max(1, self.approach_steps),
            approach_peak_speed_kmh=self.approach_peak_speed, entry_speed_kmh=self.entry_speed,
        )

    def _draw_stop_line(self):
        from panda3d.core import LineSegs
        line = LineSegs()
        line.setColor(1, 1, 1, 1)
        line.setThickness(4)
        for method, lateral in ((line.moveTo, -LANE_WIDTH_M / 2), (line.drawTo, LANE_WIDTH_M / 2)):
            x, y = self.approach_lane.position(self.stop_s, lateral)
            method(float(x), float(y), 0.05)
        self.stop_line_node = self.env.engine.worldNP.attachNewNode(line.create())

    def render(self):
        color, remaining = self.signal_state()
        return self.env.render(text={
            'Scenario seed': str(self.scenario_seed),
            'Signal': f'{color.upper()} ({remaining:.1f}s)',
            'Total reward': f'{self.total_reward:+.2f}',
        })

    def _clear_signal(self):
        if self.stop_line_node is not None:
            self.stop_line_node.removeNode()
            self.stop_line_node = None
        if self.light is not None:
            self.env.engine.clear_objects([self.light.id], force_destroy=True)
            self.light = None

    def close(self):
        self._clear_signal()
        self.env.close()


# Terminal output
def print_reward_table():
    rows = (
        ('Route progress', f'{PROGRESS_REWARD:g} * progress change'),
        ('Successful arrival', f'+{SUCCESS_REWARD:g}'),
        ('Red-light violation', f'-{RED_LIGHT_PENALTY:g}'),
        ('Collision', f'-{COLLISION_PENALTY:g}'),
        ('Road / route departure', f'-{ROAD_DEPARTURE_PENALTY:g}'),
        ('Time cost outside necessary waiting', f'-{TIME_COST_PER_SECOND:g} / second'),
        ('Necessary red/yellow waiting', '0'),
        ('Approach overspeed', f'-{APPROACH_SPEED_PENALTY:g} * squared relative excess / second'),
        ('Insufficient braking distance', f'-{BRAKING_RISK_PENALTY:g} * risk / second'),
        ('Task deadline', f'-{TIMEOUT_PENALTY:g}'),
    )
    print('\nSignal Intersection Reward')
    print(f"{'Component':<40} Value")
    print('-' * 67)
    for component, value in rows:
        print(f'{component:<40} {value}')


class TrainingProgress(BaseCallback):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage
        self.episodes = deque(maxlen=100)

    def _on_step(self):
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if done:
                self.episodes.append(info)
        if self.n_calls % PROGRESS_INTERVAL == 0:
            if self.episodes:
                success = np.mean([row['is_success'] for row in self.episodes])
                red = np.mean([row['red_light_violation'] for row in self.episodes])
                timeout = np.mean([row['timeout'] for row in self.episodes])
                print(f'{self.stage} | Steps {self.num_timesteps:,} | Recent success {success:.1%} | '
                      f'Red {red:.1%} | Timeout {timeout:.1%}', flush=True)
            else:
                print(f'Steps {self.num_timesteps:,}', flush=True)
        return True


# Checkpoint settings
def task_settings():
    return dict(
        version=TASK_VERSION, observations=list(OBSERVATION_NAMES),
        map_seed=MAP_SEED, approach_length=APPROACH_LENGTH_M, lane_width=LANE_WIDTH_M,
        stop_margin=STOP_LINE_MARGIN_M, start_distance=list(START_DISTANCE_M),
        start_speed=list(START_SPEED_KMH), max_speed=MAX_SPEED_KMH,
        red_seconds=RED_SECONDS, green_seconds=GREEN_SECONDS, yellow_seconds=YELLOW_SECONDS, episode_seconds=EPISODE_SECONDS,
        physics_dt=PHYSICS_DT, decision_repeat=DECISION_REPEAT,
        wait_zone=WAIT_ZONE_M, stopped_speed=STOPPED_SPEED_KMH,
        approach_distance=APPROACH_DISTANCE_M, approach_speed=APPROACH_SPEED_KMH,
        braking_deceleration=BRAKING_DECELERATION, braking_margin=BRAKING_MARGIN_M,
        speed_tolerance=SPEED_TOLERANCE_KMH,
        rewards=dict(progress=PROGRESS_REWARD, success=SUCCESS_REWARD, red=RED_LIGHT_PENALTY,
                     collision=COLLISION_PENALTY, departure=ROAD_DEPARTURE_PENALTY,
                     timeout=TIMEOUT_PENALTY, time_cost=TIME_COST_PER_SECOND,
                     approach=APPROACH_SPEED_PENALTY, braking=BRAKING_RISK_PENALTY),
    )


def ppo_settings(stage='signals'):
    return dict(
        learning_rate=LEARNING_RATE if stage == 'green' else SIGNAL_LEARNING_RATE,
        n_steps=ROLLOUT_STEPS, batch_size=BATCH_SIZE,
        n_epochs=UPDATE_EPOCHS, gamma=DISCOUNT_FACTOR, gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE, ent_coef=ENTROPY_COEFFICIENT, target_kl=TARGET_KL,
        use_sde=True, sde_sample_freq=SDE_SAMPLE_FREQ,
        policy_kwargs=dict(net_arch=dict(pi=list(POLICY_LAYERS), vf=list(POLICY_LAYERS)),
                           squash_output=True, log_std_init=THROTTLE_LOG_STD),
    )


def load_model(env=None, stage='signals', for_training=False, path=None):
    path = Path(path) if path is not None else MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(f'Model not found: {path}')
    overrides = ppo_settings(stage) if env is not None or for_training else {}
    model = PPO.load(str(path), device=DEVICE, **overrides)
    if not hasattr(model, 'policy_training_rewards'):
        model.policy_training_rewards = dict(getattr(model, 'intersection_settings', {}).get('rewards', {}))
    model.transferred_this_run = False
    if getattr(model, 'intersection_settings', None) != task_settings():
        previous = getattr(model, 'intersection_settings', {})
        current = task_settings()
        same_task = {k: v for k, v in previous.items() if k != 'rewards'} == {
            k: v for k, v in current.items() if k != 'rewards'
        }
        if for_training and same_task:
            model.previous_reward_settings = previous['rewards']
            model.intersection_settings = current
            print('Continuing the same task with updated reward weights.', flush=True)
        elif for_training and previous.get('version') == 'signal_intersection_v2':
            model = transfer_yellow_policy(model, stage)
        else:
            raise ValueError('The checkpoint uses different task settings. Match its settings before loading it.')
    if env is not None:
        model.set_env(env)
    return model


# Yellow observation transfer
def transfer_yellow_policy(source, stage):
    previous = source.intersection_settings
    current = task_settings()
    if previous['observations'] != list(OBSERVATION_NAMES[:-1]):
        raise ValueError('Unexpected source observations.')
    for name, value in previous.items():
        if name not in ('version', 'observations', 'rewards') and current.get(name) != value:
            raise ValueError(f'Source task differs in {name}.')
    for name, value in previous['rewards'].items():
        if current['rewards'].get(name) != value:
            raise ValueError(f'Source reward differs in {name}.')
    env = SignalIntersectionEnv(stage=stage)
    try:
        model = PPO('MlpPolicy', env, **ppo_settings(stage), seed=RANDOM_SEED, device=DEVICE, verbose=0)
        weights = source.policy.state_dict()
        for key in ('mlp_extractor.policy_net.0.weight', 'mlp_extractor.value_net.0.weight'):
            weights[key] = torch.cat((weights[key], torch.zeros_like(weights[key][:, :1])), dim=1)
        model.policy.load_state_dict(weights)
        model.policy.reset_noise()
    finally:
        env.close()
    model.intersection_settings = current
    model.num_timesteps = source.num_timesteps
    model.training_env_steps = getattr(source, 'training_env_steps', source.num_timesteps)
    model.completed_stages = [s for s in getattr(source, 'completed_stages', []) if s != 'signals']
    model.curriculum_history = []
    model.source_task_version = previous['version']
    model.policy_training_rewards = dict(previous['rewards'])
    model.transferred_this_run = True
    print('Transferred driving weights; added one yellow-light input; reset optimizer.', flush=True)
    return model


# Validation ranking
def evaluation_score(rows):
    success = float(np.mean([row['is_success'] for row in rows]))
    violations = float(np.mean([
        row['red_light_violation'] or row['collision'] or row['out_of_road'] for row in rows
    ]))
    successful_times = [row['seconds'] for row in rows if row['is_success']]
    mean_time = float(np.mean(successful_times)) if successful_times else EPISODE_SECONDS
    overspeed = float(np.mean([row['approach_overspeed_fraction'] for row in rows]))
    return success, -violations, -overspeed, -mean_time


# Training and evaluation
def train():
    if MODEL_PATH.exists() and not RESUME_TRAINING:
        raise FileExistsError('A model already exists. Set RESUME_TRAINING=True or choose another MODEL_PATH.')
    source_path = RESUME_MODEL if RESUME_MODEL is not None else MODEL_PATH
    model = load_model(for_training=True, path=source_path) if RESUME_TRAINING else None
    completed = list(getattr(model, 'completed_stages', []))
    history = list(getattr(model, 'curriculum_history', []))
    total_steps = int(getattr(model, 'training_env_steps', 0))
    print('Driver: self-trained SB3 PPO; steering and throttle/brake are learned.', flush=True)
    print('Curriculum: green -> red_start -> signals. Final evaluation always uses signals.', flush=True)
    if model is not None and getattr(model, 'transferred_this_run', False):
        print('\nSource policy on the new yellow-light task:', flush=True)
        model.baseline_test_results = evaluate(model)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='intersection_curriculum_') as temporary:
        best_path = Path(temporary) / 'stage_best.zip'
        for stage, budget in TRAINING_STAGES:
            if stage in completed and stage != 'signals':
                continue
            best_score = None
            best_rows = None
            stage_steps = 0
            if model is not None:
                best_rows = evaluate(model, stage, VALIDATION_START_SEED, VALIDATION_EPISODES, verbose=False)
                best_score = evaluation_score(best_rows)
                print(f'Starting validation {stage}: {sum(row["is_success"] for row in best_rows)}/{len(best_rows)}', flush=True)
                model.save(str(best_path))
            while stage_steps < budget:
                env = Monitor(SignalIntersectionEnv(training=True, stage=stage))
                env.env.sampler = np.random.default_rng(RANDOM_SEED + total_steps)
                try:
                    if model is None:
                        model = PPO('MlpPolicy', env, **ppo_settings(stage), seed=RANDOM_SEED, device=DEVICE, verbose=0)
                        with torch.no_grad():
                            model.policy.log_std[:, 0].fill_(STEERING_LOG_STD)
                            model.policy.log_std[:, 1].fill_(THROTTLE_LOG_STD)
                            model.policy.action_net.bias[1] = INITIAL_THROTTLE_MEAN
                        model.policy.reset_noise()
                    else:
                        model.set_env(env)
                        rate = LEARNING_RATE if stage == 'green' else SIGNAL_LEARNING_RATE
                        model.learning_rate = rate
                        model.lr_schedule = lambda _: rate
                    model.intersection_settings = task_settings()
                    model.training_scenario_seeds = (TRAIN_START_SEED, TRAIN_START_SEED + TRAIN_SCENARIOS - 1)
                    model.training_focus_seeds = list(TRAIN_FOCUS_SEEDS)
                    model.training_focus_probability = TRAIN_FOCUS_PROBABILITY
                    model.resume_source = str(source_path) if RESUME_TRAINING else None
                    before = model.num_timesteps
                    chunk = min(VALIDATION_INTERVAL, budget - stage_steps)
                    model.learn(chunk, callback=TrainingProgress(stage), reset_num_timesteps=False)
                    model.policy_training_rewards = dict(task_settings()['rewards'])
                    used = model.num_timesteps - before
                    stage_steps += used
                    total_steps += used
                finally:
                    env.close()
                rows = evaluate(model, stage, VALIDATION_START_SEED, VALIDATION_EPISODES, verbose=False)
                success = float(np.mean([row['is_success'] for row in rows]))
                score = evaluation_score(rows)
                history.append(dict(stage=stage, selected_step=model.num_timesteps,
                                    trained_steps=total_steps, success=success, validation_count=len(rows),
                                    approach_weight=APPROACH_SPEED_PENALTY, overspeed_fraction=-score[2]))
                print(f'Validation {stage}: {sum(row["is_success"] for row in rows)}/{len(rows)} | '
                      f'Red {sum(row["red_light_violation"] for row in rows)} | '
                      f'Off-road {sum(row["out_of_road"] for row in rows)} | '
                      f'Timeout {sum(row["timeout"] for row in rows)} | '
                      f'Approach overspeed {-score[2]:.1%} | Mean successful time {-score[3]:.1f}s | '
                      f'Trained steps {total_steps:,}', flush=True)
                if best_score is None or score > best_score:
                    best_score, best_rows = score, rows
                    model.curriculum_history = history.copy()
                    model.training_env_steps = total_steps
                    model.completed_stages = completed.copy()
                    model.selected_stage = stage
                    model.validation_results = rows
                    model.save(str(best_path))
                    model.save(str(MODEL_PATH))
                if success >= STAGE_SUCCESS_TARGET and -score[2] <= MAX_APPROACH_OVERSPEED_FRACTION:
                    break
            model = PPO.load(str(best_path), device=DEVICE)
            if best_score[0] >= STAGE_SUCCESS_TARGET and -best_score[2] <= MAX_APPROACH_OVERSPEED_FRACTION and stage not in completed:
                completed.append(stage)
            model.completed_stages = completed.copy()
            model.curriculum_history = history.copy()
            model.training_env_steps = total_steps
            model.selected_stage = stage
            model.validation_results = best_rows
            model.save(str(MODEL_PATH))
            print(f'Selected {stage} model saved: {MODEL_PATH}', flush=True)
            if stage != 'signals' and stage not in completed:
                print(f'{stage} has not reached the stage target; keeping this stage for the next run.', flush=True)
                break
    return model


def evaluate(model, stage='signals', first_seed=TEST_START_SEED, episodes=TEST_EPISODES, verbose=True):
    env = SignalIntersectionEnv(render=RENDER_EVALUATION and verbose, stage=stage)
    results = []
    if verbose:
        print(f"\n{'Seed':>7} {'Outcome':>12} {'Route':>8} {'Reward':>9} {'Time(s)':>9} {'Wait(s)':>9}")
    try:
        for seed in range(first_seed, first_seed + episodes):
            observation, _ = env.reset(seed=seed)
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            results.append(info)
            if verbose:
                print(f"{seed:7d} {info['outcome']:>12} {info['route_completion']:7.1%} "
                      f"{info['episode_reward']:9.2f} {info['seconds']:9.1f} {info['waiting_seconds']:9.1f}", flush=True)
            elif len(results) % 50 == 0:
                print(f'Validation progress: {len(results)}/{episodes}', flush=True)
    finally:
        env.close()
    if not verbose:
        return results
    for label, key in (('Successful arrival', 'is_success'), ('Red-light violations', 'red_light_violation'),
                       ('Collisions', 'collision'), ('Road / route departures', 'out_of_road'), ('Timeouts', 'timeout')):
        count = sum(row[key] for row in results)
        print(f'{label}: {count}/{len(results)} ({count / len(results):.1%})')
    successful = [row['seconds'] for row in results if row['is_success']]
    if successful:
        print(f'Mean successful travel time: {np.mean(successful):.1f} s')
    entries = [row['entry_speed_kmh'] for row in results if row['entry_speed_kmh'] is not None and row['is_success']]
    if entries:
        print(f'Mean successful entry speed: {np.mean(entries):.1f} km/h')
    print(f'Mean moving-approach overspeed fraction (above {APPROACH_SPEED_KMH + SPEED_TOLERANCE_KMH:g} km/h): '
          f'{np.mean([row["approach_overspeed_fraction"] for row in results]):.1%}')
    return results


def main():
    validate_settings()
    torch.set_num_threads(CPU_THREADS)
    print_reward_table()
    model = train() if RUN_MODE == 'train' else load_model()
    rows = evaluate(model)
    if RUN_MODE == 'train':
        model.test_results = rows
        model.save(str(MODEL_PATH))


if __name__ == '__main__':
    main()
