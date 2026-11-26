from collections import deque
import hashlib
import json
import logging
from pathlib import Path
import sys
import tempfile

PROJECT_DIR = Path(__file__).resolve().parent

# Run
RUN_MODE = 'evaluate'
RESUME_TRAINING = True
EXPERIMENT = 'baseline'
MODEL_NAME = {'baseline': 'ppo_yellow', 'control': 'ppo_pace_control', 'passed_signal': 'ppo_passed_signal',
              'exit_recovery': 'ppo_exit_recovery', 'exit_recovery_masked': 'ppo_exit_recovery',
              'exit_recovery_pair': 'ppo_exit_recovery', 'green_entry': 'ppo_green_entry'}[EXPERIMENT]
EXIT_RECOVERY_TRIAL = EXPERIMENT in ('exit_recovery', 'exit_recovery_masked', 'exit_recovery_pair', 'green_entry')
MASK_PASSED_SIGNAL = EXPERIMENT in ('passed_signal', 'green_entry')
MODEL_PATH = PROJECT_DIR / 'models' / f'{MODEL_NAME}.zip'
RESULTS_PATH = PROJECT_DIR / 'results' / f'{MODEL_NAME}.json'
PAIR_RESULTS_PATH = PROJECT_DIR / 'results' / 'ppo_exit_recovery_pair.json'
SOURCE_NAME = 'ppo_exit_recovery.zip' if EXPERIMENT == 'green_entry' else 'ppo_yellow_speed_candidate.zip'
RESUME_MODEL = (PROJECT_DIR / 'models' / SOURCE_NAME
                if EXPERIMENT != 'baseline' and not MODEL_PATH.exists() else None)
RENDER_EVALUATION = False
RANDOM_SEED = 42

# Training and evaluation
TRIAL_STEPS = {'passed_signal': 10_240, 'green_entry': 20_480}.get(EXPERIMENT, 40_960)
TRAINING_STAGES = (('signals', TRIAL_STEPS),) if EXPERIMENT != 'baseline' else (
    ('green', 122_880), ('red_start', 122_880), ('signals', 204_800))
VALIDATION_INTERVAL = 10_240 if EXPERIMENT != 'baseline' else 20_480
VALIDATION_START_SEED = 2_000
VALIDATION_EPISODES = 200
STAGE_SUCCESS_TARGET = 1.0
CURRICULUM_SPEED_SUCCESS_TARGET = 0.99
MIN_VALIDATION_SUCCESS_RATE = 0.98

# Road
MAP_SEED = 0
MAP_LAYOUT = 'X'
MAP_SCENARIOS = 1
LANE_COUNT = 1
SPAWN_LANE_INDEX = 0
DESTINATION_NODE = (1, 1, 1)
TRAFFIC_DENSITY = 0.0
ACCIDENT_PROBABILITY = 0.0
LIDAR_RAYS = 0
APPROACH_LENGTH_M = 70.0
LANE_WIDTH_M = 3.5
LANE_EDGE_TOLERANCE_M = 0.05
STOP_LINE_MARGIN_M = 2.0
GOAL_MARGIN_M = 5.0
START_CLEARANCE_M = 5.0
START_DISTANCE_M = (20.0, 45.0)
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

# Scenarios
TRAIN_START_SEED = 0
TRAIN_SCENARIOS = 1_000
TEST_START_SEED = 10_000
TEST_EPISODES = 50
HOLDOUT_START_SEED = 30_000
HOLDOUT_EPISODES = 100

# Rewards
PROGRESS_REWARD = 10.0
SUCCESS_REWARD = 20.0
RED_LIGHT_PENALTY = 20.0
COLLISION_PENALTY = 50.0
ROAD_DEPARTURE_PENALTY = 20.0
TIMEOUT_PENALTY = 20.0
TIME_COST_PER_SECOND = 0.2
APPROACH_DISTANCE_M = 20.0
APPROACH_SPEED_KMH = 30.0
EXIT_RECOVERY_SPEED_KMH = 25.0
EXIT_SLOW_COST_PER_SECOND = 0.6
GREEN_MIN_KMH = 25.0
GREEN_START_GRACE_S = 3.0
GREEN_REMAINING_S = 2.0
GREEN_ENTRY_BONUS = 4.0
GREEN_BONUS_TAPER_KMH = 10.0
GREEN_VALIDATION_SUCCESS = 0.995
APPROACH_SPEED_PENALTY = 8.0
ENTRY_SPEED_PENALTY_PER_KMH = 1.0
BRAKING_RISK_PENALTY = 2.0
BRAKING_DECELERATION = 3.0
BRAKING_MARGIN_M = 1.0
BRAKING_MIN_SPEED_MPS = 0.1
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
RECENT_EPISODES = 100
VALIDATION_LOG_INTERVAL = 50

# Display and units
STOP_LINE_COLOR = (1, 1, 1, 1)
STOP_LINE_WIDTH = 4
STOP_LINE_HEIGHT_M = 0.05
REWARD_COLUMN_WIDTH = 40
REWARD_TABLE_WIDTH = 67
KMH_PER_MPS = 3.6
ACTION_SIZE = 2

# Dependencies
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
TRAINING_OBJECTIVE_KEYS = ('rewards', 'approach_speed')


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
    if not START_CLEARANCE_M < START_DISTANCE_M[0] <= START_DISTANCE_M[1] < approach:
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
    if EXPERIMENT in ('exit_recovery_pair', 'green_entry'):
        holdout = set(range(HOLDOUT_START_SEED, HOLDOUT_START_SEED + HOLDOUT_EPISODES))
        if not holdout or holdout & (train_seeds | test_seeds | validation_seeds):
            raise ValueError('Holdout seeds must be nonempty and disjoint from the other splits.')
    if EXPERIMENT == 'green_entry' and not (
        0 < GREEN_MIN_KMH <= APPROACH_SPEED_KMH + SPEED_TOLERANCE_KMH <= MAX_SPEED_KMH
        and 0 < GREEN_REMAINING_S <= GREEN_SECONDS - GREEN_START_GRACE_S <= GREEN_SECONDS
        and GREEN_ENTRY_BONUS >= 0 and GREEN_BONUS_TAPER_KMH > 0 and 0 < GREEN_VALIDATION_SUCCESS <= 1
    ):
        raise ValueError('Invalid stable-green entry settings.')
    if not 0 < APPROACH_SPEED_KMH <= MAX_SPEED_KMH or APPROACH_DISTANCE_M <= 0:
        raise ValueError('Invalid approach speed or distance.')
    if BRAKING_DECELERATION <= 0 or BRAKING_MARGIN_M < 0:
        raise ValueError('Invalid braking parameters.')
    if EXIT_RECOVERY_TRIAL and not (0 < EXIT_RECOVERY_SPEED_KMH <= MAX_SPEED_KMH and EXIT_SLOW_COST_PER_SECOND >= 0):
        raise ValueError('Invalid exit speed or penalty.')
    if ENTRY_SPEED_PENALTY_PER_KMH < 0 or not 0 < MIN_VALIDATION_SUCCESS_RATE <= STAGE_SUCCESS_TARGET <= 1:
        raise ValueError('Invalid entry penalty or validation success targets.')
    if not 0 < CURRICULUM_SPEED_SUCCESS_TARGET <= STAGE_SUCCESS_TARGET:
        raise ValueError('Invalid intermediate curriculum target.')


def environment_config(render=False):
    return dict(
        use_render=render,
        image_observation=False,
        agent_policy=EnvInputPolicy,
        agent_observation=StateObservation,
        manual_control=False,
        traffic_density=TRAFFIC_DENSITY,
        accident_prob=ACCIDENT_PROBABILITY,
        random_spawn_lane_index=False,
        start_seed=MAP_SEED,
        num_scenarios=MAP_SCENARIOS,
        map_config=dict(
            type='block_sequence', config=MAP_LAYOUT, lane_num=LANE_COUNT,
            lane_width=LANE_WIDTH_M, exit_length=APPROACH_LENGTH_M,
        ),
        agent_configs={'default_agent': dict(
            spawn_lane_index=(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, SPAWN_LANE_INDEX),
            destination=InterSection.node(*DESTINATION_NODE),
            spawn_velocity_car_frame=True,
            max_speed_km_h=MAX_SPEED_KMH,
        )},
        vehicle_config=dict(
            enable_reverse=False,
            lidar=dict(num_lasers=LIDAR_RAYS), show_navi_mark=False, show_dest_mark=True,
        ),
        physics_world_step_size=PHYSICS_DT,
        decision_repeat=DECISION_REPEAT,
        horizon=None,
        store_map=True,
        log_level=logging.ERROR,
    )


# Rewards
def transition_reward(delta_progress, dt, waiting, collision, red_violation, off_road, success, timeout,
                      approach_risk=0.0, braking_risk=0.0, entry_speed_kmh=None):
    events = (
        (collision, -COLLISION_PENALTY, 'collision'),
        (red_violation, -RED_LIGHT_PENALTY, 'red_light'),
        (off_road, -ROAD_DEPARTURE_PENALTY, 'off_road'),
        (success, SUCCESS_REWARD, 'success'),
        (timeout, -TIMEOUT_PENALTY, 'timeout'),
    )
    for active, value, reason in events:
        if active:
            if reason == 'success':
                value -= entry_speed_cost(entry_speed_kmh)
            return float(value), reason
    reward = PROGRESS_REWARD * delta_progress - (0.0 if waiting else TIME_COST_PER_SECOND * dt)
    reward -= dt * (APPROACH_SPEED_PENALTY * approach_risk + BRAKING_RISK_PENALTY * braking_risk)
    reward -= entry_speed_cost(entry_speed_kmh)
    return float(reward), 'running'


def entry_speed_cost(speed_kmh):
    return 0.0 if speed_kmh is None else ENTRY_SPEED_PENALTY_PER_KMH * max(0.0, speed_kmh - APPROACH_SPEED_KMH)


def driving_risks(distance, speed, color, remaining, dt):
    if distance < 0:
        return 0.0, 0.0
    speed = max(0.0, speed)
    target = APPROACH_SPEED_KMH / KMH_PER_MPS
    approach = (max(0.0, speed - target) / target) ** 2 if distance <= APPROACH_DISTANCE_M else 0.0
    must_stop = color == 'red' or (color == 'yellow' and distance >= speed * remaining)
    required = BRAKING_MARGIN_M + speed * dt + speed ** 2 / (2 * BRAKING_DECELERATION)
    braking = float(np.clip((required - distance) / required, 0, 1)) if must_stop and speed > BRAKING_MIN_SPEED_MPS else 0.0
    return approach, braking


# Environment
class SignalIntersectionEnv(gym.Wrapper):
    def __init__(self, render=False, training=False, stage='signals', mask_passed_signal=None):
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
        self.mask_passed_signal = MASK_PASSED_SIGNAL if mask_passed_signal is None else mask_passed_signal
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
        agent['spawn_velocity'] = (speed / KMH_PER_MPS, 0.0)
        self.env.reset(seed=MAP_SEED)
        self.approach_lane = self.env.current_map.road_network.get_lane(agent['spawn_lane_index'])
        self.origin = np.asarray(self.approach_lane.position(0, 0))
        self.direction = np.asarray(self.approach_lane.direction)
        self.side = np.array([-self.direction[1], self.direction[0]])
        self.stop_s = self.approach_lane.length - STOP_LINE_MARGIN_M
        final_lane = self.env.vehicle.navigation.final_lane
        exit_start = np.asarray(final_lane.position(0, 0))
        self.exit_s = float((exit_start - self.origin) @ self.direction)
        goal = np.asarray(final_lane.position(final_lane.length - GOAL_MARGIN_M, 0))
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
        self.stable_green_entry = False
        self.entry_seconds = None
        self.exit_seconds = None
        self.exit_slow_cost = 0.0
        self.entry_penalty = 0.0
        self.last_action = np.zeros(ACTION_SIZE)
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
        if self.mask_passed_signal and self.exit_seconds is not None:
            color, remaining = 'green', GREEN_SECONDS
        scale = MAX_SPEED_KMH / KMH_PER_MPS
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
        if action.shape != (ACTION_SIZE,) or not np.isfinite(action).all():
            raise ValueError('Actions must contain finite steering and throttle/brake values.')
        self.last_action = np.clip(action, -1, 1)
        color_during_step, remaining = self.signal_state()
        previous_distance = self.stop_distance()
        previous_progress = self.route_progress()
        waiting = self.necessary_wait()
        speed = max(0.0, float(np.asarray(self.env.vehicle.velocity) @ self.direction))
        on_exit_lane = self.exit_seconds is not None
        approach_risk, braking_risk = driving_risks(previous_distance, speed, color_during_step, remaining, self.dt)
        if 0 <= previous_distance <= APPROACH_DISTANCE_M:
            if speed * KMH_PER_MPS > STOPPED_SPEED_KMH:
                self.approach_steps += 1
                self.approach_overspeed_steps += int(speed * KMH_PER_MPS > APPROACH_SPEED_KMH + SPEED_TOLERANCE_KMH)
            self.approach_peak_speed = max(self.approach_peak_speed, speed * KMH_PER_MPS)
        _, _, _, _, native_info = self.env.step(self.last_action)
        self.steps += 1
        current_distance = self.stop_distance()
        crossed = previous_distance >= 0 and current_distance < 0
        first_entry = crossed and self.entry_speed is None
        if first_entry:
            self.entry_speed = float(self.env.vehicle.speed_km_h)
            self.entry_seconds = self.steps * self.dt
            self.stable_green_entry = (color_during_step == 'green'
                                       and GREEN_REMAINING_S <= remaining <= GREEN_SECONDS - GREEN_START_GRACE_S)
        red_violation = bool(color_during_step == 'red' and crossed)
        collision = any(bool(native_info.get(key, False)) for key in COLLISION_KEYS)
        s, lateral = self._position()
        if self.exit_seconds is None and s - self._footprint()[0] >= self.exit_s:
            self.exit_seconds = self.steps * self.dt
        # Lane boundary
        off_road = bool(native_info.get('out_of_road', False)
                        or abs(lateral) + self._footprint()[1] > LANE_WIDTH_M / 2 + LANE_EDGE_TOLERANCE_M)
        success = bool(s >= self.goal_s and not (collision or red_violation or off_road))
        timeout = self.steps >= self.max_steps and not (success or collision or red_violation or off_road)
        reward, reason = transition_reward(
            self.route_progress() - previous_progress, self.dt, waiting,
            collision, red_violation, off_road, success, timeout, approach_risk, braking_risk,
            self.entry_speed if first_entry else None,
        )
        entry_penalty = entry_speed_cost(self.entry_speed) if first_entry and reason in ('running', 'success') else 0.0
        if EXIT_RECOVERY_TRIAL and on_exit_lane and reason == 'running':
            deficit = max(0.0, 1.0 - speed * KMH_PER_MPS / EXIT_RECOVERY_SPEED_KMH)
            cost = self.dt * EXIT_SLOW_COST_PER_SECOND * deficit
            reward -= cost
            self.exit_slow_cost += cost
        if EXPERIMENT == 'green_entry' and first_entry and self.stable_green_entry and reason in ('running', 'success'):
            gap = max(0.0, GREEN_MIN_KMH - self.entry_speed, self.entry_speed - APPROACH_SPEED_KMH - SPEED_TOLERANCE_KMH)
            reward += GREEN_ENTRY_BONUS * max(0.0, 1.0 - gap / GREEN_BONUS_TAPER_KMH)
        self.entry_penalty += entry_penalty
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
        info['step_entry_speed_penalty'] = entry_penalty
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
            entry_speed_penalty=self.entry_penalty,
            stable_green_entry=self.stable_green_entry,
            entry_seconds=self.entry_seconds, exit_seconds=self.exit_seconds, exit_slow_cost=self.exit_slow_cost,
        )

    def _draw_stop_line(self):
        from panda3d.core import LineSegs
        line = LineSegs()
        line.setColor(*STOP_LINE_COLOR)
        line.setThickness(STOP_LINE_WIDTH)
        for method, lateral in ((line.moveTo, -LANE_WIDTH_M / 2), (line.drawTo, LANE_WIDTH_M / 2)):
            x, y = self.approach_lane.position(self.stop_s, lateral)
            method(float(x), float(y), STOP_LINE_HEIGHT_M)
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


def print_reward_table():
    rows = (
        ('Route progress', f'{PROGRESS_REWARD:g} * progress change'),
        ('Approach target speed', f'{APPROACH_SPEED_KMH:g} km/h'),
        ('Entry speed above target (once)', f'-{ENTRY_SPEED_PENALTY_PER_KMH:g} / excess km/h'),
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
    print(f"{'Component':<{REWARD_COLUMN_WIDTH}} Value")
    print('-' * REWARD_TABLE_WIDTH)
    for component, value in rows:
        print(f'{component:<{REWARD_COLUMN_WIDTH}} {value}')
    if EXIT_RECOVERY_TRIAL:
        print(f'Exit lane below {EXIT_RECOVERY_SPEED_KMH:g} km/h: up to -{EXIT_SLOW_COST_PER_SECOND:g} / second')
    if EXPERIMENT == 'green_entry':
        print(f'Stable green entry: up to +{GREEN_ENTRY_BONUS:g} for entering at {GREEN_MIN_KMH:g} to {APPROACH_SPEED_KMH + SPEED_TOLERANCE_KMH:g} km/h')


class TrainingProgress(BaseCallback):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage
        self.episodes = deque(maxlen=RECENT_EPISODES)

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


# Checkpoints
def task_settings():
    settings = dict(
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
                     approach=APPROACH_SPEED_PENALTY, braking=BRAKING_RISK_PENALTY,
                     entry_speed_per_kmh=ENTRY_SPEED_PENALTY_PER_KMH),
    )
    if EXIT_RECOVERY_TRIAL:
        settings['rewards'].update(exit_recovery_speed_kmh=EXIT_RECOVERY_SPEED_KMH, exit_slow_cost_per_second=EXIT_SLOW_COST_PER_SECOND)
    if EXPERIMENT == 'green_entry':
        settings['rewards'].update(green_entry_bonus=GREEN_ENTRY_BONUS, green_min_kmh=GREEN_MIN_KMH,
                                   green_taper_kmh=GREEN_BONUS_TAPER_KMH, green_start_grace_s=GREEN_START_GRACE_S,
                                   green_remaining_s=GREEN_REMAINING_S)
    if MASK_PASSED_SIGNAL:
        settings['passed_signal_observation'] = 'steady_green_after_rear_exit'
    return settings


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


def load_model(path=None, for_training=False):
    path = Path(path) if path is not None else MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(f'Model not found: {path}')
    overrides = ppo_settings() if for_training else {}
    model = PPO.load(str(path), device=DEVICE, **overrides)
    if not hasattr(model, 'policy_training_rewards'):
        model.policy_training_rewards = dict(getattr(model, 'intersection_settings', {}).get('rewards', {}))
    if getattr(model, 'intersection_settings', None) != task_settings():
        previous = getattr(model, 'intersection_settings', {})
        current = task_settings()
        allowed_changes = set(TRAINING_OBJECTIVE_KEYS)
        if MASK_PASSED_SIGNAL and 'passed_signal_observation' not in previous:
            allowed_changes.add('passed_signal_observation')
        same_task = {k: v for k, v in previous.items() if k not in allowed_changes} == {
            k: v for k, v in current.items() if k not in allowed_changes
        }
        if for_training and same_task:
            model.previous_task_settings = previous
            model.previous_reward_settings = previous['rewards']
            model.intersection_settings = current
            print('Continuing with the selected experiment settings.', flush=True)
        else:
            raise ValueError('The checkpoint uses different task settings. Match its settings before loading it.')
    return model


# Metrics
def summarize_results(rows):
    if not rows:
        raise ValueError('Evaluation results must not be empty.')
    successful = [row for row in rows if row['is_success']]
    times = [row['seconds'] for row in successful]
    entries = [row['entry_speed_kmh'] for row in successful if row['entry_speed_kmh'] is not None]
    compliant = sum(speed <= APPROACH_SPEED_KMH + SPEED_TOLERANCE_KMH for speed in entries)
    summary = dict(
        episodes=len(rows), successes=len(successful), success_rate=len(successful) / len(rows),
        red_light_violations=sum(row['red_light_violation'] for row in rows),
        collisions=sum(row['collision'] for row in rows),
        out_of_road=sum(row['out_of_road'] for row in rows),
        timeouts=sum(row['timeout'] for row in rows),
        unsafe_episodes=sum(row['red_light_violation'] or row['collision'] or row['out_of_road'] for row in rows),
        mean_route_completion=float(np.mean([row['route_completion'] for row in rows])),
        mean_reward=float(np.mean([row['episode_reward'] for row in rows])),
        mean_success_seconds=float(np.mean(times)) if times else None,
        mean_success_entry_speed_kmh=float(np.mean(entries)) if entries else None,
        speed_compliant_successes=compliant,
        speed_compliant_success_rate=compliant / len(rows),
        mean_success_entry_excess_kmh=float(np.mean([max(0.0, speed - APPROACH_SPEED_KMH) for speed in entries])) if entries else None,
        mean_moving_approach_overspeed_fraction=float(np.mean([row['approach_overspeed_fraction'] for row in rows])),
    )
    timed = [row for row in successful if row.get('exit_seconds') is not None and row.get('entry_seconds') is not None]
    if 'stable_green_entry' in rows[0]:
        clear = [r['entry_speed_kmh'] for r in successful if r['stable_green_entry']]
        summary.update(stable_green_successes=len(clear),
                       stable_green_in_band=sum(GREEN_MIN_KMH <= v <= APPROACH_SPEED_KMH + SPEED_TOLERANCE_KMH for v in clear),
                       stable_green_median_kmh=float(np.median(clear)) if clear else None)
    if timed:
        summary.update(
            timed_successes=len(timed),
            mean_seconds_to_line=float(np.mean([r['entry_seconds'] for r in timed])),
            mean_seconds_crossing=float(np.mean([r['exit_seconds'] - r['entry_seconds'] for r in timed])),
            mean_seconds_on_exit_lane=float(np.mean([r['seconds'] - r['exit_seconds'] for r in timed])),
        )
    return summary


def evaluation_score(rows):
    summary = summarize_results(rows)
    if EXPERIMENT == 'green_entry':
        safe = (summary['success_rate'] >= GREEN_VALIDATION_SUCCESS and summary['collisions'] == 0
                and summary['out_of_road'] == 0)
        return (safe, summary['stable_green_in_band'] if safe else summary['successes'],
                -summary['unsafe_episodes'], summary['successes'], -(summary['mean_success_seconds'] or EPISODE_SECONDS))
    mean_time = summary['mean_success_seconds']
    acceptable = (summary['success_rate'] >= MIN_VALIDATION_SUCCESS_RATE
                  and summary['collisions'] == 0 and summary['out_of_road'] == 0)
    quality = summary['speed_compliant_success_rate'] if acceptable else summary['success_rate']
    excess = summary['mean_success_entry_excess_kmh']
    return (acceptable, quality, -summary['unsafe_episodes'] / summary['episodes'], summary['success_rate'],
            -(MAX_SPEED_KMH if excess is None else excess),
            -summary['mean_moving_approach_overspeed_fraction'],
            -(EPISODE_SECONDS if mean_time is None else mean_time))


def stage_target_met(rows, stage='signals'):
    summary = summarize_results(rows)
    if EXPERIMENT == 'green_entry' and not (summary['stable_green_in_band'] == summary['stable_green_successes'] > 0):
        return False
    speed_target = STAGE_SUCCESS_TARGET if stage == 'signals' else CURRICULUM_SPEED_SUCCESS_TARGET
    return (summary['success_rate'] >= STAGE_SUCCESS_TARGET
            and summary['speed_compliant_success_rate'] >= speed_target
            and summary['collisions'] == 0 and summary['out_of_road'] == 0
            and summary['mean_moving_approach_overspeed_fraction'] <= MAX_APPROACH_OVERSPEED_FRACTION)


def save_checkpoint(model, stage, completed, history, total_steps, rows, *paths):
    model.completed_stages = completed.copy()
    model.curriculum_history = history.copy()
    model.training_env_steps = total_steps
    model.selected_stage = stage
    model.validation_results = rows
    for path in paths:
        model.save(str(path))


# Training
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
    if model is not None:
        model.training_run_throttle_noise_scale = 1.0
        model.training_run_throttle_bias_shift = 0.0
        model.training_run_start_steps = total_steps
        model.training_run_start_policy_steps = model.num_timesteps
        model.training_run_source_sha256 = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
        print('\nSource policy with current reward settings:', flush=True)
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
                if stage != 'signals' and stage_target_met(best_rows, stage):
                    completed.append(stage)
                    save_checkpoint(model, stage, completed, history, total_steps, best_rows, MODEL_PATH)
                    print(f'{stage} already meets the intermediate target; advancing.', flush=True)
                    continue
            while stage_steps < budget:
                env = Monitor(SignalIntersectionEnv(training=True, stage=stage))
                env.env.sampler = np.random.default_rng(RANDOM_SEED + total_steps)
                try:
                    if model is None:
                        model = PPO('MlpPolicy', env, **ppo_settings(stage), seed=RANDOM_SEED, device=DEVICE, verbose=0)
                        model.training_run_start_steps = 0
                        model.training_run_start_policy_steps = 0
                        model.training_run_source_sha256 = None
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
                summary = summarize_results(rows)
                success = summary['success_rate']
                overspeed = summary['mean_moving_approach_overspeed_fraction']
                score = evaluation_score(rows)
                if EXPERIMENT == 'green_entry':
                    print(f"Stable green in band: {summary['stable_green_in_band']}/{summary['stable_green_successes']}; "
                          f"median {summary['stable_green_median_kmh']} km/h", flush=True)
                history.append(dict(stage=stage, candidate_step=model.num_timesteps,
                                    trained_steps=total_steps, success=success, validation_count=len(rows),
                                    approach_target_kmh=APPROACH_SPEED_KMH,
                                    approach_weight=APPROACH_SPEED_PENALTY,
                                    entry_weight=ENTRY_SPEED_PENALTY_PER_KMH,
                                    compliant_success_rate=summary['speed_compliant_success_rate'],
                                    overspeed_fraction=overspeed))
                if EXPERIMENT == 'green_entry':
                    history[-1]['validation_summary'] = summary
                entry_speed = summary['mean_success_entry_speed_kmh']
                entry_text = 'N/A' if entry_speed is None else f'{entry_speed:.1f} km/h'
                mean_time = summary['mean_success_seconds']
                time_text = 'N/A' if mean_time is None else f'{mean_time:.1f}s'
                print(f'Validation {stage}: {sum(row["is_success"] for row in rows)}/{len(rows)} | '
                      f'Red {sum(row["red_light_violation"] for row in rows)} | '
                      f'Off-road {sum(row["out_of_road"] for row in rows)} | '
                      f'Timeout {sum(row["timeout"] for row in rows)} | '
                      f'Approach overspeed {overspeed:.1%} | Mean successful time {time_text} | '
                      f'Compliant arrivals {summary["speed_compliant_successes"]}/{len(rows)} | '
                      f'Entry speed {entry_text} | '
                      f'Trained steps {total_steps:,}', flush=True)
                if best_score is None or score > best_score:
                    best_score, best_rows = score, rows
                    save_checkpoint(model, stage, completed, history, total_steps, rows, best_path, MODEL_PATH)
                if stage_target_met(rows, stage):
                    break
            model = PPO.load(str(best_path), device=DEVICE)
            if stage_target_met(best_rows, stage) and stage not in completed:
                completed.append(stage)
            paths = [] if EXPERIMENT == 'green_entry' else [MODEL_PATH]
            save_checkpoint(model, stage, completed, history, total_steps, best_rows, *paths)
            print(f'Selected {stage} policy at {model.num_timesteps:,} steps.', flush=True)
            if stage != 'signals' and stage not in completed:
                print(f'{stage} has not reached the stage target; keeping this stage for the next run.', flush=True)
                break
    return model


# Evaluation
def evaluate(model, stage='signals', first_seed=TEST_START_SEED, episodes=TEST_EPISODES, verbose=True,
             mask_passed_signal=None):
    env = SignalIntersectionEnv(render=RENDER_EVALUATION and verbose, stage=stage, mask_passed_signal=mask_passed_signal)
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
            elif len(results) % VALIDATION_LOG_INTERVAL == 0:
                print(f'Validation progress: {len(results)}/{episodes}', flush=True)
    finally:
        env.close()
    if not verbose:
        return results
    summary = summarize_results(results)
    print(f'Successful arrivals at or below {APPROACH_SPEED_KMH + SPEED_TOLERANCE_KMH:g} km/h: '
          f'{summary["speed_compliant_successes"]}/{len(results)}')
    for label, key in (('Successful arrival', 'successes'), ('Red-light violations', 'red_light_violations'),
                       ('Collisions', 'collisions'), ('Road / route departures', 'out_of_road'), ('Timeouts', 'timeouts')):
        count = summary[key]
        print(f'{label}: {count}/{len(results)} ({count / len(results):.1%})')
    if summary['mean_success_seconds'] is not None:
        print(f'Mean successful travel time: {summary["mean_success_seconds"]:.1f} s')
    if summary['mean_success_entry_speed_kmh'] is not None:
        print(f'Mean successful entry speed: {summary["mean_success_entry_speed_kmh"]:.1f} km/h')
    print(f'Mean moving-approach overspeed fraction (above {APPROACH_SPEED_KMH + SPEED_TOLERANCE_KMH:g} km/h): '
          f'{summary["mean_moving_approach_overspeed_fraction"]:.1%}')
    return results


# Results
def save_results(model, model_path, rows):
    path = Path(model_path).resolve()
    model_file = path.relative_to(PROJECT_DIR).as_posix() if path.is_relative_to(PROJECT_DIR) else path.as_posix()
    episodes = [
        {key: value.item() if isinstance(value, np.generic) else value for key, value in row.items()}
        for row in rows
    ]
    result = dict(
        model=dict(file=model_file, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   num_timesteps=int(model.num_timesteps)),
        evaluation=dict(task=task_settings(), traffic_density=TRAFFIC_DENSITY, deterministic=True,
                        seeds=[row['scenario_seed'] for row in episodes]),
        summary=summarize_results(episodes), episodes=episodes,
    )
    if hasattr(model, 'training_run_start_steps'):
        result['training'] = dict(
            attempted_steps=model.training_env_steps - model.training_run_start_steps,
            total_attempted_steps=model.training_env_steps,
            source_policy_steps=model.training_run_start_policy_steps,
            source_sha256=model.training_run_source_sha256,
            selected_from_last_run=model.num_timesteps > model.training_run_start_policy_steps,
            initialization='checkpoint' if model.training_run_source_sha256 else 'random',
            throttle_noise_scale=getattr(model, 'training_run_throttle_noise_scale', 1.0),
            throttle_bias_shift=getattr(model, 'training_run_throttle_bias_shift', 0.0),
        )
    if getattr(model, 'baseline_test_results', None):
        result['baseline_summary'] = summarize_results(model.baseline_test_results)
    if getattr(model, 'validation_results', None):
        result['validation_stage'] = model.selected_stage
        result['validation_summary'] = summarize_results(model.validation_results)
    if EXPERIMENT == 'green_entry' and RUN_MODE == 'train':
        result['candidate_validations'] = [
            dict(policy_steps=h['candidate_step'], summary=h['validation_summary']) for h in model.curriculum_history
            if h['trained_steps'] > model.training_run_start_steps and 'validation_summary' in h
        ]
        if result['training']['selected_from_last_run']:
            holdout = evaluate(model, first_seed=HOLDOUT_START_SEED, episodes=HOLDOUT_EPISODES, verbose=False)
            result['holdout'] = dict(seeds=[row['scenario_seed'] for row in holdout], summary=summarize_results(holdout))
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Results: {RESULTS_PATH}')


def compare_exit_observations(model):
    splits = (('test', TEST_START_SEED, TEST_EPISODES), ('holdout', HOLDOUT_START_SEED, HOLDOUT_EPISODES),
              ('validation', VALIDATION_START_SEED, VALIDATION_EPISODES))
    fields = ('scenario_seed', 'outcome', 'entry_speed_kmh', 'entry_seconds', 'exit_seconds', 'seconds')
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    report = dict(model=MODEL_PATH.name, sha256=digest, policy_steps=int(model.num_timesteps),
                  training_steps_this_run=0, deterministic=True, map=MAP_LAYOUT, traffic_density=TRAFFIC_DENSITY,
                  task=task_settings(), masked_observation='steady_green_after_rear_exit', splits={})
    for split, start, count in splits:
        rows = {}
        summaries = {}
        for arm, masked in (('original', False), ('masked', True)):
            print(f'{split} / {arm}: evaluating {count} scenarios', flush=True)
            rows[arm] = evaluate(model, first_seed=start, episodes=count, verbose=False, mask_passed_signal=masked)
            summaries[arm] = summarize_results(rows[arm])
        pairs = list(zip(rows['original'], rows['masked'], strict=True))
        if any(a['scenario_seed'] != b['scenario_seed'] for a, b in pairs):
            raise ValueError('Scenario seeds differ between arms.')
        successful = [(a, b) for a, b in pairs if a['is_success'] and b['is_success']]
        paired = {arm: summarize_results([pair[i] for pair in successful])
                  for i, arm in enumerate(rows)} if successful else {}
        report['splits'][split] = dict(
            seeds=[start, start + count - 1], summaries=summaries, paired_successes=len(successful),
            paired=paired, episodes={arm: [{k: r[k] for k in fields} for r in records] for arm, records in rows.items()})
        for arm, s in summaries.items():
            print(f"{split} {arm}: {s['successes']}/{count} arrivals, {s['red_light_violations']} red, "
                  f"exit time {s.get('mean_seconds_on_exit_lane')} s", flush=True)
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Checkpoint changed during evaluation.')
    PAIR_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAIR_RESULTS_PATH.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Results: {PAIR_RESULTS_PATH}')


def main():
    validate_settings()
    torch.set_num_threads(CPU_THREADS)
    if EXPERIMENT in ('exit_recovery_masked', 'exit_recovery_pair'):
        if RUN_MODE != 'evaluate':
            raise ValueError('Use evaluate mode for these fixed-weight experiments.')
        model = load_model()
        if EXPERIMENT == 'exit_recovery_pair':
            compare_exit_observations(model)
        else:
            evaluate(model, mask_passed_signal=True)
        return
    print_reward_table()
    model = train() if RUN_MODE == 'train' else load_model()
    rows = evaluate(model)
    selected_path = MODEL_PATH
    if RUN_MODE == 'train':
        model.test_results = rows
        if EXPERIMENT == 'green_entry' and RESUME_MODEL is not None and model.num_timesteps == model.training_run_start_policy_steps:
            selected_path = RESUME_MODEL
        else:
            model.save(str(MODEL_PATH))
    save_results(model, selected_path, rows)


if __name__ == '__main__':
    main()
