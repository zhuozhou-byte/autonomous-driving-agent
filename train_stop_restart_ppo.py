from pathlib import Path
import hashlib
import json
from statistics import fmean, median
from collections import deque

import train_intersection_ppo as road
from train_intersection_ppo import PPO, Monitor, gym, np, torch

# Run
ROOT = Path(__file__).resolve().parent
RUN_MODE = 'evaluate'
BASE_MODEL = ROOT / 'models/ppo_exit_recovery.zip'
BASE_SHA256 = 'c6b4b591dbc5bf7bed91c9739a90a3013b46b0e170beb6ce97f17fc4316bce31'
SMOOTH_BRAKING = True
WARM_START = True
LOCAL_REWARD_ONLY = True
SOURCE_MODEL = ROOT / 'models/ppo_smooth_stop.zip'
MODEL = ROOT / 'models/ppo_smooth_refined.zip'
RESUME_MODEL = MODEL if MODEL.exists() else SOURCE_MODEL
RESULT = ROOT / 'results' / (MODEL.stem + '.json')
SELECTION = (6_000, 200)
EVALUATION = {'validation': (2_000, 200), 'test': (10_000, 50),
              'previous_holdout': (30_000, 100), 'fresh': (40_000, 100), 'new': (50_000, 100),
              'unseen': (60_000, 100)}
REGRESSION_SEEDS = (2162, 30000, 30025, 30024, 30058, 40094, 50054, 50059, 50096, 30010)

# Stop and restart
CONTROL_ZONE_M = 20.0
SOURCE_CONTROL_ZONE_M = 16.0
OBSERVATION_DISTANCE_M = 16.0
BRAKING_METRIC_ZONE_M = 16.0
STOP_MARGIN_M = 2.5
HOLD_SECONDS = 0.5
RESTART_KMH = 2.0
PROMPT_SECONDS = 2.0
PEDAL_RANGE = 0.75
RED_SPEED_COST = 1.0
CREEP_COST_PER_M = 2.0
GREEN_IDLE_COST_PER_S = 1.5
SPEED_SCALE_MPS = 2.0
DECEL_TRACKING_COST = 5.0
YELLOW_STOP_COST = 10.0
LOCAL_CLEAR_REWARD = 2.0
BUFFER_FLOOR_M = 0.3
PRESERVE_STOP_POLICY = True
STOP_POLICY_ZONE_M = 6.0
POLICY_BLEND_M = 2.0
TARGETED_SHARE = 0.90
INITIAL_YELLOW_SHARE = 0.80
SHORT_STOP_TRAINING = True
START_TRAINING_AT_RED = False
LATE_YELLOW_TRAINING = (100_000, 20_000)
LATE_YELLOW_SELECTION = (7_000, 3_000)
LATE_YELLOW_SECONDS = 1.0
LATE_YELLOW_MIN_KMH = 10.0
SELECTION_POOL_SIZE = 1_000
YELLOW_LOOKAHEAD_S = (1.0, 7.0)
YELLOW_GAP_M = (3.0, 15.0)
SAMPLER_ACCEL_MPS2 = 0.65
SAMPLER_FRONT_M = 2.4

# Residual PPO
SEED = 42
CHUNK = 4_096
CHUNKS = 3
ROLLOUT = 1_024
BATCH = 128
EPOCHS = 6
LEARNING_RATE = 0.0006
DISCOUNT = 0.985
GAE_LAMBDA = 0.95
CLIP = 0.15
LAYERS = (32, 32)
TRIP_TOLERANCE_S = 1.0
MIN_QUEUE_COVERAGE = 0.9
SMOOTH_LEARNING_RATE = 0.0001
ACTION_STD = 0.10
BRAKING_WINDOW_S = 0.5
HARD_BRAKING_MPS2 = 4.0
BRAKING_TOLERANCE_MPS2 = 0.1
BRAKING_COST_TARGET_MPS2 = 3.0
EXCESS_BRAKING_COST = 1.0
BRAKE_CHANGE_COST = 1.0


def load_base():
    if hashlib.sha256(BASE_MODEL.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError(f'Unexpected base checkpoint: {BASE_MODEL}')
    road.EXIT_RECOVERY_TRIAL = True
    return PPO.load(str(BASE_MODEL), device='cpu')


def controller_settings(control_zone_m=CONTROL_ZONE_M):
    return dict(control_zone_m=control_zone_m, preserve_stop_policy=PRESERVE_STOP_POLICY,
                stop_policy_zone_m=STOP_POLICY_ZONE_M, policy_blend_m=POLICY_BLEND_M,
                stop_policy=SOURCE_MODEL.name, stop_policy_sha256=hashlib.sha256(SOURCE_MODEL.read_bytes()).hexdigest())


def experiment_settings():
    settings = dict(smooth_braking=SMOOTH_BRAKING, local_reward_only=LOCAL_REWARD_ONLY,
                stop_margin_m=STOP_MARGIN_M,
                pedal_range=PEDAL_RANGE, deceleration_tracking_weight=DECEL_TRACKING_COST,
                target_deceleration_mps2=road.BRAKING_DECELERATION, targeted_training_share=TARGETED_SHARE,
                learning_rate=SMOOTH_LEARNING_RATE if SMOOTH_BRAKING and WARM_START else LEARNING_RATE,
                braking_window_s=BRAKING_WINDOW_S, hard_braking_mps2=HARD_BRAKING_MPS2,
                observation_distance_m=OBSERVATION_DISTANCE_M, braking_metric_zone_m=BRAKING_METRIC_ZONE_M,
                braking_cost_target_mps2=BRAKING_COST_TARGET_MPS2, excess_braking_cost=EXCESS_BRAKING_COST,
                initial_yellow_share=INITIAL_YELLOW_SHARE, brake_change_cost=BRAKE_CHANGE_COST,
                action_std=ACTION_STD, late_yellow_seconds=LATE_YELLOW_SECONDS,
                late_yellow_min_kmh=LATE_YELLOW_MIN_KMH, short_stop_training=SHORT_STOP_TRAINING,
                start_training_at_red=START_TRAINING_AT_RED)
    settings.update(controller_settings())
    return settings


def initial_yellow_seeds(start, count, late_only=False):
    seeds = []
    dt = road.PHYSICS_DT * road.DECISION_REPEAT
    yellow_start = round((road.RED_SECONDS + road.GREEN_SECONDS) / dt)
    cycle = yellow_start + round(road.YELLOW_SECONDS / dt)
    for seed in range(start, start + count):
        rng = np.random.default_rng(seed)
        gap = rng.uniform(*road.START_DISTANCE_M) - SAMPLER_FRONT_M
        speed = rng.uniform(*road.START_SPEED_KMH)
        phase = int(rng.integers(cycle))
        targeted = not late_only or (speed >= LATE_YELLOW_MIN_KMH and (cycle - phase) * dt <= LATE_YELLOW_SECONDS)
        if phase >= yellow_start and gap <= CONTROL_ZONE_M and targeted:
            seeds.append(seed)
    return seeds


def selection_seeds():
    return sorted(set(seed_range(SELECTION)) | set(initial_yellow_seeds(SELECTION[0], SELECTION_POOL_SIZE))
                  | set(initial_yellow_seeds(*LATE_YELLOW_SELECTION, late_only=True)))


def yellow_training_pool(sim):
    pool = []
    cycle = sim.red_steps + sim.green_steps + sim.yellow_steps
    for seed in range(road.TRAIN_START_SEED, road.TRAIN_START_SEED + road.TRAIN_SCENARIOS):
        rng = np.random.default_rng(seed)
        distance = rng.uniform(*road.START_DISTANCE_M)
        speed = rng.uniform(*road.START_SPEED_KMH) / road.KMH_PER_MPS
        offset = int(rng.integers(cycle))
        time = ((sim.red_steps + sim.green_steps - offset) % cycle) * sim.dt
        gap = distance - SAMPLER_FRONT_M - speed * time - 0.5 * SAMPLER_ACCEL_MPS2 * time ** 2
        if YELLOW_LOOKAHEAD_S[0] <= time <= YELLOW_LOOKAHEAD_S[1] and YELLOW_GAP_M[0] <= gap <= YELLOW_GAP_M[1]:
            pool.append(seed)
    if not pool:
        raise ValueError('No training seeds meet the yellow-approach criteria.')
    return pool


class StopRestart(gym.Env):
    def __init__(self, base, training=False, sampler_seed=SEED, control_zone_m=CONTROL_ZONE_M, approach_policy=None, stop_policy=None):
        super().__init__()
        self.sim = road.SignalIntersectionEnv(mask_passed_signal=True)
        self.training = training
        self.sampler = np.random.default_rng(sampler_seed)
        self.pool = yellow_training_pool(self.sim) if training else None
        self.initial_pool = initial_yellow_seeds(*LATE_YELLOW_TRAINING, late_only=True) if training else None
        self.base = base
        self.control_zone_m = control_zone_m
        self.approach_policy = approach_policy
        self.stop_policy = stop_policy
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (7,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.focused_stop = False
        if self.training:
            draw = self.sampler.random()
            if draw < INITIAL_YELLOW_SHARE:
                seed = int(self.sampler.choice(self.initial_pool))
                self.focused_stop = SHORT_STOP_TRAINING
            elif draw < TARGETED_SHARE:
                seed = int(self.sampler.choice(self.pool))
            else:
                seed = int(self.sampler.integers(road.TRAIN_START_SEED, road.TRAIN_START_SEED + road.TRAIN_SCENARIOS))
        self.base_obs, info = self.sim.reset(seed=seed, options=options)
        self.last_color = None
        self.yellow_stop = False
        self.events = []
        self.red_event = None
        self.launch = None
        self.hold_ticks = 0
        self.training_stop_ticks = 0
        observation = self.state()
        if self.focused_stop and self.approach_policy is not None:
            while self.last_color == 'yellow':
                action = self.approach_policy.predict(observation, deterministic=True)[0] if self.active else (0.0,)
                observation, _, done, truncated, info = self.step(action)
                if done or truncated:
                    raise RuntimeError('Approach warmup ended before the red-light training segment.')
        return observation, info

    def state(self):
        sim = self.sim
        color, remaining = sim.signal_state()
        distance = sim.stop_distance()
        speed = max(0.0, float(np.asarray(sim.env.vehicle.velocity) @ sim.direction))
        if color == 'yellow' and self.last_color != 'yellow':
            stopping = speed * sim.dt + speed ** 2 / (2 * road.BRAKING_DECELERATION)
            self.yellow_stop = distance >= stopping + road.BRAKING_MARGIN_M
        self.last_color = color
        self.braking = color == 'red' or (color == 'yellow' and self.yellow_stop)
        releasing = (color == 'green' and self.launch is not None and
                     (sim.steps - self.launch['green_step']) * sim.dt < PROMPT_SECONDS)
        self.active = 0 <= distance <= self.control_zone_m and sim.entry_seconds is None and (self.braking or releasing)
        self.distance, self.speed = distance, speed
        features = (distance / OBSERVATION_DISTANCE_M, speed * road.KMH_PER_MPS / road.MAX_SPEED_KMH,
                    remaining / max(road.GREEN_SECONDS, road.RED_SECONDS, road.YELLOW_SECONDS),
                    color == 'red', color == 'yellow', bool(releasing), self.base_obs[12])
        self.observation = np.clip(features, -1, 1).astype(np.float32)
        return self.observation

    def step(self, action):
        sim = self.sim
        previous_throttle = float(self.base_obs[12])
        base_action, _ = self.base.predict(self.base_obs, deterministic=True)
        if self.active:
            correction = float(action[0])
            if self.stop_policy is not None and self.distance < STOP_POLICY_ZONE_M + POLICY_BLEND_M:
                weight = float(np.clip((STOP_POLICY_ZONE_M + POLICY_BLEND_M - self.distance) / POLICY_BLEND_M, 0, 1))
                stop_action = float(self.stop_policy.predict(self.observation, deterministic=True)[0][0])
                correction = (1 - weight) * correction + weight * stop_action
            base_action[1] = np.clip(base_action[1] + PEDAL_RANGE * correction, -1, 1)
        before = self.distance
        color = self.last_color
        had_entered = sim.entry_seconds is not None
        self.base_obs, reward, done, truncated, info = sim.step(base_action)
        if self.training and LOCAL_REWARD_ONLY and not (done or truncated):
            reward = 0.0
        after, speed_kmh = info['stop_line_distance'], float(sim.env.vehicle.speed_km_h)
        if self.active and self.braking:
            speed_limit = np.sqrt(2 * road.BRAKING_DECELERATION * max(0.0, before - STOP_MARGIN_M))
            reward -= RED_SPEED_COST * max(0.0, self.speed - speed_limit) / SPEED_SCALE_MPS * sim.dt
            target_decel = min(road.BRAKING_DECELERATION, self.speed ** 2 / (2 * max(before - STOP_MARGIN_M, BUFFER_FLOOR_M)))
            forward = float(np.asarray(sim.env.vehicle.velocity) @ sim.direction)
            error = forward - max(0.0, self.speed - target_decel * sim.dt)
            reward -= DECEL_TRACKING_COST * (abs(error) if SMOOTH_BRAKING else max(0.0, error))
            deceleration = max(0.0, self.speed - max(0.0, forward)) / sim.dt
            excess = max(0.0, deceleration - BRAKING_COST_TARGET_MPS2)
            reward -= EXCESS_BRAKING_COST * excess ** 2 * sim.dt
            if self.speed * road.KMH_PER_MPS > road.STOPPED_SPEED_KMH:
                reward -= BRAKE_CHANGE_COST * abs(float(base_action[1]) - previous_throttle)
            if after < STOP_MARGIN_M:
                reward -= CREEP_COST_PER_M * max(0.0, min(before, STOP_MARGIN_M) - after)
        if self.launch is not None and color == 'green' and self.launch['restart_s'] is None:
            forward = float(np.asarray(sim.env.vehicle.velocity) @ sim.direction) * road.KMH_PER_MPS
            if forward >= RESTART_KMH:
                self.launch['restart_s'] = round((sim.steps - self.launch['green_step']) * sim.dt, 3)
            else:
                reward -= GREEN_IDLE_COST_PER_S * sim.dt
        next_color = info['signal_color']
        if color == 'red' and not had_entered and before >= 0 and min(before, after) <= road.WAIT_ZONE_M:
            if self.red_event is None:
                self.red_event = dict(stable=False, at_green=False, green_step=None, restart_s=None, stop_gap_m=None)
                self.events.append(self.red_event)
            self.hold_ticks = self.hold_ticks + 1 if 0 <= after <= road.WAIT_ZONE_M and speed_kmh <= road.STOPPED_SPEED_KMH else 0
            if self.hold_ticks * sim.dt >= HOLD_SECONDS:
                self.red_event['stable'] = True
            if next_color == 'green' and not done:
                self.red_event['green_step'] = sim.steps
                if self.hold_ticks * sim.dt >= HOLD_SECONDS:
                    self.red_event.update(at_green=True, stop_gap_m=round(after, 3))
                    self.launch = self.red_event
        if next_color != 'red':
            self.red_event = None
            self.hold_ticks = 0
        if next_color != 'green':
            self.launch = None
        if self.training and not done and not had_entered and sim.entry_seconds is not None:
            done = True
            reward += -YELLOW_STOP_COST if color == 'yellow' and self.yellow_stop else LOCAL_CLEAR_REWARD
        if self.training and self.focused_stop:
            stopped = color == 'red' and after >= 0 and speed_kmh <= road.STOPPED_SPEED_KMH
            self.training_stop_ticks = self.training_stop_ticks + 1 if stopped else 0
            done = done or self.training_stop_ticks * sim.dt >= HOLD_SECONDS
        if done or truncated:
            info['stop_events'] = self.events
        return self.state(), reward, done, truncated, info

    def close(self):
        self.sim.close()


def evaluate(base, policy, seeds, control_zone_m=CONTROL_ZONE_M):
    stop_policy = PPO.load(str(SOURCE_MODEL), device='cpu') if PRESERVE_STOP_POLICY and policy is not None else None
    env = StopRestart(base, control_zone_m=control_zone_m, stop_policy=stop_policy)
    rows = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            releases, stopped_ticks, pending = [], 0, None
            window = round(BRAKING_WINDOW_S / env.sim.dt)
            speeds = deque([env.speed], maxlen=window + 1)
            peak_brake = 0.0
            upstream_brake = 0.0
            braking_steps = deque(maxlen=window)
            while True:
                color = env.last_color
                braking_steps.append(0 <= env.distance <= BRAKING_METRIC_ZONE_M
                                     and env.sim.entry_seconds is None and env.braking)
                action = policy.predict(obs, deterministic=True)[0] if policy is not None and env.active else (0.0,)
                obs, _, done, truncated, info = env.step(action)
                sim = env.sim
                speeds.append(env.speed)
                if len(speeds) == window + 1 and sim.entry_seconds is None:
                    deceleration = max(0.0, (speeds[0] - speeds[-1]) / BRAKING_WINDOW_S)
                    upstream_brake = max(upstream_brake, deceleration)
                    if any(braking_steps):
                        peak_brake = max(peak_brake, deceleration)
                upstream = sim.entry_seconds is None and info['stop_line_distance'] >= 0
                stopped = upstream and sim.env.vehicle.speed_km_h <= road.STOPPED_SPEED_KMH
                if color == 'red':
                    stopped_ticks = stopped_ticks + 1 if stopped else 0
                    if info['signal_color'] == 'green' and stopped_ticks * sim.dt >= HOLD_SECONDS:
                        pending = dict(green_step=sim.steps, gap_m=info['stop_line_distance'],
                                       red_hold_s=round(stopped_ticks * sim.dt, 3), restart_s=None)
                        releases.append(pending)
                else:
                    stopped_ticks = 0
                if pending is not None and color == 'green' and env.speed * road.KMH_PER_MPS >= RESTART_KMH:
                    pending['restart_s'] = round((sim.steps - pending['green_step']) * sim.dt, 3)
                    pending = None
                if info['signal_color'] != 'green':
                    pending = None
                if done or truncated:
                    keys = ('scenario_seed', 'outcome', 'seconds', 'red_light_violation', 'collision',
                            'out_of_road', 'timeout', 'entry_speed_kmh', 'stop_events')
                    rows.append(dict(**{k: info[k] for k in keys}, upstream_releases=releases,
                                     peak_control_brake_mps2=peak_brake, peak_upstream_brake_mps2=upstream_brake))
                    break
    finally:
        env.close()
    return rows


def seed_range(split):
    start, count = split
    return range(start, start + count)


def summarize(rows):
    successes = [row for row in rows if row['outcome'] == 'success']
    events = [event for row in rows for event in row['stop_events']]
    queued = [event for event in events if event['at_green']]
    delays = [event['restart_s'] for event in queued if event['restart_s'] is not None]
    summary = dict(
        episodes=len(rows), successes=len(successes),
        red_light_violations=sum(bool(row['red_light_violation']) for row in rows),
        collisions=sum(bool(row['collision']) for row in rows),
        out_of_road=sum(bool(row['out_of_road']) for row in rows),
        timeouts=sum(bool(row['timeout']) for row in rows),
        mean_success_seconds=fmean(row['seconds'] for row in successes) if successes else None,
        red_stop_opportunities=len(events), stable_stops=sum(bool(event['stable']) for event in events),
        restart_eligible=len(queued), restarts=len(delays), missed_restarts=len(queued) - len(delays),
        prompt_restarts=sum(delay <= PROMPT_SECONDS for delay in delays),
        median_restart_s=median(delays) if delays else None,
    )
    if rows and 'upstream_releases' in rows[0]:
        releases = [release for row in rows for release in row['upstream_releases']]
        actual = [r['restart_s'] for r in releases if r['restart_s'] is not None]
        summary['all_stopped_at_green'] = dict(
            count=len(releases), restarted=len(actual), within_target=sum(t <= PROMPT_SECONDS for t in actual),
            median_restart_s=median(actual) if actual else None,
            median_gap_m=median(r['gap_m'] for r in releases) if releases else None)
    if rows and 'peak_control_brake_mps2' in rows[0]:
        peaks = [r['peak_control_brake_mps2'] for r in rows]
        summary['braking'] = dict(episodes_over_limit=sum(v > HARD_BRAKING_MPS2 for v in peaks),
                                  p95_control_mps2=float(np.percentile(peaks, 95)), max_control_mps2=max(peaks))
        full = [r['peak_upstream_brake_mps2'] for r in rows]
        summary['braking'].update(upstream_episodes_over_limit=sum(v > HARD_BRAKING_MPS2 for v in full),
                                   upstream_max_mps2=max(full), p95_upstream_mps2=float(np.percentile(full, 95)))
    return summary


def compare(baseline_rows, rows):
    baseline = {row['scenario_seed']: row for row in baseline_rows}
    current = {row['scenario_seed']: row for row in rows}
    if len(baseline) != len(baseline_rows) or len(current) != len(rows) or baseline.keys() != current.keys():
        raise ValueError('Paired evaluation requires the same unique scenario seeds.')
    fixed_red, new_red, trip_changes, restart_pairs = [], [], [], []
    for seed in sorted(baseline):
        before, after = baseline[seed], current[seed]
        if before['red_light_violation'] and not after['red_light_violation']:
            fixed_red.append(seed)
        if after['red_light_violation'] and not before['red_light_violation']:
            new_red.append(seed)
        if before['outcome'] == after['outcome'] == 'success':
            trip_changes.append(after['seconds'] - before['seconds'])
        old_queue = [event for event in before['stop_events'] if event['at_green']]
        new_queue = [event for event in after['stop_events'] if event['at_green']]
        if len(old_queue) == len(new_queue) == 1:
            old, new = old_queue[0], new_queue[0]
            if old.get('green_step') == new.get('green_step'):
                restart_pairs.append((old['restart_s'], new['restart_s']))
    completed = [(old, new) for old, new in restart_pairs if old is not None and new is not None]
    result = dict(
        red_violation_cleared_seeds=fixed_red, new_red_violation_seeds=new_red,
        common_successes=len(trip_changes),
        mean_paired_trip_change_s=fmean(trip_changes) if trip_changes else None,
        paired_restart_eligible=len(restart_pairs), paired_both_restarted=len(completed),
        paired_baseline_misses=sum(old is None for old, _ in restart_pairs),
        paired_current_misses=sum(new is None for _, new in restart_pairs),
        paired_baseline_median_restart_s=median(old for old, _ in completed) if completed else None,
        paired_current_median_restart_s=median(new for _, new in completed) if completed else None,
        median_paired_restart_change_s=median(new - old for old, new in completed) if completed else None,
    )
    if rows and 'upstream_releases' in rows[0] and 'upstream_releases' in baseline_rows[0]:
        old, new = ({(r['scenario_seed'], e['green_step']): e['restart_s']
                     for r in records for e in r['upstream_releases']} for records in (baseline_rows, rows))
        common = old.keys() & new.keys()
        delays = [new[k] - old[k] for k in common if old[k] is not None and new[k] is not None]
        result['all_stopped_at_same_green'] = dict(count=len(common), both_restarted=len(delays),
                                                   median_restart_change_s=median(delays) if delays else None)
    return result


def score(summary, reference):
    q = summary
    events = q['red_stop_opportunities']
    time = q['mean_success_seconds']
    adequate = (time is not None and time <= reference['mean_success_seconds'] + TRIP_TOLERANCE_S
                and q['restart_eligible'] >= reference['restart_eligible'] * MIN_QUEUE_COVERAGE)
    if SMOOTH_BRAKING:
        stopped = q['all_stopped_at_green']
        prompt = (stopped['count'] == stopped['within_target']
                  and stopped['count'] >= max(1, reference['all_stopped_at_green']['count'] * MIN_QUEUE_COVERAGE))
        timely = time is not None and time <= reference['mean_success_seconds'] + TRIP_TOLERANCE_S
        braking, old_braking = q['braking'], reference['braking']
        upstream_ok = (braking['upstream_episodes_over_limit'] <= old_braking['upstream_episodes_over_limit']
                       and braking['upstream_max_mps2'] <= old_braking['upstream_max_mps2'] + BRAKING_TOLERANCE_MPS2)
        stopped_safely = q['stable_stops'] == events
        return (q['successes'], -q['red_light_violations'], -q['collisions'], -q['out_of_road'],
                prompt and timely and stopped_safely and upstream_ok,
                -q['braking']['episodes_over_limit'], -q['braking']['max_control_mps2'],
                -q['braking']['p95_control_mps2'],
                -(time if time is not None else road.EPISODE_SECONDS))
    return (q['successes'], -q['red_light_violations'], -q['collisions'], -q['out_of_road'],
            adequate, q['stable_stops'] / max(events, 1), q['prompt_restarts'] / max(q['restart_eligible'], 1),
            -(time if time is not None else road.EPISODE_SECONDS))


def train(base, reference=None):
    MODEL.parent.mkdir(exist_ok=True, parents=True)
    resume_hash, warm_model, initial_steps, approach_policy = None, None, 0, None
    stop_policy = PPO.load(str(SOURCE_MODEL), device='cpu') if PRESERVE_STOP_POLICY else None
    if SMOOTH_BRAKING and WARM_START:
        resume_hash = hashlib.sha256(RESUME_MODEL.read_bytes()).hexdigest()
        warm_model = PPO.load(str(RESUME_MODEL), device='cpu', learning_rate=SMOOTH_LEARNING_RATE)
        approach_policy = PPO.load(str(RESUME_MODEL), device='cpu') if START_TRAINING_AT_RED else None
        initial_steps = warm_model.num_timesteps
    elif SHORT_STOP_TRAINING and START_TRAINING_AT_RED:
        approach_policy = PPO.load(str(RESUME_MODEL), device='cpu')
    if reference is None:
        previous = PPO.load(str(SOURCE_MODEL), device='cpu') if SMOOTH_BRAKING else None
        reference = evaluate(base, previous, selection_seeds(), control_zone_m=SOURCE_CONTROL_ZONE_M)
    baseline = summarize(reference)
    candidates, best_score, selected_step, model = [], None, None, None
    source_steps = 0
    if SMOOTH_BRAKING:
        source = PPO.load(str(SOURCE_MODEL), device='cpu')
        source_steps = selected_step = source.num_timesteps
        best_score = score(baseline, baseline)
        source.stop_restart_settings = dict(getattr(source, 'stop_restart_settings', {}), **controller_settings(SOURCE_CONTROL_ZONE_M))
        source.save(str(MODEL))
        candidates.append(dict(steps=source_steps, summary=baseline, weights_updated=False,
                               control_zone_m=SOURCE_CONTROL_ZONE_M))
        starting = summarize(evaluate(base, source, selection_seeds()))
        candidates.append(dict(steps=source_steps, summary=starting, weights_updated=False, control_zone_m=CONTROL_ZONE_M))
        if score(starting, baseline) > best_score:
            best_score = score(starting, baseline)
            source.stop_restart_settings = dict(source.stop_restart_settings, **controller_settings())
            source.save(str(MODEL))
        print(f'Starting policy with earlier control: {starting}', flush=True)
        if warm_model is not None and resume_hash != hashlib.sha256(SOURCE_MODEL.read_bytes()).hexdigest():
            resumed = summarize(evaluate(base, warm_model, selection_seeds()))
            candidates.append(dict(steps=initial_steps, summary=resumed, weights_updated=False, control_zone_m=CONTROL_ZONE_M))
            if score(resumed, baseline) > best_score:
                best_score, selected_step = score(resumed, baseline), initial_steps
                warm_model.stop_restart_settings = dict(warm_model.stop_restart_settings, **controller_settings())
                warm_model.save(str(MODEL))
    for chunk in range(CHUNKS):
        env = Monitor(StopRestart(base, training=True, sampler_seed=SEED + chunk * CHUNK,
                                 approach_policy=approach_policy, stop_policy=stop_policy))
        try:
            if model is None:
                if SMOOTH_BRAKING and WARM_START:
                    model = warm_model
                    model.set_env(env)
                    model.set_random_seed(SEED)
                    reset_value = LOCAL_REWARD_ONLY and not getattr(model, 'stop_restart_settings', {}).get('local_reward_only', False)
                    with torch.no_grad():
                        model.policy.log_std.fill_(np.log(ACTION_STD))
                        if reset_value:
                            model.policy.value_net.weight.zero_()
                            model.policy.value_net.bias.zero_()
                    if reset_value:
                        model.policy.optimizer.state.clear()
                else:
                    model = PPO('MlpPolicy', env, seed=SEED, device='cpu', verbose=0, n_steps=ROLLOUT,
                            batch_size=BATCH, n_epochs=EPOCHS, learning_rate=LEARNING_RATE, gamma=DISCOUNT,
                            gae_lambda=GAE_LAMBDA, clip_range=CLIP,
                            policy_kwargs=dict(net_arch=list(LAYERS), log_std_init=float(np.log(ACTION_STD))))
                model.base_sha256 = BASE_SHA256
                model.stop_restart_settings = experiment_settings()
            else:
                model.set_env(env)
            model.learn(CHUNK, reset_num_timesteps=False)
        finally:
            env.close()
        metrics = summarize(evaluate(base, model, selection_seeds()))
        candidates.append(dict(steps=model.num_timesteps, summary=metrics, weights_updated=True, control_zone_m=CONTROL_ZONE_M))
        rating = score(metrics, baseline)
        print(f'Validation after {model.num_timesteps:,} steps: {metrics}', flush=True)
        if best_score is None or rating > best_score:
            best_score, selected_step = rating, model.num_timesteps
            model.save(str(MODEL))
    return dict(seed=SEED, training_seed_range=[road.TRAIN_START_SEED, road.TRAIN_START_SEED + road.TRAIN_SCENARIOS - 1],
                late_yellow_training_seeds=initial_yellow_seeds(*LATE_YELLOW_TRAINING, late_only=True),
                resume_model=RESUME_MODEL.name, resume_sha256=resume_hash, initial_policy_steps=initial_steps,
                validation_split=SELECTION, validation_seeds=selection_seeds(), validation_baseline=baseline, candidates=candidates,
                selected_steps=selected_step, total_steps=model.num_timesteps,
                source_policy_steps=source_steps, attempted_steps=CHUNKS * CHUNK,
                initialization='checkpoint' if SMOOTH_BRAKING and WARM_START else 'random',
                beats_validation_baseline=best_score > score(baseline, baseline))


def evaluate_splits(base, policy, baseline_cache=None):
    splits, regressions = {}, {}
    previous = PPO.load(str(SOURCE_MODEL), device='cpu') if SMOOTH_BRAKING else None
    for name, split in EVALUATION.items():
        before = baseline_cache[name] if baseline_cache is not None else evaluate(base, previous, seed_range(split), control_zone_m=SOURCE_CONTROL_ZONE_M)
        after = evaluate(base, policy, seed_range(split), control_zone_m=policy.stop_restart_settings['control_zone_m'])
        change = compare(before, after)
        splits[name] = dict(seeds=[split[0], split[0] + split[1] - 1], baseline=summarize(before),
                            trained=summarize(after), paired=change, baseline_episodes=before, episodes=after)
        print(f'{name}: {splits[name]["trained"]}', flush=True)
        for a, b in zip(before, after, strict=True):
            if a['scenario_seed'] in REGRESSION_SEEDS:
                regressions[str(a['scenario_seed'])] = dict(baseline=a, trained=b)
    return splits, regressions


def write_report(training, splits, regressions):
    selected = PPO.load(str(MODEL), device='cpu')
    measurements = dict(stop_zone_m=road.WAIT_ZONE_M, stopped_kmh=road.STOPPED_SPEED_KMH,
                        minimum_hold_s=HOLD_SECONDS, restart_forward_kmh=RESTART_KMH, prompt_s=PROMPT_SECONDS,
                        braking_window_s=BRAKING_WINDOW_S, hard_braking_mps2=HARD_BRAKING_MPS2,
                        braking_metric_zone_m=BRAKING_METRIC_ZONE_M, source_control_zone_m=SOURCE_CONTROL_ZONE_M,
                        peak_definition='Maximum speed loss over a rolling 0.5-second window')
    before = [r for split in splits.values() for r in split['baseline_episodes']]
    after = [r for split in splits.values() for r in split['episodes']]
    overall = dict(baseline=summarize(before), trained=summarize(after), paired=compare(before, after))
    result = dict(base_model=BASE_MODEL.name, base_sha256=BASE_SHA256, residual_model=MODEL.name,
                  residual_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(), map_seed=road.MAP_SEED,
                  traffic_density=road.TRAFFIC_DENSITY, measurements=measurements,
                  settings=selected.stop_restart_settings, attempted_settings=experiment_settings(),
                  source_model=SOURCE_MODEL.name, source_sha256=hashlib.sha256(SOURCE_MODEL.read_bytes()).hexdigest(),
                  training=training, overall=overall, splits=splits, regressions=regressions)
    RESULT.parent.mkdir(exist_ok=True, parents=True)
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    torch.set_num_threads(road.CPU_THREADS)
    base = load_base()
    if RUN_MODE == 'train':
        history = train(base)
        policy = PPO.load(str(MODEL), device='cpu')
        splits, regressions = evaluate_splits(base, policy)
        write_report(history, splits, regressions)
    elif RUN_MODE == 'evaluate':
        policy = PPO.load(str(MODEL), device='cpu')
        if getattr(policy, 'base_sha256', None) != BASE_SHA256:
            raise ValueError('Residual and base models do not match.')
        expected_stop = policy.stop_restart_settings.get('stop_policy_sha256')
        if PRESERVE_STOP_POLICY and expected_stop and expected_stop != hashlib.sha256(SOURCE_MODEL.read_bytes()).hexdigest():
            raise ValueError('The preserved stopping checkpoint does not match this controller.')
        print(f'Braking PPO: {MODEL.name} | Stop/restart PPO: {SOURCE_MODEL.name}')
        for row in evaluate(base, policy, REGRESSION_SEEDS, control_zone_m=policy.stop_restart_settings['control_zone_m']):
            print(f"Seed {row['scenario_seed']}: {row['outcome']}, trip {row['seconds']:.1f} s")
            print(f"  Peak braking before the line over {BRAKING_WINDOW_S:g} s: {row['peak_upstream_brake_mps2']:.2f} m/s^2")
            for release in row['upstream_releases']:
                delay = release['restart_s']
                restart = 'missed green' if delay is None else f'{delay:.1f} s'
                print(f"  Red hold {release['red_hold_s']:.1f} s | gap at green {release['gap_m']:.2f} m | restart {restart}")
    else:
        raise ValueError("Use 'train' or 'evaluate'.")


if __name__ == '__main__':
    main()
