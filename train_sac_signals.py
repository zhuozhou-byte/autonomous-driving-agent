"""Train or evaluate a SAC driver at a SUMO-controlled MetaDrive junction.

Replay a published checkpoint:
    python train_sac_signals.py --evaluate models/sac_signals.zip --scenario gaps
"""

import argparse
from datetime import datetime
import logging
import math
from pathlib import Path
import subprocess
import sys

PROJECT_DIR = Path(__file__).resolve().parent
SIGNAL_NET_FILE = PROJECT_DIR / 'sumo_signal.net.xml'
METADRIVE_PYTHON = Path(r'D:\conda-envs\metadrive\python.exe')

# Prefer the installed simulator over a sibling MetaDrive source checkout.
_import_path = sys.path.copy()
sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != PROJECT_DIR]
try:
    import gymnasium as gym
    import libsumo
    import numpy as np
    import torch
    from metadrive.component.lane.point_lane import PointLane
    from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
    from metadrive.component.vehicle.vehicle_type import DefaultVehicle
    from metadrive.manager.sumo_map_manager import SumoMapManager
    from metadrive.envs import BaseEnv
    from metadrive.obs.observation_base import BaseObservation
    from stable_baselines3 import SAC
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except ImportError:
    if __name__ == '__main__' and METADRIVE_PYTHON.is_file() and METADRIVE_PYTHON.resolve() != Path(sys.executable).resolve():
        raise SystemExit(subprocess.call([str(METADRIVE_PYTHON), '-B', str(Path(__file__).resolve()), *sys.argv[1:]]))
    raise
finally:
    sys.path[:] = _import_path


# Simulation and spawning
ENV_SEED = 42
ROLLOUT_SEEDS = (200_000, 300_000)
NUM_SCENARIOS = 1_000_000
EPISODE_STEPS = 325
PHYSICS_STEP_S = 0.02
PHYSICS_STEPS_PER_ACTION = 10
SPAWN_LONGITUDE_M = (8, 49)
SPAWN_LATERAL_M = (-0.28, 0.28)
SPAWN_SPEED_MPS = (0.0, 5.0)
PEDESTRIAN_APPROACH_M = (6.0, 12.0)
ONCOMING_APPROACH_M = (9.0, 15.0)
TURN_SPAWN_SPEED_MPS = (3.0, 4.8)
SIGNAL_APPROACH_SPEED_RATIO = (0.78, 0.98)
YELLOW_LEAD_TICKS = (2, 6)  # upper bound is exclusive
STOPPING_SURPLUS_M = (2.5, 8.0)
STOPPING_SHORTFALL_M = (3.0, 5.0)
RED_WAIT_BASE_S = 7.0
RED_START_JITTER_TICKS = 25
GREEN_START_JITTER_TICKS = 10

# Traffic rules and reward
YELLOW_BRAKING_MPS2 = 3.0
EGO_FIRST_CLEARANCE_S = 1.0
STOP_ZONE_M = 8.0
STOPPED_SPEED_MPS = 0.3
RED_WAIT_MIN_S = 1.0
REVERSE_SPEED_TOLERANCE_MPS = 0.5
FINISH_TOLERANCE_M = 6.0
PROGRESS_REWARD_PER_M = 0.7
OVERSPEED_PENALTY = 3.0
FAILURE_REWARD = -50.0
ARRIVAL_BONUS = 25.0

# Gap estimates and demonstration controller
TURN_SPEED_MPS = 4.5
TURN_LOOKAHEAD_M = 4.0
GAP_ESTIMATED_ACCEL_MPS2 = 3.0
TURN_APPROACH_DECEL_MPS2 = 1.5
STOP_DECEL_MPS2 = 1.6
STOP_BUFFER_M = 3.0
WALK_CLEAR_BUFFER_M = 0.20
TEACHER_CRUISE_RATIO = 0.85
TEACHER_LATERAL_GAIN = 0.38
TEACHER_HEADING_GAIN = 1.8
TEACHER_SPEED_GAIN = 0.48
TEACHER_STOP_COMMAND = -0.65
TEACHER_HOLD_DISTANCE_M = 3.2
TEACHER_MIN_CAR_GAP_S = 1.6
TEACHER_WALK_LEAD_S = 0.6
TURN_APPROACH_BUFFER_M = 1.0
OPPONENT_CLEAR_BUFFER_M = 1.0
PROBE_GAP_CENTER_S = 1.4
PROBE_WALK_CENTER_S = 1.5
PROBE_HALF_WIDTH_S = 0.20


# SAC and imitation learning
TRAIN_SEED = 42
ONLINE_STEPS = 10_000
DEMO_STEPS = 42_000
CLONE_UPDATES = 6_000
EXTENSION_STEPS = 8_000
EXTENSION_BUDGETS = {21: (22_000, 5_000), 27: (34_000, 6_500), 35: (34_000, 6_500)}  # Demos, clone updates
HIDDEN_LAYERS = (128, 128)
SAC_OPTIONS = dict(learning_rate=1e-4, learning_starts=0, train_freq=4, gradient_steps=1,
                   batch_size=128, gamma=0.99, ent_coef='auto_0.03')
MIN_REPLAY_SIZE = 30_000
CLONE_LR = 3e-4
EXTENSION_CLONE_LR = 2e-3
INITIAL_LOG_STD = -2.5
DEMO_GRADIENT_WEIGHT = 40.0
DEMO_NOISE_PERIOD = 6
DEMO_NOISE_CHANCE = 0.45
DEMO_NOISE_STD = (0.09, 0.10)  # Steering, throttle
RANDOM_DEMOS_PER_BATCH = 24
DEMOS_PER_BATCH = dict(turning=12, oncoming=12, gap=12, releasing=8, braking=8, signal_wait=8,
                       green_launch=8, steering=8, ped_braking=16, ped_release=12)
CORRECTION_EPISODES = 128
CORRECTION_UPDATES = 1_200
CORRECTIONS_PER_BATCH = 32
CORRECTION_BASE_WEIGHT = 0.1
DEMO_SEEDS = (100_000, 200_000)
CORRECTION_SEEDS = (300_000, 400_000)

# Gap calibration and evaluation
CALIBRATION_SEEDS = (400_001, 500_000)
CALIBRATION_SEED_LIMIT = 4_096
CALIBRATION_QUOTAS = (8, 20, 20, 20, 20, 32)  # Window endpoints and four interior bins
CALIBRATION_MIN_CAR_GAP_S = 1.75
CALIBRATION_MIN_PED_GAP_S = 0.8
CALIBRATION_GUARD_S = 0.1
CALIBRATION_MIN_PER_CASE = 8
EVAL_EPISODES = 64
EVAL_FIRST_SEED = 5_000


# Traffic scenarios. Arrival and clearance windows use simulation seconds.
APPROACHES = ('west', 'east', 'north', 'south')
ONCOMING_SPEED_MPS = (5.5, 7.0)
CROSSWALK_OFFSET_M = 3.0
CROSSWALK_WIDTH_M = 2.4
SIDEWALK_MARGIN_M = 1.2
CURB_START_JITTER_M = (0.0, 0.3)
WALK_CLEARANCE_S = {
    'ped-yield': (7.6, 10.0), 'ped-clear': (0.35, 1.55), 'ped-timed': (3.2, 7.4),
    'ped-mid-open': (2.6, 4.2), 'ped-mid-busy': (6.0, 7.4),
}
CROSSWALK_STOP_TIME_S = 0.8
SCENARIO_CYCLE = ('standard', 'red', 'standard', 'green', 'standard', 'yellow-stop', 'standard', 'yellow-go')
YIELD_SCENARIOS = ('yield-close', 'yield-gap', 'yield-varied')
ARRIVAL_WINDOWS = {'yield-close': (3.5, 4.8), 'yield-gap': (8.9, 10.2),
                   'yield-varied': (4.8, 8.9), 'yield-mid': (6.7, 8.7)}
PED_SCENARIOS = ('ped-yield', 'ped-clear', 'ped-timed')
JOINT_TRAFFIC = {
    'joint-near-busy': ('yield-close', 'ped-yield'),
    'joint-near-clear': ('yield-close', 'ped-clear'),
    'joint-far-busy': ('yield-gap', 'ped-yield'),
    'joint-far-clear': ('yield-gap', 'ped-clear'),
    'joint-varied': ('yield-varied', 'ped-timed'),
}
JOINT_SCENARIOS = tuple(JOINT_TRAFFIC)
MID_TRAFFIC = {'joint-mid-open': ('yield-mid', 'ped-mid-open'), 'joint-mid-busy': ('yield-mid', 'ped-mid-busy')}
JOINT_TRAFFIC.update(MID_TRAFFIC)
GAP_SCENARIOS = tuple(JOINT_TRAFFIC)
LEFT_SCENARIOS = YIELD_SCENARIOS + PED_SCENARIOS + GAP_SCENARIOS
SCENARIOS = ('mixed', 'single', 'joint', 'gaps', 'vehicles', 'pedestrians', 'signals',
             'standard', 'red', 'green', 'yellow-stop', 'yellow-go',
             *LEFT_SCENARIOS)


class SignalMapManager(SumoMapManager):
    """Match the ego route to the SUMO lane and traffic-light link."""

    def __init__(self, net_file=SIGNAL_NET_FILE):
        super().__init__(str(net_file))
        self.requested_movement = 'straight'
        self.light = None
        self.light_color = None

    def reset(self):
        super().reset()
        net = self.graph.sumo_net
        spawn_lane = self.engine.global_config['agent_configs']['default_agent']['spawn_lane_index']
        self.source = net.getLane(spawn_lane.removeprefix('lane_'))
        direction = {'straight': 's', 'left': 'l'}[self.requested_movement]
        connection = next(link for link in self.source.getOutgoing() if link.getDirection() == direction)
        self.tls_id = connection.getTLSID()
        self.link_index = connection.getTLLinkIndex()
        if not self.tls_id or self.link_index < 0:
            raise ValueError("The selected SUMO movement must have a traffic light.")

        self.destination = connection.getToLane()
        lanes = [self.source]
        while connection.getViaLaneID():
            via = net.getLane(connection.getViaLaneID())
            lanes.append(via)
            connection = next(link for link in via.getOutgoing() if link.getToLane() == self.destination)
        lanes.append(self.destination)
        points = np.concatenate([lane.getShape() for lane in lanes])
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]
        # At a lane boundary, use the speed of the segment ending at the retained point.
        point_speeds = np.concatenate([np.full(len(lane.getShape()), lane.getSpeed()) for lane in lanes])
        self.segment_speeds = point_speeds[keep][1:]
        points = points[keep]
        segments = np.diff(points, axis=0)
        self.segment_starts = points[:-1]
        self.segment_lengths = np.linalg.norm(segments, axis=1)
        self.directions = segments / self.segment_lengths[:, None]
        self.segment_headings = np.arctan2(self.directions[:, 1], self.directions[:, 0])
        self.arc_lengths = np.r_[0., np.cumsum(self.segment_lengths)]
        self.route_length = float(self.arc_lengths[-1])
        self.current_sdc_route = PointLane(points, width=self.source.getWidth(), speed_limit=3.6 * self.source.getSpeed())
        self.stop_position = np.asarray(self.source.getShape()[-1], dtype=float)
        self.stop_longitude, _, _ = self.project(self.stop_position)
        self.light = self.spawn_object(BaseTrafficLight, lane=self.current_sdc_route, position=self.stop_position)

    def project(self, position):
        """Return longitudinal distance, leftward offset and heading on the nearest segment."""
        offsets = np.asarray(position) - self.segment_starts
        along = (offsets * self.directions).sum(axis=1).clip(0, self.segment_lengths)
        residuals = offsets - self.directions * along[:, None]
        i = int(np.argmin((residuals * residuals).sum(axis=1)))
        dx, dy = self.directions[i]
        lateral = dx * residuals[i, 1] - dy * residuals[i, 0]
        return float(self.arc_lengths[i] + along[i]), float(lateral), float(self.segment_headings[i])

    def _segment_index(self, longitude):
        i = self.arc_lengths.searchsorted(longitude, side='right') - 1
        return min(max(i, 0), len(self.segment_headings) - 1)

    def heading_at(self, longitude):
        return float(self.segment_headings[self._segment_index(longitude)])

    def speed_limit_at(self, longitude):
        return float(self.segment_speeds[self._segment_index(longitude)])

    def before_reset(self):
        if self.light is not None:
            self.clear_objects([self.light.id], force_destroy=True)
            self.light = None
            self.light_color = None
        super().before_reset()

    def set_light_color(self, color):
        if color != self.light_color:
            {"red": self.light.set_red, "yellow": self.light.set_yellow, "green": self.light.set_green}[color]()
            self.light_color = color


class SumoSignalClock:
    """Step SUMO once per MetaDrive decision."""

    def __init__(self, net_path, tls_id, link_index, dt=0.1):
        if libsumo.isLoaded():
            raise RuntimeError('Close the active SUMO environment before opening another.')
        self.sumo = libsumo
        self.tls_id = tls_id
        self.dt = dt
        self.closed = False
        libsumo.start([
            'sumo', '--net-file', str(net_path), '--step-length', str(dt),
            '--no-step-log', 'true', '--duration-log.disable', 'true',
        ])
        try:
            lights = libsumo.trafficlight
            active_program = lights.getProgram(tls_id)
            program = next(p for p in lights.getAllProgramLogics(tls_id)
                           if p.programID == active_program)
            if program.type != 0:
                raise ValueError('Episode offsets require a fixed-time SUMO program.')
            self.phases = program.phases
            self.link_count = len(lights.getControlledLinks(tls_id))
            self.select_link(link_index)
            self.cycle_seconds = sum(p.duration for p in self.phases)
        except Exception:
            self.close()
            raise

    def select_link(self, link_index):
        if not 0 <= link_index < self.link_count:
            raise ValueError(f'Invalid signal link {link_index} for {self.tls_id}.')
        if any(p.state[link_index].lower() not in ('r', 'y', 'g') for p in self.phases):
            raise ValueError('The ego movement must use red, yellow and green indications.')
        self.link_index = link_index

    def reset(self, offset_seconds):
        elapsed = offset_seconds % self.cycle_seconds
        # Floor to the current signal phase at a decision boundary.
        elapsed = math.floor(round(elapsed / self.dt, 10)) * self.dt
        lights = self.sumo.trafficlight
        for index, phase in enumerate(self.phases):
            if elapsed < phase.duration:
                lights.setPhase(self.tls_id, index)
                lights.setPhaseDuration(self.tls_id, phase.duration - elapsed)
                # SUMO reports the last simulated interval.
                return self.step()
            elapsed -= phase.duration
        raise ValueError('Supply a finite phase offset in seconds.')

    def step(self):
        self.sumo.simulationStep()
        state = self.sumo.trafficlight.getRedYellowGreenState(self.tls_id)
        return state[self.link_index].lower()

    def close(self):
        if not self.closed:
            self.sumo.close()
            self.closed = True


class OncomingVehicle:
    """A straight-through vehicle on the opposing approach."""

    def __init__(self, manager, rng, eta_seconds):
        from shapely.geometry import LineString

        self.manager = manager
        net = manager.graph.sumo_net
        through = next(link for link in manager.source.getOutgoing() if link.getDirection() == 's')
        opposing_arm = through.getToLane().getEdge().getToNode().getID()
        inbound = net.getLane(f'{opposing_arm}_in_0')
        opposing_link = next(link for link in inbound.getOutgoing() if link.getDirection() == 's')
        origin, stop = np.asarray(inbound.getShape())[[0, -1]]
        self.tangent = (stop - origin) / np.linalg.norm(stop - origin)
        normal = np.array([-self.tangent[1], self.tangent[0]])
        # Intersect the two swept paths so clearance includes the vehicle bodies.
        points = np.vstack((manager.segment_starts, manager.current_sdc_route.end))
        side = (points - origin) @ normal
        crossings = np.flatnonzero((side[:-1] * side[1:] <= 0) & (np.abs(np.diff(side)) > 1e-8))
        if not len(crossings):
            raise ValueError('No crossing between the selected turn and opposing through lane.')
        i = int(crossings[0])
        fraction = side[i] / (side[i] - side[i + 1])
        self.conflict_point = points[i] + fraction * (points[i + 1] - points[i])
        other_end = np.asarray(opposing_link.getToLane().getShape()[-1])
        half_width = DefaultVehicle.DEFAULT_WIDTH / 2
        swept_paths = LineString(points).buffer(half_width).intersection(
            LineString((origin, other_end)).buffer(half_width))
        vertices = np.asarray(swept_paths.exterior.coords)
        arc_positions = [manager.project(vertex)[0] for vertex in vertices]
        self.ego_entry_s, self.ego_exit_s = min(arc_positions), max(arc_positions)
        along = (vertices - self.conflict_point) @ self.tangent
        self.opp_entry_offset, self.opp_exit_offset = float(along.min()), float(along.max())
        speed_min, speed_max = ONCOMING_SPEED_MPS
        self.cruise_speed = float(rng.uniform(speed_min, min(speed_max, inbound.getSpeed())))
        self.half_length = DefaultVehicle.DEFAULT_LENGTH / 2
        start = self.conflict_point - self.tangent * (eta_seconds * self.cruise_speed + self.half_length)
        if float((start - origin) @ self.tangent) < self.half_length:
            raise ValueError('Oncoming car starts beyond the available SUMO approach.')
        self.exit_distance = float((self.conflict_point - other_end) @ self.tangent)
        self.car = manager.spawn_object(
            DefaultVehicle, position=tuple(start), heading=math.atan2(self.tangent[1], self.tangent[0]),
            vehicle_config={'navigation_module': None, 'show_navi_mark': False, 'show_dest_mark': False},
        )
        self.car.set_velocity(self.tangent * self.cruise_speed)

    @property
    def front_distance(self):
        if self.car is None:
            return self.exit_distance - self.half_length
        return float((self.conflict_point - self.car.position) @ self.tangent) - self.half_length

    @property
    def rear_distance(self):
        return self.front_distance + 2 * self.half_length

    @property
    def speed(self):
        return self.cruise_speed if self.car is None else float(self.car.velocity @ self.tangent)

    def before_step(self):
        if self.car is not None:
            if self.front_distance <= self.exit_distance:
                self.close()
            else:
                self.car.set_velocity(self.tangent * self.cruise_speed)

    def close(self):
        if self.car is not None:
            self.manager.clear_objects([self.car.id], force_destroy=True)
            self.car = None


class CrossingPedestrian:
    """A pedestrian on the exit crosswalk."""

    width = CROSSWALK_WIDTH_M

    def __init__(self, manager, rng, scenario, dt):
        from metadrive.component.traffic_participants.pedestrian import Pedestrian

        self.manager = manager
        self.dt = dt
        self.radius = Pedestrian.RADIUS
        destination = manager.destination
        points = np.asarray(destination.getShape(), dtype=float)
        self.tangent = (points[1] - points[0]) / np.linalg.norm(points[1] - points[0])
        self.center = points[0] + CROSSWALK_OFFSET_M * self.tangent
        self.crosswalk_s = manager.project(self.center)[0]
        normal = np.array([-self.tangent[1], self.tangent[0]])
        sign = 1 if rng.integers(2) else -1
        self.direction = sign * normal
        lane_width = destination.getWidth()
        half_lane = lane_width / 2
        far_curb = 1.5 * lane_width
        end_offset = far_curb + SIDEWALK_MARGIN_M if sign == 1 else -half_lane - SIDEWALK_MARGIN_M
        self.end = self.center + normal * end_offset
        # Paired scenes share walking pace and direction and differ only in starting position.
        curb_distance = (half_lane if sign == 1 else far_curb) - rng.uniform(*CURB_START_JITTER_M)
        time_from_curb = rng.uniform(*WALK_CLEARANCE_S['ped-yield'])
        self.walk_speed = (curb_distance + half_lane + self.radius) / time_from_curb
        if scenario == 'ped-yield':
            distance = curb_distance
        else:
            distance = self.walk_speed * rng.uniform(*WALK_CLEARANCE_S[scenario]) - half_lane - self.radius
        position = self.center - self.direction * distance
        self.person = manager.spawn_object(
            Pedestrian, position=tuple(position), heading_theta=math.atan2(self.direction[1], self.direction[0]),
        )
        self.person.body.setDeactivationEnabled(False)
        self.person.body.setAngularFactor((0.0, 0.0, 0.0))
        self.person.set_velocity(self.direction, self.walk_speed)
        self.markings = self.paint(normal, lane_width) if manager.engine.global_config['use_render'] else None

    def paint(self, normal, lane_width):
        from panda3d.core import CardMaker

        marks = self.manager.engine.worldNP.attachNewNode('exit-crosswalk')
        card = CardMaker('crosswalk-stripe')
        card.setFrame(-self.width / 2, self.width / 2, -0.19, 0.19)
        for offset in np.arange(-lane_width / 2 + 0.35, 1.5 * lane_width, 0.7):
            stripe = marks.attachNewNode(card.generate())
            point = self.center + offset * normal
            stripe.setPos(float(point[0]), float(point[1]), 0.03)
            stripe.setHpr(math.degrees(math.atan2(self.tangent[1], self.tangent[0])), -90, 0)
        marks.setTwoSided(True)
        marks.setLightOff()
        marks.setColor(0.96, 0.96, 0.92, 1.0)
        return marks

    @property
    def position(self):
        return np.asarray(self.person.position)

    @property
    def front_distance(self):
        return float((self.center - self.position) @ self.direction)

    @property
    def speed(self):
        return float(self.person.velocity @ self.direction)

    def before_step(self):
        remaining = max(0.0, float((self.end - self.position) @ self.direction))
        target_speed = min(self.walk_speed, remaining / self.dt)
        if abs(self.speed - target_speed) > 1e-4:
            self.person.set_velocity(self.direction, target_speed)

    def close(self):
        if self.markings is not None:
            self.markings.removeNode()
            self.markings = None
        if self.person is not None:
            self.manager.clear_objects([self.person.id], force_destroy=True)
            self.person = None


class CrosswalkEncounter:
    """Measure when the pedestrian clears the lane and the car enters the crossing."""

    def __init__(self, env, walker):
        self.env = env
        self.walker = walker
        self.half_lane = env.engine.map_manager.destination.getWidth() / 2
        self.time = self.stop_seconds = 0.0
        self.entered_at = None
        self.stopped = self.resumed = self.violation = self.collision = False
        self.clearance_seconds = None
        self.distance_to_crosswalk, self.distance_to_clear = self.distances()
        self.started_clear = self.distance_to_clear <= 0.0
        self.cleared_at = 0.0 if self.started_clear else None
        if self.distance_to_crosswalk <= 0.0:
            raise ValueError('Start the car upstream of the pedestrian crossing.')

    def distances(self):
        car, walker = self.env.vehicle, self.walker
        direction = np.asarray(car.heading)
        across = np.array([-direction[1], direction[0]])
        half_extent = (car.LENGTH * abs(direction @ walker.tangent)
                       + car.WIDTH * abs(across @ walker.tangent)) / 2
        to_stripe = float((walker.center - np.asarray(car.position)) @ walker.tangent) - walker.width / 2 - half_extent
        to_clear = walker.front_distance + self.half_lane + walker.radius
        return to_stripe, to_clear

    def advance(self):
        dt = self.env.dt
        stripe, clear = self.distances()
        if self.cleared_at is None and clear <= 0:
            self.cleared_at = self.time + dt * self.distance_to_clear / (self.distance_to_clear - clear)
        if self.entered_at is None and stripe <= 0:
            self.entered_at = self.time + dt * self.distance_to_crosswalk / (self.distance_to_crosswalk - stripe)
            self.violation = self.cleared_at is None or self.entered_at < self.cleared_at
            self.resumed = self.stopped and not self.violation
            if self.cleared_at is not None and not self.started_clear:
                self.clearance_seconds = self.entered_at - self.cleared_at
        if self.entered_at is None and 0 < stripe < STOP_ZONE_M and self.env.vehicle.speed <= STOPPED_SPEED_MPS:
            self.stop_seconds += dt
            self.stopped |= self.stop_seconds >= CROSSWALK_STOP_TIME_S - 1e-9
        else:
            self.stop_seconds = 0.0
        self.collision |= bool(self.env.vehicle.crash_human)
        self.violation |= self.collision
        self.distance_to_crosswalk, self.distance_to_clear = stripe, clear
        self.time += dt
        return self.violation

    def info(self):
        return dict(ped_violation=self.violation, ped_collision=self.collision,
                    ped_entered=self.entered_at is not None,
                    ped_cleared_first=self.entered_at is not None and not self.violation,
                    ped_stopped=self.stopped, ped_resumed=self.resumed,
                    ped_clearance_seconds=self.clearance_seconds)


class JunctionObservation(BaseObservation):
    size = 35

    @property
    def observation_space(self):
        return gym.spaces.Box(-1.0, 1.0, shape=(self.size,), dtype=np.float32)

    def reset(self, env, vehicle=None):
        self.env = env

    def observe(self, vehicle):
        env = self.env
        longitude, lateral, heading, _, distance = env.pose(vehicle)
        return np.clip(np.array([
            lateral / env.half_lane, math.sin(heading), vehicle.speed / env.speed_limit,
            distance / 65.0, distance / 12.0, float(distance <= 0),
            float(env.signal_color == 'r'), float(env.signal_color == 'y'),
            float(env.signal_color == 'g'), float(env.yellow_clearance),
            float(env.movement == 'left'), math.sin(env.preview_bearing(longitude, 4.0)),
            math.sin(env.preview_bearing(longitude, 9.0)),
            (longitude - env.stop_at) / 25.0,
            *env.traffic_observation(longitude),
            *env.walker_observation(longitude),
            *env.joint_observation(longitude),
        ][:self.size], dtype=np.float32), -1, 1)


class PedestrianObservation(JunctionObservation):
    """Input layout of the pedestrian checkpoint."""
    size = 27


class GapFeatures(BaseFeaturesExtractor):
    """Append the gap decision learned from the two clearance margins."""

    def __init__(self, observation_space, min_gap, max_walk_margin):
        super().__init__(observation_space, features_dim=observation_space.shape[0] + 1)
        self.min_gap = min_gap
        self.max_walk_margin = max_walk_margin

    def forward(self, observations):
        joint = observations[:, 27:28] > 0.5
        opposing_gap = observations[:, 28:29] * 6.0
        walk_margin = observations[:, 29:30] * 2.5
        opportunity = joint & (opposing_gap >= self.min_gap) & (walk_margin <= self.max_walk_margin)
        return torch.cat((observations, opportunity.to(observations.dtype)), dim=1)


def vehicle_corners(vehicle):
    forward = np.asarray(vehicle.heading)
    side = np.array([-forward[1], forward[0]])
    half_length, half_width = forward * (vehicle.LENGTH / 2), side * (vehicle.WIDTH / 2)
    return (vehicle.position + end + edge for end in (half_length, -half_length)
            for edge in (half_width, -half_width))


class YieldEncounter:
    """Measure when each vehicle enters and clears the shared conflict zone."""

    def __init__(self, env):
        self.env = env
        self.time = 0.0
        self.times = [None] * 4
        self.previous = self.distances()
        self.initial_gap = env.projected_gap(env.pose(env.vehicle)[0])
        self.order = ''
        self.gap = None
        self.overlap = False

    @property
    def turn_window(self):
        """Sampling coordinate for offline gap calibration."""
        slack = min(self.initial_gap - PROBE_GAP_CENTER_S, PROBE_WALK_CENTER_S - self.initial_walk_margin)
        return float(np.clip((slack + PROBE_HALF_WIDTH_S) / (2 * PROBE_HALF_WIDTH_S), 0.0, 1.0))

    def fits_gap(self, limits):
        min_gap, max_walk_margin = limits
        return self.initial_gap >= min_gap and self.initial_walk_margin <= max_walk_margin

    def distances(self):
        npc = self.env.opponent
        project = self.env.engine.map_manager.project
        positions = [project(corner)[0] for corner in vehicle_corners(self.env.vehicle)]
        return (npc.ego_entry_s - max(positions), npc.ego_exit_s - min(positions),
                npc.front_distance + npc.opp_entry_offset, npc.rear_distance + npc.opp_exit_offset)

    def advance(self):
        now = self.distances()
        dt = self.env.dt
        for i, (before, after) in enumerate(zip(self.previous, now)):
            if self.times[i] is None and before > 0 >= after:
                self.times[i] = self.time + dt * before / (before - after)
        self.time += dt
        self.previous = now
        ego_in, ego_out, other_in, other_out = self.times
        if ego_in is None or other_in is None:
            return False
        ego_first = ego_in < other_in
        self.order = 'ego-first' if ego_first else 'oncoming-first'
        first_exit, second_entry = (ego_out, other_in) if ego_first else (other_out, ego_in)
        self.gap = None if first_exit is None else second_entry - first_exit
        self.overlap = first_exit is None or self.gap < 0.0
        return self.overlap or (ego_first and self.gap < EGO_FIRST_CLEARANCE_S)


class SignalEnv(BaseEnv):
    """Drive straight or turn left at a four-arm SUMO-controlled junction."""

    def __init__(self, render=False, *, approach=None, scenario='mixed', movement='mixed',
                 observation=JunctionObservation, gap_curriculum=True):
        self.clock = None
        self.opponent = self.encounter = None
        self.walker = self.crosswalk = None
        self.fixed_approach = approach
        self.mixed_joint_cases = GAP_SCENARIOS if gap_curriculum else JOINT_SCENARIOS
        self.default_scenario = scenario
        self.default_movement = movement
        self.scenarios = np.random.default_rng(ENV_SEED)
        super().__init__(dict(
            use_render=render, log_level=logging.ERROR, agent_observation=observation,
            num_scenarios=NUM_SCENARIOS, horizon=EPISODE_STEPS,
            physics_world_step_size=PHYSICS_STEP_S, decision_repeat=PHYSICS_STEPS_PER_ACTION,
            vehicle_config=dict(navigation_module=None, show_navi_mark=False, show_dest_mark=False,
                                spawn_velocity_car_frame=True),
            agent_configs={'default_agent': dict(spawn_lane_index='lane_west_in_0')},
        ))
        self.dt = self.config['physics_world_step_size'] * self.config['decision_repeat']

    def setup_engine(self):
        super().setup_engine()
        self.engine.register_manager('map_manager', SignalMapManager(SIGNAL_NET_FILE))

    def select_scenario(self, seed, options):
        arm_count = len(APPROACHES)
        self.approach = options.get('approach', self.fixed_approach) or APPROACHES[seed % arm_count]
        if self.approach not in APPROACHES:
            raise ValueError(f'Unknown junction approach: {self.approach}')
        self.movement = options.get('movement', self.default_movement)
        self.scenario = options.get('scenario', self.default_scenario)
        if self.scenario not in SCENARIOS:
            raise ValueError(f'Unknown scenario: {self.scenario}')
        arm_cycle = int(seed) // arm_count
        suite = {'pedestrians': PED_SCENARIOS, 'joint': JOINT_SCENARIOS, 'gaps': GAP_SCENARIOS}.get(self.scenario)
        if suite:
            self.scenario = suite[arm_cycle % len(suite)]
        if self.movement == 'mixed':
            self.movement = 'left' if self.scenario in LEFT_SCENARIOS else ('straight', 'left')[arm_cycle % 2]
        if self.movement not in ('straight', 'left'):
            raise ValueError(f'Unknown movement: {self.movement}')
        block = arm_cycle // 2
        traffic_slot = block % 8
        if self.scenario == 'mixed' and self.movement == 'left' and block % 3 == 0:
            self.scenario = self.mixed_joint_cases[(block // 3) % len(self.mixed_joint_cases)]
        elif self.scenario in ('mixed', 'single') and self.movement == 'left' and block % 16 >= 12:
            self.scenario = ('ped-yield', 'ped-clear', 'ped-timed', 'ped-yield')[traffic_slot % 4]
        elif self.scenario in ('mixed', 'single', 'vehicles') and self.movement == 'left' and traffic_slot < 5:
            self.scenario = YIELD_SCENARIOS[min(traffic_slot, 2)]
        elif self.scenario in ('mixed', 'single', 'vehicles', 'signals'):
            self.scenario = SCENARIO_CYCLE[block % len(SCENARIO_CYCLE)]
        self.car_case, self.walk_case = JOINT_TRAFFIC.get(self.scenario, (
            self.scenario if self.scenario in YIELD_SCENARIOS else None,
            self.scenario if self.scenario in PED_SCENARIOS else None,
        ))
        if self.scenario in LEFT_SCENARIOS and self.movement != 'left':
            raise ValueError('This scenario requires a left turn.')

    def reset(self, seed=None, options=None):
        self.clear_traffic()
        if seed is None:
            seed = int(self.scenarios.integers(*ROLLOUT_SEEDS))
        self.select_scenario(seed, options or {})
        self.lazy_init()
        self.engine.map_manager.requested_movement = self.movement
        rng = np.random.default_rng(seed)
        spawn = self.config['agent_configs']['default_agent']
        spawn['spawn_lane_index'] = f'lane_{self.approach}_in_0'
        spawn['spawn_longitude'] = float(rng.uniform(*SPAWN_LONGITUDE_M))
        spawn['spawn_lateral'] = float(rng.uniform(*SPAWN_LATERAL_M))
        spawn['spawn_velocity'] = (float(rng.uniform(*SPAWN_SPEED_MPS)), 0.0)
        self.phase_sample = float(rng.random())
        if self.scenario in LEFT_SCENARIOS:
            lane = self.engine.map_manager.graph.sumo_net.getLane(f'{self.approach}_in_0')
            bounds = PEDESTRIAN_APPROACH_M if self.scenario in PED_SCENARIOS else ONCOMING_APPROACH_M
            distance = rng.uniform(*bounds)
            spawn['spawn_longitude'] = float(lane.getLength() - DefaultVehicle.DEFAULT_LENGTH / 2 - distance)
            spawn['spawn_velocity'] = (float(rng.uniform(*TURN_SPAWN_SPEED_MPS)), 0.0)
            self.traffic_rng = rng
            if self.car_case:
                self.opponent_eta = rng.uniform(*ARRIVAL_WINDOWS[self.car_case])
        elif self.scenario != 'standard':
            self.set_short_approach(spawn, rng)
        self.waited = self.restarted = self.had_overspeed = False
        self.wait_duration = 0.0
        self.cross_color = ''
        self.yellow_clearance = False
        self.step_info = {}
        return super().reset(seed=int(seed))

    def set_short_approach(self, spawn, rng):
        lane = self.engine.map_manager.graph.sumo_net.getLane(f'{self.approach}_in_0')
        speed = float(rng.uniform(*SIGNAL_APPROACH_SPEED_RATIO) * lane.getSpeed())
        stopping_distance = self.yellow_stopping_distance(speed)
        self.lead_ticks = int(rng.integers(*YELLOW_LEAD_TICKS))
        if self.scenario.startswith('yellow'):
            margin = (rng.uniform(*STOPPING_SURPLUS_M) if self.scenario == 'yellow-stop'
                      else -rng.uniform(*STOPPING_SHORTFALL_M))
            distance = stopping_distance + speed * self.lead_ticks * self.dt + margin
        else:
            distance = stopping_distance + rng.uniform(*STOPPING_SURPLUS_M)
        spawn['spawn_longitude'] = float(lane.getLength() - DefaultVehicle.DEFAULT_LENGTH / 2 - distance)
        spawn['spawn_velocity'] = (speed, 0.0)

    def initial_phase_offset(self):
        ticks = round(self.clock.cycle_seconds / self.dt)
        if self.scenario == 'standard':
            return int(ticks * self.phase_sample) * self.dt
        target = 'y' if self.scenario.startswith('yellow') else 'g'
        start = 0.0
        for phase in self.clock.phases:
            if phase.state[self.clock.link_index].lower() == target:
                break
            start += phase.duration
        if target == 'y':
            return start - self.lead_ticks * self.dt
        if self.scenario == 'red':
            return start - (RED_WAIT_BASE_S + int(RED_START_JITTER_TICKS * self.phase_sample) * self.dt)
        if self.scenario in LEFT_SCENARIOS:
            return start
        return start + int(GREEN_START_JITTER_TICKS * self.phase_sample) * self.dt

    def _get_reset_return(self, reset_info):
        manager = self.engine.map_manager
        self.length = manager.route_length
        self.entry_heading = manager.heading_at(0)
        self.entry_tangent = np.array([math.cos(self.entry_heading), math.sin(self.entry_heading)])
        self.stop_at = manager.stop_longitude
        lane = manager.source
        self.speed_limit = lane.getSpeed()
        self.half_lane = lane.getWidth() / 2
        if self.clock is None:
            self.clock = SumoSignalClock(SIGNAL_NET_FILE, manager.tls_id, manager.link_index, dt=self.dt)
        else:
            self.clock.select_link(manager.link_index)
        self.phase_offset = self.initial_phase_offset() % self.clock.cycle_seconds
        self.signal_color = self.clock.reset(self.phase_offset)
        if self.car_case:
            self.opponent = OncomingVehicle(manager, self.traffic_rng, self.opponent_eta)
            self.encounter = YieldEncounter(self)
        if self.walk_case:
            self.walker = CrossingPedestrian(manager, self.traffic_rng, self.walk_case, self.dt)
            self.crosswalk = CrosswalkEncounter(self, self.walker)
        if self.encounter is not None and self.crosswalk is not None:
            self.encounter.initial_walk_margin = self.crosswalk_braking_margin(self.pose(self.vehicle)[0])
            self.encounter.exit_blocked_at_start = self.exit_wait_blocks_turn()
        self.update_yellow(self.vehicle)
        self.show_signal()
        return super()._get_reset_return(reset_info)

    def pose(self, vehicle):
        manager = self.engine.map_manager
        position = vehicle.position
        longitude, lateral, lane_heading = manager.project(position)
        yaw = vehicle.heading_theta
        entry_yaw = yaw - self.entry_heading
        half_front = (vehicle.LENGTH * abs(math.cos(entry_yaw)) + vehicle.WIDTH * abs(math.sin(entry_yaw))) / 2
        distance = float((manager.stop_position - position) @ self.entry_tangent) - half_front
        vx, vy = vehicle.velocity
        forward_speed = vx * math.cos(lane_heading) + vy * math.sin(lane_heading)
        return longitude, lateral, yaw - lane_heading, forward_speed, distance

    def outside_lane(self, vehicle, longitude, lateral, heading):
        if self.movement == 'straight' or abs(longitude - self.stop_at) > 25:
            half_width = (vehicle.WIDTH * abs(math.cos(heading)) + vehicle.LENGTH * abs(math.sin(heading))) / 2
            return abs(lateral) + half_width > self.half_lane
        project = self.engine.map_manager.project
        return any(abs(project(corner)[1]) > self.half_lane for corner in vehicle_corners(vehicle))

    def yellow_stopping_distance(self, speed):
        return speed * self.dt + speed**2 / (2 * YELLOW_BRAKING_MPS2)

    def update_yellow(self, vehicle):
        self.yellow_clearance = (self.signal_color == 'y'
                                 and self.pose(vehicle)[-1] < self.yellow_stopping_distance(vehicle.speed))

    def show_signal(self):
        self.engine.map_manager.set_light_color({'r': 'red', 'y': 'yellow', 'g': 'green'}[self.signal_color])

    def traffic_observation(self, longitude):
        car = self.opponent
        if car is None:
            return (0.0,) * 7
        distance_to_entry = car.front_distance + car.opp_entry_offset
        distance_to_exit = car.rear_distance + car.opp_exit_offset
        return (1.0, distance_to_entry / 65.0, car.speed / 9.0, distance_to_exit / 10.0,
                self.projected_gap(longitude) / 6.0, self.encounter.initial_gap / 6.0,
                self.encounter.previous[0] / 10.0)

    def projected_gap(self, longitude):
        car = self.opponent
        remaining = car.ego_exit_s - longitude + self.vehicle.LENGTH / 2
        ego_time = remaining / TURN_SPEED_MPS + max(0.0, TURN_SPEED_MPS - self.vehicle.speed) / GAP_ESTIMATED_ACCEL_MPS2
        other_time = (car.front_distance + car.opp_entry_offset) / max(0.1, car.speed)
        return other_time - ego_time

    def route_distance_to_stripe(self, longitude):
        return self.walker.crosswalk_s - self.walker.width / 2 - longitude - self.vehicle.LENGTH / 2

    def walker_observation(self, longitude):
        if self.walker is None:
            return (0.0,) * 6
        distance = self.route_distance_to_stripe(longitude)
        remaining = self.crosswalk.distance_to_clear
        return (1.0, distance / 30.0, distance / 7.0, remaining / 12.0,
                remaining / 3.0, self.walker.speed / 2.0)

    def crosswalk_braking_margin(self, longitude):
        """Seconds the walker still needs when the turning car would start braking."""
        braking_distance = STOP_BUFFER_M + TURN_SPEED_MPS**2 / (2 * STOP_DECEL_MPS2)
        brake_time = max(0.0, self.route_distance_to_stripe(longitude) - braking_distance) / TURN_SPEED_MPS
        walk_time = max(0.0, self.crosswalk.distance_to_clear + WALK_CLEAR_BUFFER_M) / max(self.walker.speed, 0.1)
        return walk_time - brake_time

    def joint_observation(self, longitude):
        if self.encounter is None or self.crosswalk is None:
            return (0.0,) * 8
        npc = self.opponent
        remaining = self.encounter.previous
        return (1.0, self.encounter.initial_gap / 6.0, self.encounter.initial_walk_margin / 2.5,
                remaining[0] / 12.0, remaining[3] / max(npc.speed, 0.1) / 10.0,
                self.route_distance_to_stripe(longitude) / 15.0, self.crosswalk.distance_to_clear / 6.0,
                (self.projected_gap(longitude) - max(0.0, self.crosswalk_braking_margin(longitude))) / 6.0)

    def _step_simulator(self, actions):
        before, _, _, _, old_distance = self.pose(self.vehicle)
        old_color = self.signal_color
        clearance = self.yellow_clearance
        if self.opponent is not None:
            self.opponent.before_step()
        if self.walker is not None:
            self.walker.before_step()
        engine_info = super()._step_simulator(actions)
        vehicle = self.vehicle
        longitude, lateral, heading, forward_speed, distance = self.pose(vehicle)
        crossing = not self.cross_color and old_distance > 0 >= distance
        if crossing:
            self.cross_color = old_color
        red = crossing and old_color == 'r'
        yellow = crossing and old_color == 'y' and not clearance
        out_of_lane = self.outside_lane(vehicle, longitude, lateral, heading)
        wrong_way = forward_speed < -REVERSE_SPEED_TOLERANCE_MPS or math.cos(heading) <= 0
        collision = bool(vehicle.crash_vehicle or vehicle.crash_human or vehicle.crash_object
                         or vehicle.crash_sidewalk or vehicle.crash_building)
        encounter = self.encounter
        yield_violation = encounter.advance() if encounter is not None else False
        ped_violation = self.crosswalk.advance() if self.crosswalk is not None else False
        failed = red or yellow or yield_violation or ped_violation or out_of_lane or wrong_way or collision
        arrived = longitude >= self.length - FINISH_TOLERANCE_M and not failed
        self.speed_limit = self.engine.map_manager.speed_limit_at(longitude)
        overspeed = vehicle.speed > self.speed_limit
        self.had_overspeed |= overspeed
        if old_color == 'r' and 0 < distance <= STOP_ZONE_M and vehicle.speed <= STOPPED_SPEED_MPS:
            self.wait_duration += self.dt
            self.waited |= self.wait_duration >= RED_WAIT_MIN_S
        else:
            self.wait_duration = 0.0
        self.restarted |= self.waited and old_color == 'g' and crossing
        progress = min(longitude - before, self.speed_limit * self.dt)
        reward = PROGRESS_REWARD_PER_M * progress - self.dt * OVERSPEED_PENALTY * max(0, vehicle.speed - self.speed_limit)
        if failed:
            reward = FAILURE_REWARD
        elif arrived:
            reward += ARRIVAL_BONUS
        self.step_info = dict(
            arrived=arrived, compliant=arrived and not self.had_overspeed,
            red_violation=red, yellow_violation=yellow, out_of_lane=out_of_lane,
            wrong_way=wrong_way, collision=collision, overspeed=overspeed, had_overspeed=self.had_overspeed,
            ped_violation=ped_violation, ped_collision=bool(vehicle.crash_human),
            waited=self.waited, restarted=self.restarted, crossing_color=self.cross_color,
            failed=failed, reward=reward, signal=old_color, stop_distance=distance,
            yield_violation=yield_violation, conflict_overlap=bool(encounter and encounter.overlap),
            conflict_order=encounter.order if encounter else '',
            conflict_gap_seconds=encounter.gap if encounter else None,
        )
        if self.crosswalk is not None:
            self.step_info.update(self.crosswalk.info())
        self.signal_color = self.clock.step()
        if self.signal_color != old_color:
            self.update_yellow(vehicle)
            self.show_signal()
        return engine_info

    def reward_function(self, vehicle_id):
        return float(self.step_info.get('reward', 0)), self.step_info

    def done_function(self, vehicle_id):
        done = self.step_info.get('failed', False) or self.step_info.get('arrived', False)
        return bool(done), {'max_step': self.episode_lengths[vehicle_id] >= self.config['horizon'] and not done}

    def cost_function(self, vehicle_id):
        return 0.0, {}

    def preview_bearing(self, longitude, lookahead):
        """Direction to a point ahead on the route, relative to the vehicle."""
        route = self.engine.map_manager.current_sdc_route
        target = route.position(min(longitude + lookahead, self.length), 0)
        offset = target - self.vehicle.position
        angle = math.atan2(offset[1], offset[0]) - self.vehicle.heading_theta
        return math.atan2(math.sin(angle), math.cos(angle))

    def exit_wait_blocks_turn(self):
        """Predict whether a crosswalk stop would leave the car in the opposing path."""
        stripe_s = self.walker.crosswalk_s - self.walker.width / 2
        room = stripe_s - self.opponent.ego_exit_s - self.vehicle.LENGTH
        return room < STOP_BUFFER_M and self.encounter.initial_walk_margin >= -TEACHER_WALK_LEAD_S

    def teacher_action(self, gap_limits=None):
        longitude, lateral, heading, _, distance = self.pose(self.vehicle)
        steer = -TEACHER_LATERAL_GAIN * lateral - TEACHER_HEADING_GAIN * math.sin(heading)
        speed = self.vehicle.speed
        target = TEACHER_CRUISE_RATIO * self.speed_limit
        if self.movement == 'left':
            bearing = self.preview_bearing(longitude, TURN_LOOKAHEAD_M)
            wheelbase = self.vehicle.FRONT_WHEELBASE + self.vehicle.REAR_WHEELBASE
            angle = math.atan(2 * wheelbase * math.sin(bearing) / TURN_LOOKAHEAD_M)
            steer = angle / math.radians(self.vehicle.max_steering)
            turn_end = self.length - self.engine.map_manager.destination.getLength()
            if longitude < turn_end:
                approach = max(0.0, self.stop_at - longitude - TURN_APPROACH_BUFFER_M)
                target = min(target, math.sqrt(TURN_SPEED_MPS**2 + 2 * TURN_APPROACH_DECEL_MPS2 * approach))

        stop_for_light = self.signal_color == 'r' or (self.signal_color == 'y' and not self.yellow_clearance)
        stops = [distance] if stop_for_light and distance > 0 else []
        car = self.opponent
        if car is not None:
            entry = car.ego_entry_s - longitude - self.vehicle.LENGTH / 2
            must_yield = self.encounter.initial_gap < TEACHER_MIN_CAR_GAP_S
            if self.walker is not None:
                if gap_limits is not None:
                    must_yield = not self.encounter.fits_gap(gap_limits)
                else:
                    must_yield |= self.encounter.exit_blocked_at_start
            if must_yield and entry > 0 and car.rear_distance + car.opp_exit_offset > -OPPONENT_CLEAR_BUFFER_M:
                stops.append(entry)
        if self.walker is not None and self.crosswalk.distance_to_clear > -WALK_CLEAR_BUFFER_M:
            stripe = self.route_distance_to_stripe(longitude)
            if stripe > 0:
                stops.append(stripe)
        if stops:
            distance = min(stops)
            target = min(target, math.sqrt(2.0 * STOP_DECEL_MPS2 * max(0, distance - STOP_BUFFER_M)))
        throttle = TEACHER_SPEED_GAIN * (target - speed)
        if stops and distance <= TEACHER_HOLD_DISTANCE_M:
            throttle = TEACHER_STOP_COMMAND
        return np.clip([steer, throttle], -1, 1).astype(np.float32)

    def clear_traffic(self):
        if self.walker is not None:
            self.walker.close()
            self.walker = self.crosswalk = None
        if self.opponent is not None:
            self.opponent.close()
            self.opponent = self.encounter = None

    def close(self):
        try:
            self.clear_traffic()
            if self.clock is not None:
                self.clock.close()
                self.clock = None
        finally:
            super().close()


def extend_actor(actor, checkpoint):
    """Keep the old policy fixed and train only the added input columns."""
    base = checkpoint if isinstance(checkpoint, SAC) else SAC.load(checkpoint, device='cpu')
    if base.observation_space.shape not in ((21,), (27,), (35,)) or base.action_space.shape != (2,):
        raise ValueError('--extend requires a 21-, 27- or 35-input signal policy.')
    state = base.actor.state_dict()
    first = actor.latent_pi[0].weight
    old_weight = state['latent_pi.0.weight']
    width = old_weight.shape[1]
    if width >= first.shape[1]:
        raise ValueError('The checkpoint already uses the requested gap features.')
    expanded = torch.zeros_like(first)
    expanded[:, :width] = old_weight
    state['latent_pi.0.weight'] = expanded
    actor.load_state_dict(state)
    actor.requires_grad_(False)
    first.requires_grad_(True)
    mask = torch.ones_like(first)
    mask[:, :width] = 0.0
    return first.register_hook(lambda gradient: gradient * mask)


def calibrate_gap_limits(env, seed):
    """Fit the opposing-car gap and walker delay against measured clearances."""
    rng = np.random.default_rng(seed)
    cases = ('joint-mid-open', 'joint-varied')
    quotas = np.array(CALIBRATION_QUOTAS)
    counts = np.zeros((len(cases), len(quotas)), dtype=int)
    probes = []
    for probe_seed in rng.permutation(np.arange(*CALIBRATION_SEEDS))[:CALIBRATION_SEED_LIMIT]:
        for case_id, case in enumerate(cases):
            if np.all(counts[case_id] == quotas):
                continue
            env.reset(seed=int(probe_seed), options={'scenario': case, 'movement': 'left'})
            window = env.encounter.turn_window
            bucket = 1 + int((len(quotas) - 2) * window) if window > 0 else 0
            if counts[case_id, bucket] == quotas[bucket]:
                continue
            counts[case_id, bucket] += 1
            gap, walk = env.encounter.initial_gap, env.encounter.initial_walk_margin
            # Probe a go-first turn while keeping pedestrian braking enabled.
            while True:
                _, _, done, truncated, info = env.step(env.teacher_action(gap_limits=(-math.inf, math.inf)))
                if done or truncated:
                    break
            car_clearance = info.get('conflict_gap_seconds')
            ped_clearance = info.get('ped_clearance_seconds')
            safe = (not truncated and info['compliant'] and info['conflict_order'] == 'ego-first'
                    and car_clearance is not None and car_clearance >= CALIBRATION_MIN_CAR_GAP_S
                    and ped_clearance is not None and ped_clearance >= CALIBRATION_MIN_PED_GAP_S)
            probes.append((case_id, gap, walk, safe))
        if np.all(counts == quotas):
            break

    case_ids, gaps, walks, safe = np.array(probes).T
    unsafe = ~safe.astype(bool)
    limits = (99.0, -99.0)
    best_support = (0, 0)
    accepted = [0, 0]
    if np.all(counts == quotas):
        # Scan in tenths of a second, staying within the observation's unclipped range.
        gap_ticks = range(min(59, math.ceil(gaps.max() * 10)), max(-59, math.floor(gaps.min() * 10)) - 1, -1)
        walk_ticks = range(max(-24, math.floor(walks.min() * 10)), min(24, math.ceil(walks.max() * 10)) + 1)
        for gap_tick in gap_ticks:
            min_gap = gap_tick / 10
            for walk_tick in walk_ticks:
                max_walk = walk_tick / 10
                guarded = (gaps >= min_gap - CALIBRATION_GUARD_S) & (walks <= max_walk + CALIBRATION_GUARD_S)
                if (guarded & unsafe).any():
                    continue
                inside = (gaps >= min_gap) & (walks <= max_walk)
                by_case = [int(np.sum(inside & (case_ids == i))) for i in range(len(cases))]
                support = (sum(by_case), min(by_case))
                if min(by_case) >= CALIBRATION_MIN_PER_CASE and support > best_support:
                    best_support, limits, accepted = support, (min_gap, max_walk), by_case
    print(f'Gap calibration: {int(safe.sum())}/{len(probes)} robust go-first demonstrations; '
          f'minimum opposing gap {limits[0]:.1f}s, maximum walking delay {limits[1]:.1f}s; '
          f'accepted {accepted[0]} mid-open and {accepted[1]} varied.', flush=True)
    return limits


def collect_demonstrations(env, model, demos, rng, gap_limits):
    states = np.empty((demos, *env.observation_space.shape), dtype=np.float32)
    targets = np.empty((demos, *env.action_space.shape), dtype=np.float32)
    masks = {name: np.zeros(demos, dtype=bool) for name in DEMOS_PER_BATCH}
    totals = dict.fromkeys(('arrived', 'red_violation', 'waited', 'restarted'), 0)
    episodes = ped_episodes = ped_stops = ped_violations = 0
    done = True
    for i in range(demos):
        if done:
            obs, _ = env.reset(seed=int(rng.integers(*DEMO_SEEDS)))
            episode_step = 0
        label = np.clip(env.teacher_action(gap_limits), -1.0, 1.0).astype(np.float32)
        speed = env.vehicle.speed_km_h
        throttle = label[1]
        green = env.signal_color == 'g'
        oncoming = env.opponent is not None and -0.33 < obs[13] < 0.65
        gap = env.encounter is not None and obs[17] > 0 and -0.65 < obs[13] < 0.3
        if gap and gap_limits is not None and env.crosswalk is not None:
            gap = env.encounter.fits_gap(gap_limits)
        at_crosswalk = env.walker is not None and -3 < env.crosswalk.distance_to_crosswalk < 12
        masks['turning'][i] = env.movement == 'left' and abs(obs[13]) < 0.9 and speed > 3
        masks['oncoming'][i] = oncoming
        masks['gap'][i] = gap
        masks['releasing'][i] = oncoming and throttle > 0.35 and speed < 12
        masks['braking'][i] = not green and speed > 1.0 and throttle < -0.05
        masks['signal_wait'][i] = not green and speed < 1.0
        masks['green_launch'][i] = green and speed < 8.0 and throttle > 0.1
        masks['ped_braking'][i] = at_crosswalk and throttle < -0.05
        masks['ped_release'][i] = at_crosswalk and throttle > 0.1

        if episode_step % DEMO_NOISE_PERIOD == 0:
            noise = rng.normal(0, DEMO_NOISE_STD) if rng.random() < DEMO_NOISE_CHANCE else np.zeros(2)
        action = np.clip(label + noise, -1.0, 1.0).astype(np.float32)
        states[i], targets[i] = obs, model.policy.scale_action(label)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        info['TimeLimit.truncated'] = bool(truncated and not terminated)
        model.replay_buffer.add(obs[None], next_obs[None], model.policy.scale_action(action)[None],
                                np.array([reward]), np.array([done]), [info])
        obs = next_obs
        episode_step += 1
        if done:
            episodes += 1
            for name in totals:
                totals[name] += int(info[name])
            if env.crosswalk is not None:
                ped_episodes += 1
                ped_stops += int(info['ped_stopped'])
                ped_violations += int(info['ped_violation'])
    print(f"Demonstrations: {demos} steps, {totals['arrived']}/{episodes} arrivals, "
          f"{totals['red_violation']} red violations, {totals['waited']} red-light waits, "
          f"{totals['restarted']} green restarts", flush=True)
    if ped_episodes:
        print(f'Pedestrian demonstrations: {ped_episodes} episodes, {ped_stops} stops, '
              f'{ped_violations} crossing violations.', flush=True)
    masks['steering'] = np.abs(targets[:, 0]) > 0.04
    all_indices = np.arange(demos)
    groups = [(all_indices, RANDOM_DEMOS_PER_BATCH)]
    for name, count in DEMOS_PER_BATCH.items():
        members = np.flatnonzero(masks[name])
        groups.append((members if members.size else all_indices, count))
    return torch.from_numpy(states), torch.from_numpy(targets), groups


def collect_corrections(env, model, rng, gap_limits):
    scenario = env.default_scenario
    if env.default_movement == 'straight' or scenario not in ('mixed', 'single', 'vehicles', 'pedestrians',
                                                               'joint', 'gaps', *LEFT_SCENARIOS):
        return None
    cases = ('yield-gap', 'yield-varied', 'yield-close', 'yield-varied', *PED_SCENARIOS)
    if scenario == 'mixed':
        cases += GAP_SCENARIOS
    elif scenario == 'gaps' or scenario in MID_TRAFFIC:
        cases = (*GAP_SCENARIOS, 'joint-mid-open', 'joint-varied')
    elif scenario == 'joint' or scenario in JOINT_SCENARIOS:
        cases = (*JOINT_SCENARIOS, 'joint-varied')
    elif scenario == 'pedestrians' or scenario in PED_SCENARIOS:
        cases = PED_SCENARIOS

    states, labels, weights = [], [], []
    for i, seed in enumerate(rng.integers(*CORRECTION_SEEDS, size=CORRECTION_EPISODES)):
        obs, _ = env.reset(seed=int(seed), options={'scenario': cases[i % len(cases)], 'movement': 'left'})
        while True:
            action, _ = model.predict(obs, deterministic=True)
            near_opponent = env.encounter is not None and obs[17] > -0.3 and obs[13] < 0.65
            near_walk = env.walker is not None and -4 < env.crosswalk.distance_to_crosswalk < 12
            if near_opponent or near_walk:
                label = env.teacher_action(gap_limits)
                states.append(obs)
                labels.append(label)
                weights.append(CORRECTION_BASE_WEIGHT + abs(float(label[1] - action[1])))
            obs, _, done, truncated, _ = env.step(action)
            if done or truncated:
                break
    weights = np.asarray(weights, dtype=float)
    return torch.from_numpy(np.stack(states)), torch.from_numpy(np.stack(labels)), weights / weights.sum()


def train_agent(env, steps=ONLINE_STEPS, demos=DEMO_STEPS, seed=TRAIN_SEED, clone_updates=CLONE_UPDATES, checkpoint=None):
    if min(steps, demos, clone_updates) < 1:
        raise ValueError('Online steps, demonstration steps and cloning updates must be positive.')
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    practice_gaps = (env.observation_space.shape == (JunctionObservation.size,) and env.mixed_joint_cases == GAP_SCENARIOS
                     and env.default_movement != 'straight'
                     and env.default_scenario in ('mixed', 'joint', 'gaps', *GAP_SCENARIOS))
    gap_limits = calibrate_gap_limits(env, seed) if practice_gaps else None
    policy_kwargs = {'net_arch': list(HIDDEN_LAYERS)}
    if gap_limits is not None:
        policy_kwargs.update(features_extractor_class=GapFeatures,
                             features_extractor_kwargs={'min_gap': gap_limits[0], 'max_walk_margin': gap_limits[1]})
    model = SAC('MlpPolicy', env, **SAC_OPTIONS, buffer_size=max(MIN_REPLAY_SIZE, demos + steps + 4),
                policy_kwargs=policy_kwargs, seed=seed, device='cpu', verbose=0)
    transfer_hook = extend_actor(model.actor, checkpoint) if checkpoint is not None else None
    inputs, labels, groups = collect_demonstrations(env, model, demos, rng, gap_limits)
    corrections = None

    def imitation_loss():
        indices = np.concatenate([rng.choice(members, count) for members, count in groups])
        batch_inputs, batch_labels = inputs[indices], labels[indices]
        if corrections is not None:
            extra_inputs, extra_labels, probabilities = corrections
            extra = rng.choice(len(probabilities), CORRECTIONS_PER_BATCH, p=probabilities)
            batch_inputs = torch.cat((batch_inputs, extra_inputs[extra]))
            batch_labels = torch.cat((batch_labels, extra_labels[extra]))
        predictions = model.actor(batch_inputs, deterministic=True)
        return torch.nn.functional.mse_loss(predictions, batch_labels)

    optimizer = torch.optim.Adam(model.actor.parameters(), lr=EXTENSION_CLONE_LR if transfer_hook else CLONE_LR)

    def clone(updates):
        for _ in range(updates):
            optimizer.zero_grad()
            imitation_loss().backward()
            optimizer.step()

    clone(clone_updates)
    corrections = collect_corrections(env, model, rng, gap_limits)
    if corrections is not None:
        clone(CORRECTION_UPDATES)
        print(f'Traffic corrections: {len(corrections[0])} actor-visited states, '
              f'{CORRECTION_UPDATES} supervised updates.', flush=True)
    if transfer_hook is None:
        with torch.no_grad():
            model.actor.log_std.weight.zero_()
            model.actor.log_std.bias.fill_(INITIAL_LOG_STD)

    # Add imitation gradients to the actor's existing SAC gradient before each optimizer step.
    def add_demo_gradient(optimizer, args, kwargs):
        with torch.enable_grad():
            (DEMO_GRADIENT_WEIGHT * imitation_loss()).backward()

    model.get_env().seed(int(rng.integers(*ROLLOUT_SEEDS)))
    print(f'Cloning updates: {clone_updates}. Starting {steps} online SAC steps.', flush=True)
    hook = model.actor.optimizer.register_step_pre_hook(add_demo_gradient)
    try:
        model.learn(total_timesteps=steps)
    finally:
        hook.remove()
        if transfer_hook is not None:
            transfer_hook.remove()
    print(f'Online SAC: {model.num_timesteps} steps, {model._n_updates} gradient updates.', flush=True)
    return model


def evaluate(env, policy, first_seed, count, render=False):
    totals = dict.fromkeys(('arrived', 'compliant', 'red_violation', 'yellow_violation', 'yield_violation',
                            'ped_violation', 'ped_collision', 'waited', 'restarted', 'had_overspeed',
                            'out_of_lane', 'wrong_way', 'collision', 'timeout'), 0)
    crossings = dict.fromkeys(('g', 'y', 'r'), 0)
    routes, pedestrian_results, encounters, measured_gaps, trip_seconds = {}, {}, {}, {}, {}
    obs_size = policy.observation_space.shape[0]
    if obs_size not in (10, 14, 19, 21, 27, 35, env.observation_space.shape[0]):
        raise ValueError(f'Unexpected signal-policy observation size: {obs_size}')
    for seed in range(first_seed, first_seed + count):
        obs, _ = env.reset(seed=seed)
        elapsed_steps = 0
        while True:
            action, _ = policy.predict(obs[:obs_size], deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            elapsed_steps += 1
            if render:
                env.render(text={'Route': f'{env.approach} / {env.movement}',
                                 'Signal': {'r': 'Red', 'y': 'Yellow', 'g': 'Green'}[env.signal_color],
                                 'Speed': f'{env.vehicle.speed_km_h:.1f} km/h'})
            if done or truncated:
                break

        if info['compliant'] and not truncated:
            trip_seconds.setdefault(env.scenario, []).append(elapsed_steps * env.dt)
        for key in totals:
            totals[key] += int(truncated if key == 'timeout' else info[key])
        if info['crossing_color']:
            crossings[info['crossing_color']] += 1
        route = routes.setdefault((env.approach, env.movement), [0, 0])
        route[0] += int(info['arrived'])
        route[1] += 1
        if env.walker is not None:
            counts = pedestrian_results.setdefault(env.scenario, dict.fromkeys(
                ('episodes', 'arrived', 'compliant', 'ped_violation', 'ped_collision', 'ped_cleared_first',
                 'ped_stopped', 'ped_resumed'), 0))
            counts['episodes'] += 1
            for key in counts.keys() - {'episodes'}:
                counts[key] += int(info[key])
        if env.encounter is not None:
            key = (env.scenario, info['conflict_order'] or 'incomplete')
            encounters[key] = encounters.get(key, 0) + 1
            if info['conflict_gap_seconds'] is not None:
                measured_gaps.setdefault(key, []).append(info['conflict_gap_seconds'])

    print(f'Seeds {first_seed}..{first_seed + count - 1}, {count} episodes: {totals}', flush=True)
    print(f'Stop-line crossings by indication: {crossings}', flush=True)
    print('Arrivals by route: ' + ', '.join(f'{arm}/{move} {n}/{total}'
          for (arm, move), (n, total) in sorted(routes.items())), flush=True)
    if encounters:
        print(f'Observed conflict-zone entry order: {encounters}', flush=True)
        for (scenario, order), gaps in sorted(measured_gaps.items()):
            print(f'{scenario}, {order}: clearance in seconds, minimum {min(gaps):.3f}, '
                  f'median {np.median(gaps):.3f}', flush=True)
    if pedestrian_results:
        print(f'Pedestrian crossings: {pedestrian_results}', flush=True)
    for scenario, times in sorted(trip_seconds.items()):
        print(f'{scenario}: median trip {np.median(times):.1f} seconds over {len(times)} compliant arrivals', flush=True)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--evaluate', type=Path, metavar='MODEL', help='Evaluate a saved SAC policy.')
    mode.add_argument('--extend', type=Path, metavar='MODEL',
                      help='Extend a vehicle (21), pedestrian (27) or joint (35) policy to the next traffic curriculum.')
    parser.add_argument('--render', action='store_true', help='Show the junction during evaluation.')
    parser.add_argument('--episodes', type=int, default=EVAL_EPISODES, help='Number of evaluation episodes.')
    parser.add_argument('--first-seed', type=int, default=EVAL_FIRST_SEED, help='First evaluation scenario seed.')
    parser.add_argument('--approach', choices=('all', *APPROACHES), help='Select the junction entrance.')
    parser.add_argument('--movement', choices=('mixed', 'straight', 'left'), help='Select the planned movement.')
    parser.add_argument('--scenario', choices=SCENARIOS, help='Select the signal and approach conditions.')
    args = parser.parse_args()
    if args.episodes < 1 or args.first_seed < 0 or args.first_seed + args.episodes > NUM_SCENARIOS:
        parser.error(f'Choose a positive episode count and seeds in [0, {NUM_SCENARIOS}).')
    if args.render and args.evaluate is None:
        parser.error('--render requires --evaluate MODEL.')
    left_only = ('pedestrians', 'joint', 'gaps', *LEFT_SCENARIOS)
    if args.movement == 'straight' and (args.extend or args.scenario in left_only):
        parser.error('This scenario requires --movement left or mixed.')

    torch.set_num_threads(1)
    model = SAC.load(args.evaluate, device='cpu') if args.evaluate is not None else None
    base = SAC.load(args.extend, device='cpu') if args.extend is not None else None
    model_size = model.observation_space.shape[0] if model is not None else None
    base_size = base.observation_space.shape[0] if base is not None else None
    if base is not None and base.observation_space.shape not in ((21,), (27,), (35,)):
        parser.error('--extend requires a 21-, 27- or 35-input signal policy.')
    if base is not None and base.actor.latent_pi[0].in_features > JunctionObservation.size:
        parser.error('This checkpoint already includes calibrated gap selection.')

    eval_suites = {10: 'standard', 14: 'signals', 19: 'vehicles', 21: 'vehicles', 27: 'single'}
    scenario = args.scenario or ('single' if base_size == 21 else eval_suites.get(model_size, 'mixed'))
    movement = args.movement or ('straight' if model_size == 10 and scenario not in left_only else 'mixed')
    approach = args.approach or ('west' if model_size == 10 else 'all')
    observation = PedestrianObservation if base_size == 21 else JunctionObservation
    training_suite = {21: 'pedestrians', 27: 'joint', 35: 'gaps'}.get(base_size)
    gap_curriculum = (model is None or model.actor.latent_pi[0].in_features > JunctionObservation.size)
    if base is not None:
        gap_curriculum = training_suite == 'gaps'
    env = SignalEnv(render=args.render, approach=None if approach == 'all' else approach, movement=movement,
                    scenario=scenario, observation=observation, gap_curriculum=gap_curriculum)
    try:
        if model is None:
            if base is None:
                model = train_agent(env)
            else:
                demos, clone_updates = EXTENSION_BUDGETS[base_size]
                env.default_scenario = training_suite
                try:
                    model = train_agent(env, steps=EXTENSION_STEPS, demos=demos, clone_updates=clone_updates,
                                        checkpoint=base)
                finally:
                    env.default_scenario = scenario
            output = PROJECT_DIR / 'demo_outputs' / 'sac_signals' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            output.mkdir(parents=True)
            checkpoint = output / 'sac_model.zip'
            model.save(checkpoint)
            print(f'Saved model: {checkpoint}', flush=True)
        evaluate(env, model, args.first_seed, args.episodes, render=args.render)
    finally:
        env.close()


if __name__ == '__main__':
    main()
