# autonomous-driving-agent

Autonomous driving experiments with MetaDrive and SUMO.

An ELG 5255 project exploring driving decisions in simulation. The main experiments
train our own PPO policies with Stable-Baselines3. The repository also includes
rule-based baselines, MetaDrive's pretrained PPO expert, and a small
SUMO-to-MetaDrive map import.

## Core experiments

| Entry point | Experiment | Default action |
| --- | --- | --- |
| `train_intersection_ppo.py` | PPO at a signal-controlled intersection | Evaluate `models/ppo_yellow.zip` |
| `train_ppo.py` | PPO on procedurally generated curved roads | Evaluate `models/ppo_curve.zip` |

The two included checkpoints are our trained policies, separate from MetaDrive's
pretrained expert. Both PPO entry points run without a window or video recording
by default and print their results in the terminal. In PyCharm, select your Python
environment and run the chosen file.

```powershell
python train_intersection_ppo.py
python train_ppo.py
```

The main intersection experiment uses a fixed native `X` map with no background
traffic. Scenario seeds vary starting position, initial speed, and signal phase.
The cycle is green 8 seconds, yellow 3 seconds, and red 8 seconds. Yellow entry is
allowed; crossing the stop line during red ends the episode. PPO learns both
steering and throttle/braking from a 14-value observation, including the signal.

The reward includes route progress, arrival, red-light violations, collisions,
road departure, time, approach speed, and braking risk. The target in the last
20 metres before the stop line is 20 km/h; this is a reward target, not a speed
limiter. Stopping to wait near the line during red/yellow is exempt from time cost.

### Selected intersection checkpoint

Evaluation on scenario seeds 10000-10049, with deterministic actions:

| Metric | Result |
| --- | --- |
| Successful arrival | 49/50 (98%) |
| Red-light violations | 1/50 |
| Collisions / departures / timeouts | 0 / 0 / 0 |
| Mean successful travel time | 14.6 s |
| Mean successful entry speed | 38.8 km/h |
| Moving approach time above 22 km/h, averaged across episodes | 81.7% |

The checkpoint achieved 194/200 arrivals on validation seeds 2000-2199.
It still approaches too fast: the 20 km/h target has not been learned reliably.
Increasing the approach-speed penalty from 2 to 8 reduced speed in some candidates
but lowered arrival success. The selected checkpoint and defaults retain weight 2.

Intersection training scenarios use seeds 0-999. The current curve PPO script is
configured to train on map seeds 2000-2999, validate on 100-299, and test on
1000-1099, without traffic.
Its default continued-training mode freezes the learned steering network and trains
the speed network.

`models/ppo_curve.zip` is an earlier checkpoint. Its metadata records a 0-99
training-map pool and curvature-aware speed rewards; the current runner uses
MetaDrive's default rewards. It is included as a reference policy and starting
point for further training. A publication smoke check found a road departure on
seed 1000; it is not a perfect driving policy.

### Training settings

All editable settings are at the top of each script.

- Intersection PPO, from scratch: set `RUN_MODE='train'`,
  `RESUME_TRAINING=False`, and choose a new `MODEL_PATH` under `demo_outputs/`.
  The curriculum runs green-only, red-start, then full signal cycles.
- Curve PPO, from scratch: set `RUN_MODE='train'`, `RESUME_MODEL=None`, and
  `TRAIN_SPEED_ONLY=False`.
- Continued intersection PPO training: set `RUN_MODE='train'`, keep
  `RESUME_TRAINING=True`, set `RESUME_MODEL=PROJECT_DIR / 'models' / 'ppo_yellow.zip'`,
  and set `MODEL_PATH=PROJECT_DIR / 'demo_outputs' / 'intersection_ppo' / 'continued.zip'`.
- Continued curve PPO training: set `RUN_MODE='train'` and retain `RESUME_MODEL`.
  New checkpoints are written to a timestamped directory under `demo_outputs/`.

## Supporting SUMO map demo

Run `python sumo_import_demo.py` to open the 3D scene, or
`python sumo_import_demo.py --headless` to run without a window.

`sumo_import_demo.py` loads `sumo_small.net.xml` with `SumoMapManager`, connects
the entry lane, internal junction lane, and exit lane into a route, and controls
one car with MetaDrive's rule-based `TrajectoryIDMPolicy`. The car enters from
the west, turns left, and exits north. The terminal reports arrival, collision,
and simulation time. Press B for an overhead view, Q for the follow camera, or
Esc to exit.

The script generates the small map with SUMO's `netconvert` if the XML file is
missing. It currently uses placeholder observations and a zero reward to focus
on map integration. The existing PPO runner uses MetaDrive's native maps.

## Install on another computer

The tested training environment uses Windows, Python 3.11, MetaDrive 0.4.3,
Stable-Baselines3 2.9.2, PyTorch 2.15.0, Gymnasium 1.4.0, and NumPy 2.4.6.
SUMO is only required for `sumo_import_demo.py`: install
[Eclipse SUMO](https://sumo.dlr.de/docs/Installing/index.html) and ensure
`SUMO_HOME` points to its installation directory or `netconvert` is on PATH.
From this project's root, create a Python environment and install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install "git+https://github.com/metadriverse/metadrive.git@325d9e2f9c20e495bde66f30e0ef2b40efbce39c" "stable-baselines3==2.9.2" "torch==2.15.0" "gymnasium==1.4.0" "numpy==2.4.6"
.venv\Scripts\python.exe train_intersection_ppo.py
```

This pins MetaDrive to the source revision used by the project. Git is required
for the install command. In PyCharm, select `.venv\Scripts\python.exe` as the
project interpreter. MetaDrive may download its assets on the first run.
For the supporting SUMO demo and MP4 exporters, also install `sumolib==1.28.0`
and `imageio-ffmpeg==0.6.0`.

## Supporting baseline demos

For the pretrained PPO baseline, open `ppo_baseline.py` in PyCharm and click Run. It fixes the driver to the original library ExpertPolicy and uses the default environment reward. It reuses the reporting functions through `metadrive_demo.py`, so keep both files.

For the adaptive rule-based demo, run `metadrive_demo.py`.
It handles driving, built-in reward recording, plots, and table exports.
No separate reward script is required.

On the original development machine, the interpreter is `D:\conda-envs\metadrive\python.exe`.
The script can also relaunch itself with this interpreter when MetaDrive is unavailable.

## Optional run arguments

- No arguments: adaptive rule-based demo, 3 episodes, offscreen 3D MP4 export.
- `--policy expert-mean`: pretrained PPO with deterministic mean actions.
- `--policy expert`: original sampled-action PPO expert.
- `--policy idm`: library IDM policy.
- `--view none --no-mp4 --episodes 3`: run without a display or video.
- `--view 3d --no-mp4`: open a 3D window without recording.

The default adaptive driver is a rule-based demo, not a trained model.
These baseline demo modes use the current library's default reward without custom weights.

## Outputs

`metadrive_demo.py` creates `demo_outputs/<timestamp>/`; `ppo_baseline.py` creates `demo_outputs/ppo_baseline/<timestamp>/`. Each run contains:

- The terminal prints a MetaDrive Default Reward table with Component and Value columns at the end of the run.
- `tables/results_summary.png`: results across completed episodes.
- `tables/episode_results_1.png`: per-episode results (additional pages when needed).
- CSV copies of the tables for editing.
- `rewards/`: per-step CSV logs and plots.
- `run_config.json` and `summary.json`: configuration and recorded results.

The tables identify the controller actually used. Current evaluation rewards are not
verified original training settings of the pretrained expert. A few demo runs do not
establish general performance.

## Other folders

- `models/`: the two selected PPO checkpoints included with this repository.
- `metadrive/`: local dependency source used by the original environment; excluded from Git. Other computers install the pinned dependency above.
- `.idea/`: PyCharm settings.
- `.venv/`: interpreter environment currently referenced by PyCharm settings.
- `__pycache__/`: automatically generated Python cache.
- `demo_outputs/`: results, backups, and archived old helper scripts.

Old helper scripts and the detailed historical reward notes are archived under
`demo_outputs/backups/cleanup_<timestamp>/`. They are not used by the demo.



## MP4 export
Run ppo_baseline.py with --mp4 --view none --episodes 1. MP4 export records the 3D follow camera offscreen, without screen recording. Clips default to at most 40 seconds at simulated real-time speed; shorter episodes remain shorter. Use --video-seconds to change the limit. H.264 encoding requires imageio-ffmpeg, installed in the MetaDrive environment. Videos are saved in the timestamped run directory. Recording stops at the clip limit while episode evaluation continues.
