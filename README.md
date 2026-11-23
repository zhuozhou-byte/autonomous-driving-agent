# autonomous-driving-agent

Autonomous driving experiments with MetaDrive and SUMO.

An ELG 5255 project exploring driving decisions in simulation. The current code
contains rule-based driving demos, evaluation of MetaDrive's pretrained PPO
expert, and a small SUMO-to-MetaDrive map import. Model training is a later step.

## Start with the SUMO map demo

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

The current setup uses Windows, Python 3.11, SUMO 1.28.0, and MetaDrive 0.4.3.
Install [Eclipse SUMO](https://sumo.dlr.de/docs/Installing/index.html) and ensure
`SUMO_HOME` points to its installation directory or `netconvert` is on PATH.
From this project's root, create a Python environment and install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install "git+https://github.com/metadriverse/metadrive.git@325d9e2f9c20e495bde66f30e0ef2b40efbce39c" "sumolib==1.28.0" "imageio-ffmpeg==0.6.0"
.venv\Scripts\python.exe sumo_import_demo.py --headless
```

This pins MetaDrive to the source revision used by the project. Git is required
for the install command. In PyCharm, select `.venv\Scripts\python.exe` as the
project interpreter. MetaDrive may download its assets on the first run.

## Choose the script to run

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
All modes use the current library's default reward without custom weights.

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

- `metadrive/`: local dependency source used by the original environment; excluded from Git. Other computers install the pinned dependency above.
- `.idea/`: PyCharm settings.
- `.venv/`: interpreter environment currently referenced by PyCharm settings.
- `__pycache__/`: automatically generated Python cache.
- `demo_outputs/`: results, backups, and archived old helper scripts.

Old helper scripts and the detailed historical reward notes are archived under
`demo_outputs/backups/cleanup_<timestamp>/`. They are not used by the demo.



## MP4 export
Run ppo_baseline.py with --mp4 --view none --episodes 1. MP4 export records the 3D follow camera offscreen, without screen recording. Clips default to at most 40 seconds at simulated real-time speed; shorter episodes remain shorter. Use --video-seconds to change the limit. H.264 encoding requires imageio-ffmpeg, installed in the MetaDrive environment. Videos are saved in the timestamped run directory. Recording stops at the clip limit while episode evaluation continues.
