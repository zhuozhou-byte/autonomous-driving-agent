"""Evaluate MetaDrive's original pretrained PPO expert with library-default rewards.

Run this file in PyCharm. No training, fine-tuning, custom rewards, or IDM control.
The shared demo runner provides logging and tables; the driver is fixed to ExpertPolicy.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', choices=['3d', 'topdown', 'none'], default='none')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1, help='first map seed')
    parser.add_argument('--map', default='SCS', help='road sequence or number of road blocks')
    parser.add_argument('--traffic', type=float, default=0.1)
    parser.add_argument('--horizon', type=int, default=1500)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'demo_outputs' / 'ppo_baseline')
    parser.add_argument('--video-seconds', type=float, default=40.0, help='maximum MP4 length in seconds')
    parser.add_argument('--mp4', action=argparse.BooleanOptionalAction, default=True, help='export 3D follow-camera MP4 without screen recording')
    args = parser.parse_args()

    # Use a subprocess so the existing interpreter fallback cannot lose the PPO selection.
    runner = Path(__file__).resolve().with_name('metadrive_demo.py')
    if not runner.exists():
        parser.error('Keep ppo_baseline.py and metadrive_demo.py in the same folder.')
    python = Path(os.environ.get('METADRIVE_PYTHON', r'D:\conda-envs\metadrive\python.exe'))
    interpreter = str(python) if python.exists() else sys.executable
    command = [interpreter, str(runner), '--policy', 'expert',
               '--view', args.view, '--episodes', str(args.episodes),
               '--seed', str(args.seed), '--map', args.map,
               '--traffic', str(args.traffic), '--horizon', str(args.horizon),
               '--output-dir', str(args.output_dir)]
    if args.mp4:
        command.extend(['--mp4', '--video-seconds', str(args.video_seconds)])
    else:
        command.append('--no-mp4')
    print('BASELINE: original MetaDrive ExpertPolicy (pretrained PPO, sampled actions).', flush=True)
    print('Library-default evaluation reward. Model weights are not updated.', flush=True)
    try:
        return subprocess.run(command, check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
