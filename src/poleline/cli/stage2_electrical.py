"""Run electrical Stage 2 from durable Stage 1 prediction artifacts."""
import runpy
import sys
from poleline._compat import ROOT


def main():
    directory = ROOT / 'experiments/v4_stage2_stage1_electrical_v10_opt'
    sys.path[:0] = [str(ROOT / 'v4'), str(directory)]
    runpy.run_path(str(directory / 'run_v4_stage2_stage1_electrical_tracks_opt.py'), run_name='__main__')


if __name__ == '__main__':
    main()
