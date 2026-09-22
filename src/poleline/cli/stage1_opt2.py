"""Run the accepted E7 Stage 1 entry point with opt-in H100 optimizations."""
import runpy
import sys
from poleline._compat import ROOT


def main():
    directory = ROOT / 'experiments/v4_stage1_inference_opt2'
    sys.path[:0] = [str(ROOT / 'v4'), str(directory)]
    runpy.run_path(str(directory / 'run_v4_stage1_opt2.py'), run_name='__main__')


if __name__ == '__main__':
    main()
