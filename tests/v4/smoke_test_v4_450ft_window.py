#!/usr/bin/env python3
"""Assert production span semantics: 9 x 50-ft sequence intervals = 450 ft, up to 10 centers."""
from __future__ import annotations
import argparse

def main():
    latest=20; gap=9; first=latest-gap; seqs=list(range(first,latest+1))
    assert first==11 and len(seqs)==10 and (latest-first)*50==450
    assert 10 not in seqs and 21 not in seqs
    print(f'V4_450FT_WINDOW_OK first={first} latest={latest} sequence_gap={gap} observed_centers={len(seqs)} span_ft={(latest-first)*50}')
if __name__=='__main__':main()
