#!/bin/bash

design=N3-index0-L20-phiTa0.9-tol1e-8-mineig1.0.p.gz
pair_file=pair-15.dat

python3 gen_design.py fullrank $design 17.321 20 phase-15.json $pair_file