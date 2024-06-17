#!/bin/bash

design=N3-index0-L25-phiTa0.9-tol1e-8-mineig1.0.p.gz
pair_file=pair-25.dat

python3 gen_design.py fullrank $design 22.361 20 phase-25.json $pair_file
