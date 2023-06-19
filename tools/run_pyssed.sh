#!/bin/bash

#SBATCH --job-name=pyssed_test
#SBATCH --constraint=A100
#SBATCH --time=10-23
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/share/nas2/ela/EXPLORE/PySSED/logs/run_pyssed-out-slurm_%j.out


pwd;

nvidia-smi 
echo ">>>start"
source /share/nas2/ela/EXPLORE/PySSED/venv/bin/activate
which python
echo ">>>running"
python /share/nas2/ela/EXPLORE/PySSED/v0.3-development/src/pyssed.py list "test.list" simple
