#!/bin/bash

#SBATCH --job-name=pyssed90
#SBATCH --constraint=A100
#SBATCH --time=10-23
#SBATCH --ntasks-per-node=10
#SBATCH --nodes=10
#SBATCH --output=/share/nas2/ela/EXPLORE/PySSED/logs/pyssed-90-slurm_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=emma.alexander@mmanchester.ac.uk

nvidia-smi 
echo ">>>start"

LIST="/share/nas2/ela/EXPLORE/PySSED/v0.3-development/90percent/90percent_sample.list"

#change second number to be the number of splits
for i in {1..100}
do
	runList+="run"$i" "
	listList+="list"$i" "
done
pwd;
source /share/nas2/ela/EXPLORE/PySSED/venv/bin/activate
which python
echo ">>>running"
##python /share/nas2/ela/EXPLORE/PySSED/v0.3-development/src/pyssed.py list $LIST simple
python /share/nas2/ela/EXPLORE/PySSED/v0.3-development/src/split_list.py $LIST $i

foo () {
    local run=$1
    echo $run
    echo "hello"
    sleep 2
    echo $run
    cp -r /share/nas2/ela/EXPLORE/PySSED/v0.3-development-empty /share/nas2/ela/EXPLORE/PySSED/v0.3-development-$run
    cd /share/nas2/ela/EXPLORE/PySSED/v0.3-development-$run/src/
    mv /share/nas2/ela/EXPLORE/PySSED/$run".list" /share/nas2/ela/EXPLORE/PySSED/v0.3-development-$run/src/$run".list"
    python /share/nas2/ela/EXPLORE/PySSED/v0.3-development-$run/src/pyssed.py list /share/nas2/ela/EXPLORE/PySSED/v0.3-development-$run/src/$run".list" simple
}
for run in $runList; do foo "$run" & done
