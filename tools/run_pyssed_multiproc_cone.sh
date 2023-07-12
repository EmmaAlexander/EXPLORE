#!/bin/bash

echo ">>>start"

FOLDER="Users/user/Projects/EXPLORE/pyssed/"
echo $PATH
##change second number to be the number of splits
for i in {1..10}
do
    runList+="run"$i" "
done
pwd; 
conda acivate explore
which python
echo ">>>running"
echo $FOLDER"v0.3-development/cone/" ##mkdir
foo () {
    local run=$1
    cmd=$(python3 random_coords.py)
    echo $cmd
    sleep 1
    echo $run
    cp -r $FOLDER"v0.3-development" $FOLDER"v0.3-development-"$run
    cd $FOLDER"v0.3-development-"$run/src/
    python $FOLDER-$run/src/pyssed.py cone $cmd simple
    cp $FOLDER-$run/output/default_output.dat $FOLDER/"cone/output_run-"$i.dat
}
for run in $runList; do foo "$run" & done
