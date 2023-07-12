#!/bin/bash

echo ">>>start"

FOLDER="/Users/user/Projects/EXPLORE/pyssed/v0.3-development"
OUTFOLDER="/Users/user/Projects/EXPLORE/mainSequenceStars"
echo $PATH

split_and_run() {
    local run=$1
    cmd=$(python3 /Users/user/Projects/EXPLORE/mainSequenceStars/random_coords.py)
    echo $cmd
    sleep 1 ## just give it a breather :)
    echo $run
    cp -r $FOLDER $FOLDER-$run
    cd $FOLDER-$run/src/
    python $FOLDER-$run/src/pyssed.py cone $cmd simple
    cp $FOLDER-$run/output/output.dat "/Users/user/Projects/EXPLORE/mainSequenceStars/cone_output/output_"$run.dat
    cp -r $OUTFOLDER/v0.3-development $OUTFOLDER/runs/fld-$run
    cd $OUTFOLDER/runs/fld-$run/src/
    python3 $OUTFOLDER/runs/fld-$run/src/pyssed.py cone $cmd simple
    cp $OUTFOLDER/runs/fld-$run/output/output.dat "/Users/user/Projects/EXPLORE/mainSequenceStars/cone_output/output_"$run.dat
    cp $OUTFOLDER/runs/fld-$run/output/hrd.png "/Users/user/Projects/EXPLORE/mainSequenceStars/cone_output/hrd_"$run.png
}


#this will start at 1*100 = run100 and will end at (1+99)*100=run10000
splitsize=100
nsplits=99
nstart=1
for (( n=${nstart}; n<${nsplits}; n++ )); do
    min=$(echo "$n * $splitsize" | bc)
    max=$(echo "($n+1) * $splitsize" | bc)
    echo $min $max
    for (( i=${min}; i<${max}; i++ )); do split_and_run "run"$i" " & done
    wait ## this means it doesn't move onto the next batch until everything in the loop above has finished. 
done


