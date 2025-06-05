#!/bin/bash -l
# Batch script to run a multi-threaded MATLAB job under SGE.
# Request wallclock time (format hours:minutes:seconds).
#$ -l h_rt=48:0:0
 

# Request GPU(s)     
#$ -l gpu=1


# Request 1 gigabyte of RAM per core. 
#$ -l mem=4G
# Request 15 gigabytes of TMPDIR space (default is 10 GB)
#$ -l tmpfs=25G
# Request a number of threads (which will use that number of cores). 
# On Myriad you can set the number of threads to a maximum of 36. 
#$ -pe smp 16
# Request one MATLAB licence - makes sure your job doesn't start 
# running until sufficient licenses are free.
#$ -l matlab=1



# Set the name of the job.
#$ -N NEW50

# Set the working directory to somewhere in your scratch space.
# This is a necessary step as compute nodes cannot write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID.
# This directory must already exist.
#$ -wd /home/zcemydo/Scratch/TrainV4

# Your work should be done in $TMPDIR
cd $TMPDIR

module load xorg-utils/X11R7.7
module load matlab/full/r2023a/9.14
# outputs the modules you have loaded
module list

# Optional: copy your script and any other files into $TMPDIR.
# This is only practical if you have a small number of files.
# If you do not copy them in, you must always refer to them using a
# full path so they can be found, eg. ~/Scratch/Matlab_examples/analyse.m

cp ~/Scratch/TrainV5/TRAIN.m $TMPDIR
cp ~/Scratch/TrainV5/cocoAnnotationMATReader.m $TMPDIR
cp -r ~/Scratch/TrainV5/src $TMPDIR



# These echoes output what you are about to run
echo ""
echo "Running MATLAB..."
echo ""

#run matlab script(s)
matlab -nosplash -nodesktop -nodisplay < TRAIN.m


# tar up all contents of $TMPDIR back into your space
tar zcvf $HOME/Scratch/TrainV4/files_from_job_${JOB_ID}.tgz $TMPDIR

# Make sure you have given enough time for the copy to complete!
