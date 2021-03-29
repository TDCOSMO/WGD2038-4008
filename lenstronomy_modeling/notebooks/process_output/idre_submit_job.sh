#### submit_job.sh START #####
#!/bin/bash

name="vd"
slots=1
mem=999 # this will give you mem Megabyte per proc
time=6 # this will give you 24 hour runtime
hp="" #",highp"

function usage {
    echo -e "\nUsage:\n $0 <config_string>"
}

if [ $# == 0 ]; then
    echo -e "\n Please provide a config_string"
    usage
    exit
fi

config_string="$1"
run_type="$2"
start_index="$3"
compute_chunk="$4"

cat << EOF > ./${name}_${config_string}_${start_index}.cmd
#!/bin/bash
#  UGE job for run_sequence.py built Thu Feb 16 09:35:24 PST 2017
#
#  The following items pertain to this script
#  Use current working directory
#$ -cwd
#  input           = /dev/null
#  output          = /u/home/a/ajshajib/Logs/joblog
#$ -o /u/home/a/ajshajib/Logs/joblog.\$JOB_ID
#  error           = Merged with joblog
#$ -j y
#  The following items pertain to the user program
#  user program    = /u/home/a/ajshajib/Scripts/run_sequence.py
#  arguments       = mcmc_test
#  program input   = Specified by user program
#  program output  = Specified by user program
#  Parallelism:  $slots-way parallel
#  Resources requested
#$ -pe shared $slots
#$ -l h_data=${mem}M,h_rt=${time}:00:00$hp
#
#$ -M $USER@mail
#  Notify at beginning and end of job
#$ -m a
#  Job is not rerunable
#$ -r n
#  Uncomment the next line to have your environment variables used by SGE
# -V
#
#
echo ""
echo "Job (run_sequence.py) \$JOB_ID started on:   "\` hostname -s \`
echo "Job (run_sequence.py) \$JOB_ID started on:   "\` date \`
echo ""
#
# Run the user program
#
. /u/local/Modules/default/init/modules.sh
module load intel/14.cs
#module load intelmpi/5.0.0
export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=4

#module load python/2.7.16
module load python/3.7.2
#module load anaconda3/2019.03
#module load python/anaconda3

#export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
#export LD_LIBRARY_PATH=/u/local/apps/anaconda3/lib:${LD_LIBRARY_PATH}

module load gcc/7.2.0
#module load intelmpi/5.0.0
#export PYTHONPATH=$HOME/python_packages/lib/python2.7/site-packages:$HOME/python_packages:$PYTHONPATH
#export LD_LIBRARY_PATH=/u/home/a/ajshajib/mybin/MultiNest/lib:$LD_LIBRARY_PATH

module list
which mpirun
which python3
echo $LD_LIBRARY_PATH

echo "\`which mpirun\` -np ${slots} \`which python3\`  \\
         /u/home/a/ajshajib/2038_process_output/process_output.py $config_string $run_type $start_index $compute_chunk >& /u/home/a/ajshajib/Logs/output.\$JOB_ID"

time \`which mpirun\` -np ${slots} \`which python3\`  \\
         /u/home/a/ajshajib/2038_process_output/process_output.py $config_string $run_type $start_index $compute_chunk >& /u/home/a/ajshajib/Logs/output.\$JOB_ID


echo ""
echo "job (run_sequence.py) \$JOB_ID  finished at:  "` date `
echo ""

EOF

chmod u+x ${name}_${config_string}_${start_index}.cmd

if [[ -x ${name}_${config_string}_${start_index}.cmd ]]; then
    echo "qsub ${name}_${config_string}_${start_index}.cmd"
    qsub ${name}_${config_string}_${start_index}.cmd
fi
#### submit_job.sh END #####
