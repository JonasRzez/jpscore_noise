#!/bin/bash -x
#SBATCH -J db
#SBATCH --account=jias70
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --output=logfile/mpi-out.%j
#SBATCH --error=logfile/mpi-err.%j
#SBATCH --time=00:05:00
#SBATCH --mail-user=j.rzezonka@fz-juelich.de
#SBATCH --mail-type=ALL

module load CMake
module load GCC
module load ParaStationMPI
module load Boost
module load GCCcore/.8.3.0


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun ./jps_jureca/build/bin/jpscore ./jps_jureca/demos/scenario_2_bottleneck/bottleneck_ini.xml


