#!/bin/bash -l
#SBATCH -J l6-k16
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=72
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=george.trenins@mpsd.mpg.de
##SBATCH --mem=120000
# Initial working directory:
#SBATCH -D .

# Load compiler and MPI modules (must be the same as used for compiling the code)
module purge
module load anaconda/3/2021.11 intel/21.7.1 impi/2021.7 mkl/2023.1 cuda/12.1 

for p in "${MKL_HOME}/lib/intel64" \
         "${INTEL_HOME}/compiler/latest/linux/compiler/lib/intel64_lin" ; do
    export LD_LIBRARY_PATH="${p}:${LD_LIBRARY_PATH}"
done

VERSION=240717
#export AIMS_HOME=/u/getren/software/FHIaims/build_$VERSION
export AIMS_HOME=/u/getren/software/FHIaims/build_fixfric
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
ulimit -s unlimited

CTRL=control.in
cp $CTRL.template $CTRL
cat ../../../../../basissets/01_H_default >> $CTRL
cat ../../../../../basissets/29_Cu_default >> $CTRL
GEOM=geometry.in
cp ../../geopt/$GEOM.next_step $GEOM
sed -i '/atom_frac .* H/a\     calculate_friction .true.' $GEOM

srun $AIMS_HOME/aims.$VERSION.scalapack.mpi.x < /dev/null > aims.out
