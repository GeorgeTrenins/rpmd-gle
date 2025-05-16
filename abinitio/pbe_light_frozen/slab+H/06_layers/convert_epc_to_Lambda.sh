#!/bin/bash -e

# Directory containing *.csc files
if [ -z "$1" ] ; then
    echo "Please specify the directory containing the EPC data in *.csc format"
    exit 1
fi
TARGET="$(realpath $1)"
cd $TARGET

# Set up the environment
source /opt/intel/oneapi/setvars.sh &> /dev/null 

# Compute the friction spectra
Hidx=81
CWD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sigma=0.01 # eV
k=16       # number of k-points
T=300      # electronic temperature in Kelvin
TT=$(printf "%03d" ${T})
# Cycle over xyz
for i in {1..3} ; do
    for j in {1..3} ; do
        if [ $j -lt $i ]; then
            continue
        fi
        mpirun -n 8 python -m eftools.friction . -v 2 -T ${T} --a1 $Hidx --c1 $i --a2 $Hidx --c2 $j \
            --raw-lambda --max-freq 2.0 --freq-step 0.0025 --broadening ${sigma} --mode 2
        mv Lambda_atom_0000${Hidx}_cart_${i}_0000${Hidx}_cart_${j}_temp_${T}.00K.csv \
            sigma_${sigma}_Lambda_atom_0000${Hidx}_cart_${i}_0000${Hidx}_cart_${j}_temp_${TT}.00K.csv
    done
done
