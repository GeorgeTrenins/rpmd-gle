#!/bin/bash -e

#        0  1  2  3  4   5   6   7   8   9  10  11  12  13
Temps=( 50 60 70 80 90 100 125 150 160 175 200 225 250 300 )
# bead specifications:
ones=( 1 1 1 1 1 1 1 1 1 1 1 1 1 1 )  
HBhi=( 96 64 64 48 48 48 32 32 24 24 24 24 16 16 )    
HBlo=( 64 48 48 32 32 32 24 24 16 16 16 16 12 12 )    
DBhi=( 64 48 48 32 32 32 24 24 16 16 16 16 12 12 )    
DBlo=( 48 32 32 24 24 24 16 16 12 12 12 12  8  8 )    
# zero-frequency value of Lambda for H1 in [1/ps]
L0=0.15   # <- check! this value depends on the broadening (here, s=0.02)
# number of metal layers
l=6
# number of k-points
k=16
# broadening of spectral density, in eV
s=0.02
# ignore spatial variation of friction
linear="--linear"
# propagation timestep
dt="0.50 fs"
# PILE thermostatting constant
tau="10 fs"

model="pbe_light_frozen"
bias=0.010

make_stuff() {
    for idx in "${indices[@]}"  ; do
        T=${Temps[$idx]}
        echo "T = $T K"
        TT=$(printf "%03d" ${T})
        N=${nbeads[$idx]}
        echo "N = $N"
        NN=$(printf "%03d" ${N})
        for eta in "${etas[@]}"; do
            echo "η = ${eta}"
            h=$(printf "%d" ${eta})
            scaledL0=$( bc -l <<< "scale=4; ${L0}*${h}" )
            dir="${root}/${TT}K/nbeads_${NN}/eta_${eta}.00/kTST"
            mkdir -p ${dir}
            cp ktst.rpmd.template run.sh
            sed -i "s%#SBATCH -J .*%#SBATCH -J ${TT}-${eta}-TST%g" run.sh
            sed -i "s%T=.*%T=${T}%g" run.sh
            sed -i "s%N=.*%N=${N}%g" run.sh
            python make_input.py ${model} ${bias}  -l $l -k $k -s $s \
                --wc ${wc} --naux 0 -T ${T} -N $N \
                --dt="${dt}" --tau="${tau}" --L0 ${scaledL0} ${deuterate}
            mkdir -p "${dir}"
            mv run.sh "${dir}"
            mv *.json "${dir}"
        done
    done
}

for wc in 2000 4000 ; do
    # protium
    deuterate=""
    # use lower number of beads
    nbeads=("${HBlo[@]}")
    root="${model}/H/wcut_${wc}"
    # scaling factors
    etas=("01" "02" "05" "10") 
    # temperature indices
    indices=(0 5 7)
    make_stuff
    # deuterium
    deuterate="--D"
    # use lower number of beads
    nbeads=("${DBlo[@]}")
    root="${model}/D/wcut_${wc}"
    # scaling factors
    etas=("01" "02" "05" "10") 
    # temperature indices
    indices=(0 5 7)
    make_stuff
done
