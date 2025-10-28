#!/bin/bash

# Parameters for sbatch
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=1

# Conda setup
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=dpvo

# Get name argument
NAME="$1"

# Submit the SLURM job
sbatch \
    -N $NUM_NODES \
    -c $NUM_CORES \
    --gres=gpu:$NUM_GPUS \
    --job-name=$NAME \
    -o "slurm-%N-%j.out" \
    --export=ALL,NAME="$NAME",CONDA_HOME="$CONDA_HOME",CONDA_ENV="$CONDA_ENV" \
<<'EOF'
#!/bin/bash

# Activate conda environment
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

echo "Running with name: $NAME"

echo "ATTACKING"
python3 PGD_attack_final.py --name "$NAME" --vid --stride=5
EOF
