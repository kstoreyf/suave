#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=avx0.3
#SBATCH --output=logs/avx0.3_%j.out

cd /scratch/ksf293
overlay_ext3=/scratch/ksf293/overlay-50G-10M.ext3
singularity \
exec --overlay $overlay_ext3:ro \
/scratch/work/public/singularity/centos-7.8.2003.sif /bin/bash \
-c "source /ext3/env.sh; \
/bin/bash; \
cd /home/ksf293/suave/tests/scripts; \
conda activate suaveenv; \
#python accuracy_test.py
python generate_numpart_scaling.py
"
