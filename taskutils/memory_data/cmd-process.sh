
# Step 1 Download the benchmarks from internet
# hope clone https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets
PROJ_ROOT=$(pwd)

# Step 2
cd $PROJ_ROOT/taskutils/memory_data
source /usr/local/conda/etc/profile.d/conda.sh
conda activate base

export PATH="/usr/local/conda/bin:$PATH:~/.local/bin"

DATAROOT="$PROJ_ROOT/data/test"
mkdir -p $DATAROOT
export TOKENIZERS_PARALLELISM=false
data_sources="hotpotqa,2wikimultihopqa,nq,triviaqa"
n_subset=128
python3 process_test.py --local_dir $DATAROOT --data_sources $data_sources --n_subset $n_subset