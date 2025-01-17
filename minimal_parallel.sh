export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3,6,7
export HF_HOME=/ceph/jbrinkma/.cache/transformers
export HF_DATASETS_CACHE=/ceph/jbrinkma/.cache/datasets

accelerate launch minimal_parallel.py
