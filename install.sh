uv sync --active
uv pip install flash-attn --no-build-isolation
uv pip install -e .

# Pythia connections
gdown --folder 1h5W_rX1DpVE937BIEkO3pcJOI5RI2tea

# monology/pile-uncopyrighted
# Make sure hf-transfer is installed
# huggingface-cli download monology/pile-uncopyrighted \
#     train/00.jsonl.zst \
#     --repo-type dataset \
#     --local-dir /root/dictionary_learning/pile-uncopyrighted