source .venv/bin/activate

pip install uv
pip install huggingface-hub # Jacob: added because the pyproject.toml install wasn't working

uv sync --active
uv pip install wheel # Added to ensure wheel is available for flash-attn
uv pip install flash-attn --no-build-isolation
uv pip install -e .

# monology/pile-uncopyrighted
# Make sure hf-transfer is installed
huggingface-cli download kh4dien/pile-uncopyrighted-sample \
    --repo-type dataset \
    --local-dir /root/dictionary_learning/pile-uncopyrighted

wandb login aa9a791c5e40fa7ab2f08d555ff72352c1cecaa2