# monology/pile-uncopyrighted
# Make sure hf-transfer is installed
huggingface-cli download monology/pile-uncopyrighted \
    train/00.jsonl.zst \
    train/01.jsonl.zst \
    --repo-type dataset \
    --local-dir /root/dictionary_learning/pile-uncopyrighted