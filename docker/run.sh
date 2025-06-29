docker run --gpus all \
    --rm -dit \
    --shm-size=30g \
    --memory=200g \
    -v "$(pwd)":/rlgdg \
    -v "$HOME/.cache/huggingface":/rlgdg/.cache/huggingface \
    --name rlgdg_container \
    rlgdg bash
