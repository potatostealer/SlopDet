import os

# Optional: point the HuggingFace caches at a large disk before downloading, e.g.:
# os.environ["HF_HOME"] = "/path/to/hf_home"
# os.environ["HF_DATASETS_CACHE"] = "/path/to/hf_datasets_cache"
# os.environ["TMPDIR"] = "/path/to/tmp"
# If the dataset requires authentication:  export HF_TOKEN=<your HuggingFace token>

from datasets import load_dataset

ds = load_dataset("<DATASET_NAME>", cache_dir=os.environ.get("HF_DATASETS_CACHE"))
