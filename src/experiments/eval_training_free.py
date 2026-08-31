"""Training-free K-NN evaluation on the labelled test set.

Run from the repo root:
    python -m src.experiments.eval_training_free [--config src/configs/training_free.yml] [--device cuda:4] [--k 50] [--limit 64]

Identical to training_free.py -- same frozen Siglip2 pooled image embeddings,
same reference set (the train split of data.dataset_config), same cosine K-NN
majority vote with exact ties reported as indecisive, same embedding caches --
except that the queries are the test directories of the config's test block:
images under test.real_img_test_ds_path have ground truth 0 (real), images
under test.aigen_img_test_ds_path ground truth 1 (AI generated, the positive
class of precision / recall / F1).

Prints the same report and writes
<output.dir>/<run_name>/test/k<K>_<weighting>/metrics.json, failures.csv and
indecisive.csv.
"""

from src.experiments.training_free import main

if __name__ == "__main__":
    main(split="test")
