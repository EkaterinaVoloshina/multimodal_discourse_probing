echo "Generating datasets for experiment 1"
python generate_datasets.py --relative true --text_dir ../data/preprocessed_dii --dataset dii

echo "Generating datasets for experiment 2"
python generate_datasets.py --relative true --text_dir ../data/preprocessed_sis --dataset sis