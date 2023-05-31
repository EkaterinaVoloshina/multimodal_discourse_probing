mkdir -p vectors
echo "Calculating for train.."
python calculate_text_similarity.py --input_file ../data/preprocessed_dii/captions_train.json --device cuda --split train
echo "Train is created!"
echo "Calculating for validation.."
python calculate_text_similarity.py --input_file ../data/preprocessed_dii/captions_valid.json --device cuda --split valid
echo "Validation is created!"
echo "Calculating for test.."
python calculate_text_similarity.py --input_file ../data/preprocessed_dii/captions_test.json --device cuda --split test
echo "Test is created!"
