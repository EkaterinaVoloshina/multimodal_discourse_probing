echo "Running on random captions.."
python run_probing.py --original_data_path ../probing/data/preprocessed_sis/caption_test.json --shuffled_data_path ../probing/preprocessing/datasets/caption_test_random.json --orig_lmdb sis_files/original_images_test.lmdb --shuf_lmdb sis_files/original_images_test.lmdb --log_file vist_logs/log_random_sis.json

echo "Running on similar captions (relative).."
python run_probing.py --original_data_path ../probing/data/preprocessed_sis/caption_test.json --shuffled_data_path ../probing/preprocessing/datasets/caption_test_similar_captions_relative.json --orig_lmdb sis_files/original_images_test.lmdb --shuf_lmdb sis_files/original_images_test.lmdb --log_file vist_logs/log_similar_captions_relative_sis.json

echo "Running on dissimilar captions (relative).."
python run_probing.py --original_data_path ../probing/data/preprocessed_sis/caption_test.json --shuffled_data_path ../probing/preprocessing/datasets/caption_test_dissimilar_captions_relative.json --orig_lmdb sis_files/original_images_test.lmdb --shuf_lmdb sis_files/original_images_test.lmdb --log_file vist_logs/log_dissimilar_captions_relative_sis.json

#echo "Running on dissimilar captions (absolute).."
#python run_probing.py --original_data_path ../probing/data/preprocessed_dii/caption_test.json --shuffled_data_path ../probing/preprocessing/datasets/caption_test_dissimilar_captions_absolute.json --orig_lmdb vist_files/original_images_test.lmdb --shuf_lmdb vist_files/original_images_test.lmdb --log_file vist_logs/log_dissimilar_captions_absolute.json

#echo "Running on similar images (relative).."
#python run_probing.py --original_data_path ../probing/data/preprocessed_sis/caption_test.json --shuffled_data_path ../probing/preprocessing/datasets/caption_test_similar_images_relative.json --orig_lmdb sis_files/original_images_test.lmdb --shuf_lmdb sis_files/similar_images_relative_test.lmdb --log_file vist_logs/log_similar_images_relative_sis.json --task texts

#echo "Running on similar images (absolute).."
#python run_probing.py --original_data_path ../probing/data/preprocessed_dii/caption_test.json --shuffled_data_path ../probing/preprocessing/datasets/caption_test_dissimilar_images_absolute.json --orig_lmdb vist_files/original_images_test.lmdb --shuf_lmdb vist_files/similar_images_absolute_test.lmdb --log_file vist_logs/log_similar_images_absolute.json --task texts

#echo "Running on dissimilar images (relative).."
#python run_probing.py --original_data_path ../probing/data/preprocessed_sis/caption_test.json --shuffled_data_path ../probing/preprocessing/datasets/caption_test_dissimilar_images_relative.json --orig_lmdb sis_files/original_images_test.lmdb --shuf_lmdb sis_files/dissimilar_images_relative_test.lmdb --log_file vist_logs/log_dissimilar_images_relative_sis.json --task texts

#echo "Running on dissimilar images (absolute).."
#python run_probing.py --original_data_path ../probing/data/preprocessed_dii/caption_test.json --shuffled_data_path ../probing/preprocessing/datasets/caption_test_dissimilar_images_absolute.json --orig_lmdb vist_files/original_images_test.lmdb --shuf_lmdb vist_files/dissimilar_images_absolute_test.lmdb --log_file vist_logs/log_dissimilar_images_absolute.json --task texts