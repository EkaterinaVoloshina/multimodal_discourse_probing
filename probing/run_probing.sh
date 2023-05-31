#echo "Running on random captions.."
#python run_probing.py --original_data_path ../probing/data/preprocessed_dii/caption_test.json --shuffled_data_path ../probing/preprocessing/datasets/caption_test_random_dii.json --orig_lmdb vist_files/original_images_test_dii.lmdb --shuf_lmdb vist_files/original_images_test_dii.lmdb --log_file vist_logs/log_random_dii.json

#echo "Running on similar captions (relative).."
#python run_probing.py --original_data_path ../probing/data/preprocessed_dii/caption_test.json --shuffled_data_path ../probing/preprocessing/datasets/caption_test_similar_captions_relative_dii.json --orig_lmdb vist_files/original_images_test_dii.lmdb --shuf_lmdb vist_files/original_images_test_dii.lmdb --log_file vist_logs/log_similar_captions_relative_dii.json

#echo "Running on dissimilar captions (relative).."
#python run_probing.py --original_data_path ../probing/data/preprocessed_dii/caption_test.json --shuffled_data_path ../probing/preprocessing/datasets/caption_test_dissimilar_captions_relative_dii.json --orig_lmdb vist_files/original_images_test_dii.lmdb --shuf_lmdb vist_files/original_images_test_dii.lmdb --log_file vist_logs/log_dissimilar_captions_relative_dii.json

#echo "Running on similar images (relative).."
#python run_probing.py --original_data_path ../probing/data/preprocessed_dii/caption_test.json --shuffled_data_path ../probing/preprocessing/datasets/caption_test_similar_images_relative_dii.json --orig_lmdb vist_files/original_images_test_dii.lmdb --shuf_lmdb vist_files/similar_images_test_relative_dii.lmdb --log_file vist_logs/log_similar_images_relative_dii.json --task texts

#echo "Running on dissimilar images (relative).."
#python run_probing.py --original_data_path ../probing/data/preprocessed_dii/caption_test.json --shuffled_data_path ../probing/preprocessing/datasets/caption_test_dissimilar_images_relative_dii.json --orig_lmdb vist_files/original_images_test_dii.lmdb --shuf_lmdb vist_files/dissimilar_images_test_relative_dii.lmdb --log_file vist_logs/log_dissimilar_images_relative_dii.json --task texts


#echo "Running on random captions.."
#python run_probing.py --original_data_path ../probing/data/preprocessed_sis/caption_test.json --shuffled_data_path ../probing/preprocessing/datasets/caption_test_random_sis.json --orig_lmdb vist_files/original_images_test_sis.lmdb --shuf_lmdb vist_files/original_images_test_sis.lmdb --log_file vist_logs/log_random_sis.json

#echo "Running on similar captions (relative).."
#python run_probing.py --original_data_path ../probing/data/preprocessed_sis/caption_test.json --shuffled_data_path ../probing/preprocessing/datasets/caption_test_similar_captions_relative_sis.json --orig_lmdb vist_files/original_images_test_sis.lmdb --shuf_lmdb vist_files/original_images_test_sis.lmdb --log_file vist_logs/log_similar_captions_relative_sis.json

#echo "Running on dissimilar captions (relative).."
#python run_probing.py --original_data_path ../probing/data/preprocessed_sis/caption_test.json --shuffled_data_path ../probing/preprocessing/datasets/caption_test_dissimilar_captions_relative_sis.json --orig_lmdb vist_files/original_images_test_sis.lmdb --shuf_lmdb vist_files/original_images_test_sis.lmdb --log_file vist_logs/log_dissimilar_captions_relative_sis.json

echo "Running on similar images (relative).."
python run_probing.py --original_data_path ../probing/data/preprocessed_sis/caption_test.json --shuffled_data_path ../probing/preprocessing/datasets/caption_test_similar_images_relative_sis.json --orig_lmdb vist_files/original_images_test_sis.lmdb --shuf_lmdb vist_files/similar_images_test_relative_sis.lmdb --log_file vist_logs/log_similar_images_relative_sis.json --task texts

echo "Running on dissimilar images (relative).."
python run_probing.py --original_data_path ../probing/data/preprocessed_sis/caption_test.json --shuffled_data_path ../probing/preprocessing/datasets/caption_test_dissimilar_images_relative_sis.json --orig_lmdb vist_files/original_images_test_sis.lmdb --shuf_lmdb vist_files/dissimilar_images_test_relative_sis.lmdb --log_file vist_logs/log_dissimilar_images_relative_sis.json --task texts

#echo "Running on story captions.."
#python run_probing.py --original_data_path ../probing/data/preprocessed_sis/caption_test.json --shuffled_data_path ../probing/preprocessing/datasets/stories_captions_test.json --orig_lmdb vist_files/original_images_test_sis.lmdb --shuf_lmdb vist_files/original_images_test_sis.lmdb --log_file vist_logs/log_stories_captions_sis.json

echo "Running on story images.."
python run_probing.py --original_data_path ../probing/data/preprocessed_sis/caption_test.json --shuffled_data_path ../probing/preprocessing/datasets/stories_images_test.json --orig_lmdb vist_files/original_images_test_sis.lmdb --shuf_lmdb vist_files/stories_images_test.lmdb --log_file vist_logs/log_stories_images_sis.json --task texts
