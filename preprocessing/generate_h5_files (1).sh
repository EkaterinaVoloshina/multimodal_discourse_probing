echo "Generating a file for original images..."
python flickr30k_boxes36_h5-proposal.py --root ../probing/data/vist_images/test --outdir vist_files/ --captions ../probing/data/preprocessed_dii/caption_test.json --split original_images_test --original true

echo "Generating a file for similar images..."
python flickr30k_boxes36_h5-proposal.py --root ../probing/data/vist_images/test --outdir vist_files/ --captions ../probing/preprocessing/datasets/caption_test_similar_images_relative.json --split similar_images_test_relative

python flickr30k_boxes36_h5-proposal.py --root ../probing/data/vist_images/test --outdir vist_files/ --captions ../probing/preprocessing/datasets/caption_test_similar_images_absolute.json --split similar_images_test_absolute

echo "Generating a file for dissimilar images..."
python flickr30k_boxes36_h5-proposal.py --root ../probing/data/vist_images/test --outdir vist_files/ --captions ../probing/preprocessing/datasets/caption_test_dissimilar_images_relative.json --split dissimilar_images_test_relative

python flickr30k_boxes36_h5-proposal.py --root ../probing/data/vist_images/test --outdir vist_files/ --captions ../probing/preprocessing/datasets/caption_test_dissimilar_images_absolute.json --split dissimilar_images_test_absolute
