import numpy as np
import json
import random
import argparse
import os

from sklearn.metrics.pairwise import cosine_similarity

def confuse_captions(vectors, images):
        # implement distribution estimation and calculate quantiles
        cos_sim = cosine_similarity(vectors)
        low_threshold, top_threshold = np.quantile(cos_sim, [0.25, 0.75])

        similar_vectors = {}
        dissimilar_vectors = {}

        for i, vector in enumerate(cos_sim):
            sorted_vector = np.argsort(vector)
            vector = np.sort(vector)
            dissimilar = sorted_vector[vector <= low_threshold]
            dissimilar_values = vector[vector <= low_threshold]
            similar = sorted_vector[vector >= top_threshold]
            similar_values = vector[vector >= top_threshold]
            similar_vectors[images[i]] = list(zip(map(lambda x: images[x], similar), map(str,similar_values)))
            dissimilar_vectors[images[i]] = list(zip(map(lambda x: images[x], dissimilar), map(str, dissimilar_values)))
       
        return similar_vectors, dissimilar_vectors
     
def load_data(dirname, input_vectors, input_ids):
    with open(os.path.join(dirname, input_ids)) as file:
        ids = file.read().split()
    vectors = np.load(os.path.join(dirname,input_vectors))
    return vectors, ids

def read_json(filename):
    with open(filename) as file:
        captions = json.load(file)
    return captions
    
def write_json(filename, captions):
    with open(filename, "w") as file:
        json.dump(captions, file)
        
def get_random_cap(captions):
    shuffled_captions = {}
    all_captions = list(captions.values())
    for image, original_cap in captions.items():
        random_cap = random.choice(all_captions)
        while original_cap == random_cap:
            random_cap = random.choice(all_captions)
        shuffled_captions[image] = random_cap
    return shuffled_captions

def get_caption_by_similarity(captions, similar, shuffle_captions=True):
    shuffled_captions = {}
    for image, original_cap in captions.items():
        if image in similar and similar[image] != []:
            similar_image = random.choice(similar[image])
            if shuffle_captions:
                similar_cap = captions[similar_image[0]]
                shuffled_captions[image] = similar_cap
            else:
                shuffled_captions[similar_image[0]] = original_cap
    return shuffled_captions

def generate_datasets(captions, text_vectors, text_ids, image_vectors, image_ids, output_dir):
    similar_captions, dissimilar_captions = confuse_captions(text_vectors, text_ids)
    similar_images, dissimilar_images = confuse_captions(image_vectors, image_ids)
    
    random_captions = get_random_cap(captions) 
    write_json(os.path.join(output_dir, "caption_valid_random.json"), random_captions)

    similar_captions = get_caption_by_similarity(captions, similar_captions)
    write_json(os.path.join(output_dir, "caption_valid_similar_captions.json"), similar_captions)

    dissimilar_captions = get_caption_by_similarity(captions, dissimilar_captions)
    write_json(os.path.join(output_dir, "caption_valid_dissimilar_captions.json"), dissimilar_captions)

    similar_images = get_caption_by_similarity(captions, similar_images, shuffle_captions=False)
    write_json(os.path.join(output_dir, "caption_valid_similar_images.json"), similar_images)

    dissimilar_images = get_caption_by_similarity(captions, dissimilar_images, shuffle_captions=False)
    write_json(os.path.join(output_dir, "caption_valid_dissimilar_images.json"), dissimilar_images)
    
def main():
    parser = argparse.ArgumentParser(description="Generates datasets.")
    parser.add_argument("--dirname", default="vectors", help="a directory where vectors are saved.")
    parser.add_argument("--output_dir", default="datasets", help="a directory to save datasets to.")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
        
    print("Loading_data..")
    captions = read_json("caption_valid.json")
    text_vectors, text_ids = load_data(args.dirname, "text_vectors.npy", "text_ids.txt")
    image_vectors, image_ids = load_data(args.dirname, "image_vectors.npy", "image_ids.txt")
    
    print("Generating data..")
    generate_datasets(captions, text_vectors, text_ids, image_vectors, image_ids, args.output_dir)
    print("Done!")
    
if __name__=="__main__":
    main()