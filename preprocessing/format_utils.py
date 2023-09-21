import json


def reformat_dii(filename, output):
    with open(filename) as f:
        content = json.load(f)

    captions = {}
    for annotation in content["annotations"]:
        if annotation[0]["photo_flickr_id"] not in captions:
            captions[annotation[0]["photo_flickr_id"]] = annotation[0]["original_text"]

    with open(output, "w") as f:
        json.dump(captions, f)
