# Are Language-and-Vision Transformers Sensitive to Discourse? The case study of ViLBERT

This is a repository with the code for Master’s thesis [“Are Language-and-Vision Transformers Sensitive to Discourse? The case study of ViLBERT”](https://github.com/EkaterinaVoloshina/multimodal_discourse_probing/blob/main/thesis_text.pdf)

## What it is about?

Language-and-vision models have shown good performance on tasks such as image-caption matching and caption generation. However, it is challenging for such models to generate *pragmatically* correct captions, which adequately reflect what is happening in one or several images. Here we explore to what extent contextual language-and-vision models are sensitive to different discourses: textual, visual, and situation-level.
We evaluate ViLBERT, one of the multi-modal transformers, if it can match descriptions and images, differentiating them from distractors of different degree of similarity. We run three experiments on ViLBERT with different conditions. As for the data, we use Visual Storytelling dataset. 
Our results reveal that the model can distinguish two different narratives but is not sensitive to difference within one story and the effect of dis/similarity of distractors becomes more pronounced when the nature of discourse becomes more challenging.

## Setup and Usage

To reproduce the experiments, clone the repository and install the dependencies:

```bash
git clone https://github.com/EkaterinaVoloshina/multimodal_discourse_probing
cd multimodal_discourse_probing
pip install -r requirements.txt
```

To download the Visual Storytelling dataset, run the following code:

```bash
sh data/download_vist.sh
```

To generate datasets for experiments, run the following commands:

```bash
cd preprocessing
echo "Calculating image similarity"
python calculate_image_similarity.py

echo "Calculating text similarity"
python calculate_text_similarity.py --input_file ../data/preprocessed_sis/caption_test.json --device cuda --split test_sis

echo "Generating datasets"
sh generate_h5_files.sh
sh generate_files.sh
```

To run the probing experiments, use the following commands (note that the code for encoders in VOLTA was modified):

```bash
cd probing
sh run_probing.sh
sh run_sis_probing.sh
```
