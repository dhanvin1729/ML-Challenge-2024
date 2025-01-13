# ML-Challenge-2024
ML Challenge Problem Statement
Feature Extraction from Images
In this hackathon, the goal is to create a machine learning model that extracts entity values from images. This capability is crucial in fields like healthcare, e-commerce, and content moderation, where precise product information is vital. As digital marketplaces expand, many products lack detailed textual descriptions, making it essential to obtain key details directly from images. These images provide important information such as weight, volume, voltage, wattage, dimensions, and many more, which are critical for digital stores.

Data Description:
The dataset consists of the following columns:

1) index: An unique identifier (ID) for the data sample
2) image_link: Public URL where the product image is available for download.
3) group_id: Category code of the product
4) entity_name: Product entity name. For eg: “item_weight”
5) entity_value: Product entity value. For eg: “34 gram”

Output Format:
The output file should be a csv with 2 columns:

1) index: The unique identifier (ID) of the data sample. Note the index should match the test record index.
2) prediction: A string which should have the following format: “x unit” where x is a float number in standard formatting and unit is one of the allowed units (allowed units are mentioned in the Appendix). The two values should be concatenated and have a space between them. For eg: “2 gram”, “12.5 centimetre”, “2.56 ounce” are valid. 

File Descriptions:

_Source files_

src/sanity.py: Sanity checker to ensure that the final output file passes all formatting checks. Note: the script will not check if less/more number of predictions are present compared to the test file. See sample code in src/test.ipynb

src/utils.py: Contains helper functions for downloading images from the image_link.
src/constants.py: Contains the allowed units for each entity type.

sample_code.py: We also provided a sample dummy code that can generate an output file in the given format. Usage of this file is optional.

_Dataset files_

dataset/train.csv: Training file with labels (entity_value).

dataset/test.csv: Test file without output labels (entity_value). Generate predictions using your model/solution on this file's data and format the output file to match sample_test_out.csv (Refer the above section "Output Format")

dataset/sample_test.csv: Sample test input file.

dataset/sample_test_out.csv: Sample outputs for sample_test.csv. The output for test.csv must be formatted in the exact same way. Note: The predictions in the file might not be correct

# Our Solution
We decided to fine-tune the PaliGemma model, available on Hugging Face. PaliGemma is a versatile and lightweight vision-language model (VLM) inspired by PaLI-3 and based on open components such as the SigLIP vision model and the Gemma language model. It takes both image and text as input and generates text as output, supporting multiple languages. PaliGemma is the composition of a Transformer decoder and a Vision Transformer image encoder, with a total of 3 billion params. The text decoder is initialized from Gemma-2B. The image encoder is initialized from SigLIP-So400m/14.

**Data Preparation**

The dataset is read from a CSV file, and unnecessary columns are dropped. The data is then split into training and evaluation datasets. Each dataset entry contains:
	•	An image link.
	•	A corresponding question.
	•	The expected answer.

**Pre-trained Model Setup**

The project uses PaliGemma, a model pre-trained for vision-language tasks:
	•	The model is loaded in 4-bit quantized mode for efficient GPU memory utilization.
	•	The processor (PaliGemmaProcessor) is configured to handle text and image inputs.
 
**Fine-tuning with LoRA**

LoRA is applied for lightweight fine-tuning:
	•	Target modules like q_proj, k_proj, and others are adapted for task-specific tuning.
	•	Trainable parameters are minimized while retaining high performance.

 **Training Loop**

A custom data collator handles image loading and tokenization:
	•	Images are fetched from the URLs, resized, and converted to RGB format.
	•	Questions and answers are tokenized into the input format required by the model.

The SFTTrainer from Hugging Face’s trl library is used for supervised fine-tuning:
	•	Key training hyperparameters, including batch size, learning rate, and weight decay, are configured.
	•	The training process logs progress and saves model checkpoints.

