# ML-Challenge-2024
ML Challenge Problem Statement
Feature Extraction from Images
In this hackathon, the goal is to create a machine learning model that extracts entity values from images. This capability is crucial in fields like healthcare, e-commerce, and content moderation, where precise product information is vital. As digital marketplaces expand, many products lack detailed textual descriptions, making it essential to obtain key details directly from images. These images provide important information such as weight, volume, voltage, wattage, dimensions, and many more, which are critical for digital stores.

Data Description:
The dataset consists of the following columns:

index: An unique identifier (ID) for the data sample
image_link: Public URL where the product image is available for download. Example link - https://m.media-amazon.com/images/I/71XfHPR36-L.jpg
To download images use download_images function from src/utils.py. See sample code in src/test.ipynb.
group_id: Category code of the product
entity_name: Product entity name. For eg: “item_weight”
entity_value: Product entity value. For eg: “34 gram”
Note: For test.csv, you will not see the column entity_value as it is the target variable.
Output Format:
The output file should be a csv with 2 columns:

index: The unique identifier (ID) of the data sample. Note the index should match the test record index.
prediction: A string which should have the following format: “x unit” where x is a float number in standard formatting and unit is one of the allowed units (allowed units are mentioned in the Appendix). The two values should be concatenated and have a space between them. For eg: “2 gram”, “12.5 centimetre”, “2.56 ounce” are valid. Few invalid cases: “2 gms”, “60 ounce/1.7 kilogram”, “2.2e2 kilogram” etc.
Note: Make sure to output a prediction for all indices. If no value is found in the image for any test sample, return empty string, i.e, “”. If you have less/more number of output samples in the output file as compared to test.csv, your output won’t be evaluated.
File Descriptions:
source files

src/sanity.py: Sanity checker to ensure that the final output file passes all formatting checks. Note: the script will not check if less/more number of predictions are present compared to the test file. See sample code in src/test.ipynb
src/utils.py: Contains helper functions for downloading images from the image_link.
src/constants.py: Contains the allowed units for each entity type.
sample_code.py: We also provided a sample dummy code that can generate an output file in the given format. Usage of this file is optional.
Dataset files

dataset/train.csv: Training file with labels (entity_value).
dataset/test.csv: Test file without output labels (entity_value). Generate predictions using your model/solution on this file's data and format the output file to match sample_test_out.csv (Refer the above section "Output Format")
dataset/sample_test.csv: Sample test input file.
dataset/sample_test_out.csv: Sample outputs for sample_test.csv. The output for test.csv must be formatted in the exact same way. Note: The predictions in the file might not be correct
