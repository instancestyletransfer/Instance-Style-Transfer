# Instance-Style-Transfer

This repo hosts the code for the paper, A Method for Arbitrary Instance Style Transfer.

Stats of our method:
1. Space of the entire folder (after downloading all the pre-trained models and removing all the zip files): 986MB
2. Run Time: less than 1 min on all three types of machines specified at the bottom

We hightly encourage users to create a virtual environment to run the code in this repo. In addition, please clone this repo to a folder rather than putting the package on your desktop (i.e. ./Desktop/.../Instance-Style-Transfer-master) because users will download pre-trained models.


Please follow the steps below to use the codes in this repo:
1. run `pip install -r requirements.txt` to install the necessary packages
2. Download pre-trained models using the *download_models* and *download_vgg* batch or bash scripts in the ***models*** folder and place these models (relu1_1, relu2_1, relu3_1, relu4_1, relu5_1, and vgg_normalised.t7) within the ***models*** folder.
3. For example, if users want to transfer the style of the "water.jpg" image in the ***style_images*** folder to the bird in "bird.jpg" in the ***content_images*** folder, run

`python instance_style_transfer.py .//content_images//bird.jpg .//style_images//water.jpg`

4. The output "bird-water.jpg" will be saved in the ***output*** folder






*The steps have been tested on AWS Deep Learning AMI (Amazon Linus) instance - python 3.6.5, MacBook Air macOS virtual environment - python 3.6.4, and Windows 7 virtual environment - python 3.6.5*
