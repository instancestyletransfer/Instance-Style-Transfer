# Instance-Style-Transfer

This repo hosts the code for the paper, A Method for Arbitrary Instance Style Transfer.

We hightly encourage users to create a virtual environment to run the code in this repo. In addition, please clone this repo within a folder rather than on your desktop (i.e. ./Desktop/.../Instance-Style-Transfer) because users will download pre-trained models.


Please follow the steps below to use the codes in this repo:
1. run `pip install -r requirements.txt` to install the necessary packages
2. Download pre-trained models using the batch or bash scripts in the ***models*** folder and place these models (relu1_1, relu2_1, relu3_1, relu4_1, relu5_1, and vgg_normalised.t7) within the ***models*** folder.
3. For example, if users want to transfer the style of the "water.jpg" image in the ***style_images*** folder to the bird in "bird.jpg" in the ***content_images*** folder, run

`python instance_style_transfer.py .//content_images//bird.jpg .//style_images//water.jpg`

4. The output "bird-water.jpg" will be saved in the ***output*** folder






*Tested on AWS Deep Learning AMI (Amazon Linus) instance and Windows 7 virtual environment*
