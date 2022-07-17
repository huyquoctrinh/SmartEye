# Introduction
Smart App for Visually Impaired of No name team in IT hackathon.

# Data
We use [Flickr](https://shannon.cs.illinois.edu/DenotationGraph/) 30k images for training the model .

# Model
We build the model with VGG16 the attention model, our model is based on the idea
of [CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/You_Image_Captioning_With_CVPR_2016_paper.pdf).

#Evaluation 
Below is our evaluation:

Metrics  | Score
------------- | -------------
BLEU  | 0.71
CrossentropyLoss  | 0.38

# Packages
All package to run the server in the ```requirements.txt``` , run the following command to install.

``` bash
  pip install -r requirements.txt
```

# Run Server

OCR server:

- To run the server for OCR:

``` bash
 cd ./ocr-api
 sudo python3 server.py
```

- To run the captioning server:

```bash
sudo python3 server.py
```
# License
[MIT](https://choosealicense.com/licenses/mit/)