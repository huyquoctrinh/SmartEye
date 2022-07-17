# Introduction
Smart App for Visually Impaired of No name team in IT hackathon :clown_face: :clown_face:

## Data
We use [Flickr](https://shannon.cs.illinois.edu/DenotationGraph/) 30k images for training the model .

## Model
We build the model with VGG16 the attention model, our model is based on the idea
of [CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/You_Image_Captioning_With_CVPR_2016_paper.pdf).

## Evaluation 
Below is our evaluation:

Metrics  | Score
------------- | -------------
BLEU  | 0.71
CrossentropyLoss  | 0.38

## Packages
All package to run the server in the ```requirement.txt``` , run the following command to install.

``` bash
  pip install -r requirement.txt
```
## System architecture

On AWS, we set up the Auto Scaling group with the following architecture to help scale the product. Moreover, we need to setup the Message Broker 

## Run Server

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
The server will run on the port 80, let deploy on 2 different instance and setup nginx with uwsgi for proxy server.

## App demo

Our app demo is written by React-native, all of the implementation can be seen on the following link [Github](https://github.com/sonhv3112/SmartEye)

## Contribution

All of the model, architecture was implemented by No Name team, thanks to all for the great IT hackathon 2022 !!! :drooling_face: :drooling_face: :drooling_face:

## License
[MIT](https://choosealicense.com/licenses/mit/)
