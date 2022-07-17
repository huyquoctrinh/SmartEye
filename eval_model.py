
import jsonpickle
import numpy as np
import tensorflow as tf
import cv2
from pickle import load
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input,VGG16
from keras.models import Model
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
import load_data as ld
import generate_model as gen
import argparse
# Initialize the Flask application

def load_model_fea():
  # load the model
  model = VGG16(include_top = False,weights="imagenet")
  # re-structure the model
  model.layers.pop()

  model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

  return model
def extract_features(model,filename):
  # load the photo
  image = load_img(filename, target_size=(224, 224))
  # convert the image pixels to a numpy array
  image = img_to_array(image)
  # reshape data for the model
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  # prepare the image for the VGG model
  image = preprocess_input(image)
  # get features
  feature = model.predict(image, verbose=0)
  return feature

# generate a description for an image
def generate_desc(model, tokenizer, photo, index_word, max_length, beam_size=5):

  captions = [['startseq', 0.0]]
  # seed the generation process
  in_text = 'startseq'
  # iterate over the whole length of the sequence
  for i in range(max_length):
    all_caps = []
    # expand each current candidate
    for cap in captions:
      sentence, score = cap
      # if final word is 'end' token, just add the current caption
      if sentence.split()[-1] == 'endseq':
        all_caps.append(cap)
        continue
      # integer encode input sequence
      sequence = tokenizer.texts_to_sequences([sentence])[0]
      # pad input
      sequence = pad_sequences([sequence], maxlen=max_length)
      # predict next words
      y_pred = model.predict([photo,sequence], verbose=0)[0]
      # convert probability to integer
      yhats = np.argsort(y_pred)[-beam_size:]

      for j in yhats:
        # map integer to word
        word = index_word.get(j)
        # stop if we cannot map the word
        if word is None:
          continue
        # Add word to caption, and generate log prob
        caption = [sentence + ' ' + word, score + np.log(y_pred[j])]
        all_caps.append(caption)

    # order all candidates by score
    ordered = sorted(all_caps, key=lambda tup:tup[1], reverse=True)
    captions = ordered[:beam_size]

  return captions
def evaluate_model(model, descriptions, photos, tokenizer, index_word, max_length):
  actual, predicted = list(), list()
  # step over the whole set
  for key, desc_list in descriptions.items():
    # generate description
    yhat = generate_desc(model, tokenizer, photos[key], index_word, max_length)[0]
    # store actual and predicted
    references = [d.split() for d in desc_list]
    actual.append(references)
    # Use best caption
    predicted.append(yhat[0].split())
  # calculate BLEU score
  print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
  print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
  print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
  print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

def eval_test_set(model, descriptions, photos, tokenizer, index_word, max_length):
  actual, predicted = list(), list()
  # step over the whole set
  for key, desc_list in descriptions.items():
    # generate description
    yhat = generate_desc(model, tokenizer, photos[key], index_word, max_length)[0]
    # store actual and predicted
    references = [d.split() for d in desc_list]
    actual.append(references)
    # Use best caption
    predicted.append(yhat[0].split())
  predicted = sorted(predicted)
  actual = [x for _,x in sorted(zip(actual,predicted))]
def cap(model_cap,model_fea,index_word,tokenizer,file_img):
  # tokenizer = load(open('tokenizer.pkl', 'rb'))
  # index_word = load(open('index_word.pkl', 'rb'))
  # pre-define the max sequence length (from training)
  max_length = 34

  # load the model
  
  # model = load_model(filename)
    # load and prepare the photograph
  photo = extract_features(model_fea,file_img)
  print(photo.shape)
    # generate description
  captions = generate_desc(model_cap, tokenizer, photo, index_word, max_length)
  print(captions)
  return captions[1][0].replace('startseq',"").replace('endseq',"")

# model_fea = load_model_fea()
# filename = 'model_weight.h5'
# model_cap = tf.keras.models.load_model(filename)
# tokenizer = load(open('tokenizer.pkl', 'rb'))
# index_word = load(open('index_word.pkl', 'rb'))

# print(cap(model_cap,model_fea,index_word,tokenizer,"d.jpg"))

