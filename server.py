from flask import Flask,request,redirect, url_for, send_from_directory,Response,jsonify
from eval_model import *
from googletrans import Translator
import os
from translation import translate_text
trans = Translator()
model_fea = load_model_fea()
filename = 'huy.h5'
model_cap = tf.keras.models.load_model(filename)
tokenizer = load(open('tokenizer.pkl', 'rb'))
index_word = load(open('index_word.pkl', 'rb'))

app = Flask(__name__)

@app.route("/caption",methods = ["GET", "POST"])
def captioning():
    if request.method == "GET":
        return "ping success"
    elif request.method == "POST":
        img_file = request.files['img_file']
        # name = request.form["function"]
        img_file.save(img_file.filename)
        caption = cap(model_cap,model_fea,index_word,tokenizer,img_file.filename)
        os.remove(img_file.filename)
        return jsonify(
            {
                'status':'sucess',
                'content': translate_text("vi",caption)
            }
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)