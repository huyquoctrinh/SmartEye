from flask import Flask,request,redirect, url_for, send_from_directory,Response,jsonify
from ocr import detect_text
import os
app = Flask(__name__)

@app.route("/detect",methods = ["GET","POST"])
def detection():
    if request.method == "GET":
        return "ping success"
    elif request.method == "POST":
        imgFile = request.files['img_file']
        print(imgFile.filename)
        imgFile.save(imgFile.filename)
        res = detect_text(imgFile.filename)
        os.remove(imgFile.filename)
        return jsonify({
            "content":res,
            "status":"success",
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)