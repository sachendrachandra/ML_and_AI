from flask import Flask, render_template, request
import Caption_it
# import pickle
# import pandas as pd
# import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/", methods=['POST'])
def predict():
    # print("I was here 1")
    if request.method == 'POST':
        f=request.files['userfile']
        path = "./static/{}".format(f.filename)
        f.save(path)

        Caption = Caption_it.caption(path)
        # print(Caption)
        query = Caption
        stopwords = ['startseq','endseq']
        querywords = query.split()

        resultwords  = [word for word in querywords if word.lower() not in stopwords]
        result = ' '.join(resultwords)

        result_dic={
        'image': path,
        'cap': result
        }
    return render_template("index.html", your_result = result_dic)


if __name__ == "__main__":

    app.run(host='0.0.0.0',port=5001,debug=True)
