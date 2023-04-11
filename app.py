from flask import Flask, render_template, request
import joblib as jbl
from skimage.io import imread, imshow
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder='templates')


@app.route('/', methods = ['GET'])
def hello():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def predict():
    imagefile = request.files['image']
    image_path = './image_out/'+ imagefile.filename
    imagefile.save(image_path)

    knn_ocr = jbl.load('HindiKNN-3.pkl')
        
    image = imread(image_path, as_gray=True)
    i = np.reshape(image, (32 * 32))
    i_ = pd.DataFrame(i)
    img = i_.T

    xyz = knn_ocr.predict(img)

    classes = {0:u'\u0915',1:u'\u0916',2:u'\u0917',3:u'\u0918',4:u'\u0919',5:u'\u091A',6:u'\u091B',7:u'\u091C',8:u'\u091D',9:u'\u091E',10:u'\u091F',11:u'\u0920',12:u'\u0921',13:u'\u0922',14:u'\u0923',15:u'\u0924',16:u'\u0925',17:u'\u0926',18:u'\u0927',19:u'\u0928',20:u'\u092A',21:'\u092B',22:u'\u092C',23:u'\u092D',24:u'\u092E',25:u'\u092F',26:u'\u0930',27:u'\u0932',28:u'\u0935',29:u'\u0936',30:u'\u0937',31:u'\u0938',32:u'\u0939',33:'ksha',34:'tra',35:'gya',36:u'\u0966',37:u'\u0967',38:u'\u0968',39:u'\u0969',40:u'\u096a',41:u'\u096b',42:u'\u096c',43:u'\u096d',44:u'\u096e',45:u'\u096f'}

    letter = classes[xyz[0]]

    return render_template('index.html', prediction = letter)

if __name__ == '__main__':
    app.run(port=3000, debug=True)