from flask import Flask
from google_drive_downloader import GoogleDriveDownloader as gdd
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from flask_cors import CORS


app = Flask(__name__,
            static_url_path='',
            static_folder='public')

CORS(app, support_credentials=True)
DIR_PATH = dir_path = os.path.dirname(os.path.realpath(__file__))
new_model = tf.keras.models.load_model(os.path.join(DIR_PATH, f'content/best_model.h5'))


@app.route('/predict/<string:image_id>', methods=['GET'])
def predict(image_id):

    gdd.download_file_from_google_drive(file_id=image_id,
                                        dest_path=os.path.join(DIR_PATH, f'imgs/{image_id}'), overwrite=True)

    img = cv2.imread(os.path.join(DIR_PATH, f'imgs/{image_id}'))
    img = np.array(img)/255

    images_list = [np.array(img)]
    x = np.asarray(images_list)

    prediction = new_model.predict(x)[0]
    print(prediction)

    classes = ["Dark", "Green", "Light", "Medium"]
    bean_prediction = classes[prediction.argmax()]

    os.remove(os.path.join(DIR_PATH, f'imgs/{image_id}'))
    
    #response = {'bean': bean_prediction}
    #response.headers.add('Access-Control-Allow-Origin', '*')

    return {'bean': bean_prediction}


if __name__ == '__main__':
    app.run(port=4242, debug=True)
