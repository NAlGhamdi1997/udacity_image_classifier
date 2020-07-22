#import libararies
import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image


#Setting Up a Parser
parser = argparse.ArgumentParser(description='Flowers Images Classifier')

#Defining Arguments
parser.add_argument('--image_path', default='./test_images/orange_dahlia.jpg', help='The path of image')
parser.add_argument('--model', default = 'best_model.h5', help='The model path')
parser.add_argument('--top_k',type= int, default=5, help = 'Number of top images')
parser.add_argument('--classes', default = 'label_map.json', help='The names of classes')

#Parsing a Command-Line
args = parser.parse_args()

image_path = args.image_path
model = args.model
top_k = args.top_k
classes = args.classes

#Label Mapping
with open('label_map.json', 'r') as f:
    class_names = json.load(f)
    
#Load the Keras model
reloaded_keras_model = tf.keras.models.load_model(model, custom_objects={'KerasLayer': hub.KerasLayer})

#Image Pre-processing 
image_size = 224
def process_image(image):
    tensor_img = tf.convert_to_tensor(image)
    resized_img = tf.image.resize(tensor_img,(image_size, image_size))
    norm_img = resized_img/255
    np_img = norm_img.numpy()
    
    return np_img

#Image Prediction
def predict(image_path, model, top_k):
    img = Image.open(image_path)
    numpy_img = np.asarray(img)
    trans_img = process_image(numpy_img)
    extra_dim = np.expand_dims(trans_img,axis=0)
    prob_preds = model.predict(extra_dim)
    
    prob_preds = prob_preds.tolist()
    
    #print(prob_preds)
    
    values, index = tf.math.top_k(prob_preds , k=top_k)
    
    probs = values.numpy().tolist()[0]
    classes = index.numpy().tolist()[0]
    
    for n in classes:
        flower_names = class_names[str(n)]
        print(flower_names, probs)
    #return probs , classes

#Main
if __name__ == '__main__':
    predict(image_path, reloaded_keras_model, top_k)
