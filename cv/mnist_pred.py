# Importing the Keras libraries and packages
from keras.models import load_model
import numpy as np
model = load_model('mnist_NN.h5')

from PIL import Image
import numpy as np

for index in range(10):
    img = Image.open('data/' + str(index) + '.png').convert("L")
    img = img.resize((28,28))
    im2arr = np.array(img)
    # normalize inputs from 0-255 to 0-1
    im2arr = im2arr / 255
    im2arr = im2arr.reshape(1,784)    
    # Predicting the Test set results
    y_pred = model.predict(im2arr)
    print(str(index) , "=" , y_pred)
    print(str(index) , "=" , np.argmax(y_pred))
    