import tensorflow as tf
import numpy as np

# from utils.plotter import display



class UnetInferrer:
    def __init__(self):
        self.image_size = 180
        self.class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
        self.TF_MODEL_FILE_PATH = 'unet/model.tflite'
        self.classify_lite = self.loadModel()
        # print(list(    self.model.signatures.keys()))

        # self.predict = self.model.signatures["serving_default"]
        # print(self.predict.structured_output  s)

    def loadModel(self):
        interpreter = tf.lite.Interpreter(model_path=self.TF_MODEL_FILE_PATH)
        interpreter.get_signature_list()
        return interpreter.get_signature_runner('serving_default')
        

    def preprocess(self, image):
        image = tf.image.resize(image, (self.image_size, self.image_size))
        return tf.cast(image, tf.float32) / 255.0

    def infer(self, image=None):
        tensor_image = tf.convert_to_tensor(image, dtype=tf.float32)
        tensor_image = self.preprocess(tensor_image)
        shape= tensor_image.shape
        tensor_image = tf.reshape(tensor_image,[1, shape[0],shape[1], shape[2]])
        # print(tensor_image.shape)
        self.loadModel()
        predictions_lite = self.classify_lite(sequential_2_input=tensor_image)['outputs']
        score_lite = tf.nn.softmax(predictions_lite)

        # pred = self.predict(tensor_image)['conv2d_transpose_4']
        # display([tensor_image[0], pred[0]])
        # pred = pred.numpy().tolist()
        return { "This image most likely belongs to {} with a {:.2f} percent confidence.".format(self.class_names[np.argmax(score_lite)], 100 * np.max(score_lite))}