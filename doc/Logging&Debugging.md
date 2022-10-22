# Using Python debugger and the logging module to find errors in your AI application

## How to debug Deep Learning?
Deep Learning debugging is more difficult than normal software because of multiple reasons:

- Poor model performance doesn’t necessarily mean bugs in the code
- The iteration cycle (building the model, training, and testing) is quite long
- Training/testing data can also have errors and anomalies
- Hyperparameters affect the final accuracy
- It’s not always deterministic (e.g. probabilistic machine learning)
- Static computation graph (e.g. Tensorflow and CNTK)

When experimenting with our model, we should start from a very simple algorithm, with only a handful of features and gradually keep expanding by adding features and tuning hyperparameters while keeping the model simple.

## Python debugger (Pdb)
Python debugger is part of the standard python library. The debugger is essentially a program that can monitor the state of our own program while it is running. The most important command of any debugger is called a breakpoint. We can set a breakpoint anywhere in our code and the debugger will stop the execution in this exact point and give us access to the values of all the variables at that point as well as the traceback of python calls. 

## Debug Data ( schema validation)
Now that we have a decent way to find bugs in the code, let’s have a look at the second most common source of errors in Machine Learning: Data. Data isn't always in perfect form (in fact they never are). They may contain corrupted data points, some values may be missing, they may have a different format or they may have a different range/distribution than expected.

To catch all these before training or prediction, one of the most common ways is Schema Validation. We can define schema as a contract of the format of our data. Practically, the schema is a JSON file containing all the required features for a model, their form and their type. Note that it can be whichever format we want (many TensorFlow models use proto files). To monitor the incoming data and catch abnormalities, we can validate them against the schema.

Schema validation is especially useful when the model is deployed on a production environment and accepts user data. In the case of our project, once we have the UNet running in production, the user will be able to send whatever image he wants. So we need to have a way to validate them.

Since our input data are images, which is literally a 4-dimensional array of shape [batch, channels, height, width], our JSON schema will look as depicted below. (This is not what your typical schema will look like but because in our case we have images as input, we should be concise and give the correct schema)

```json 
SCHEMA = {
  "type": "object",
  "properties": {
    "image":{
        "type":"array",
        "items":{
            "type": "array",
            "items": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    }
                }
            }
        }
    }
  },
  "required":["image"]
}
```

Our data type is a python object as you can see in the first line. This object contains a property called “image” which is of type array and has the shown items. Typically your schema will end in this point but in our case, we need to go deep to declare all 4 dimensions of our image.

You can think of it as a type of recursion where we define the same item inside the other. Deep into the recursion, you define the type of our values to be numeric. Finally, the last line of the schema indicates all the required properties of our object. In this case, it’s just the image.

A more typical example of a schema will look like this:

```json 
SCHEMA = {
  "type": "object",
  "properties":{
    "feature-1":{
        "type":"string"
      },
    "feature-2":{
        "type":"integer"
     },
    "feature-3":{
        "type":"string"
    }
  },
  "required":["feature-1", "feature-3"]
}
```

Once we have our schema, we can use it to validate our data. In python the built-in jsonschema package, which can help us do exactly that.

```python
import jsonschema
from configs.data_schema import SCHEMA


class DataLoader:
    """Data Loader class"""

    @staticmethod
    def validate_schema(data_point):
        jsonschema.validate({'image':data_point.tolist()}, SCHEMA)
```

So we can call the “validate_schema” function whenever we like to check our data against our schema. Pretty easy I dare to say.


## Logging
Logging goes toe to toe with Debugging. But why do we need to keep logs? Logs are an essential part of troubleshooting application and infrastructure performance. When our code is executed on a production environment in a remote machine, let’s say Google cloud, we can’t really go there and start debugging stuff. Instead, in such remote environments, we use logs to have a clear image of what’s going on. Logs are not only to capture the state of our program but also to discover possible exceptions and errors.

But why not use simple print statements? Aren’t they enough? Actually no they are not! Why? Let’s outline some advantages logs provide over the print statement:

- We can log different severity levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) and choose to show only the level we care about. For example, we can stuff our code with debug logs but we may not want to show all of them in production to avoid having millions of rows and instead show only warnings and errors.
- We can choose the output channel (not possible with prints as they always use the console). Some of our options are writing them to a file, sending them over http, print them on the console, stream them to a secondary location, or even send them over email.
- Timestamps are included by default.
- The format of the message is easily configurable.

# Python Logging module

Python’s default module for logging is called… well logging. In order to use it all we have to do is:

```python
import logging

logging.warning('Warning. Our pants are on fire...")
```

But since we are developing a production-ready pipeline, highly extensible code, let’s use it in a more elegant way. I will go into the utils folder and create a file called “logger.py” so we can import it anywhere we like.


A simple configuration file in yaml looks something like this:
```yaml
version: 1
formatters:
  simple:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    formatter: simple
    stream: ext://sys.stdout
Root:
  Level: DEBUG
  handlers: [console]
```
As you can see, we set the default format in the formatters node, we define the console as the output channel and streaming as a transmission method and we set the default level to DEBUG. That means all logging above that level will be printed. For future reference, the order of levels are: DEBUG < INFO < WARNING < ERROR < CRITICAL

So whenever we need to log something all we have to do is import the file and use the built-in functions such as .info(), .debug() and .error()

```python
from utils.logger import get_logger

LOG = get_logger('unet')

def evaluate(self):
    """Predicts results for the test dataset"""

    predictions = []
    LOG.info('Predicting segmentation map for test dataset')

    for image, mask in self.test_dataset:
        LOG.debug(f'Predicting segmentation map {image}')
        predictions.append(self.model.predict(image))
    return predictions
```

Tensorflow code is not your normal node and as we said it’s not trivial to debug and test it. One of the main reasons is that Tensorflow used to have a static computational graph, meaning that you had to define the model, compile it and then run it. This made debugging much much harder, because you couldn’t access variables and states as you normally do in python.

However, in Tensorflow 2.0 the default execution mode is the eager (dynamic) mode, meaning that the graph is dynamic following the PyTorch pattern. Of course, there are still cases when the code can’t be executed eagerly. And even now, the computational graph still exists in the background. That’s why we need these functions as they have been built with that in mind. They just provide additional flexibility that normal logging simply won’t.

- tf.print: is Tensorflow built-in print function that can be used to print tensors but also let us define the output stream and the current level. It’s ease of use is based on the fact that it is actually a separate component inside the computational graph, so it communicates by default with all other components. Especially in the case that some function is not run eagerly, normal print statements won’t work and we have to use tf.print().

- tf.Variable.assign: can be used to assign values to a variable during runtime, in case you want to test things or explore different alternatives. It will directly change the computational graph so that the new value can be picked from the rest of the nodes.

- tf.summary: provides an api to write summary data into files. Let’s say you want to save metrics on a file or some tensor to track its values. You can do just that with tf.summary. In essence it’s a logging system to save anything you like into a file. Plus it is integrated with Tensorboard so you can visualize your summaries with little to no effort.

- tf.debugging: is a set of assert functions (tailored to tensors) that can be put inside your code to validate your data, your weights or your model.

- tf.debugging.enable_check_numerics: is part of the same module but I had to mention it separately because it’s simply amazing. This little function will cause the code to error out as soon as an operation’s output tensor contains infinity or NaN. Do I need to say more?

- get_concrete_function(input).graph: This simple but amazing simple function can be used to convert any python function into a tf.Graph so we can access all sorts of things from here (shapes, value types etc).

- tf.keras.callbacks: are functions that are used during training to pass information to external sources. The most common use case is passing training data into Tensorboard but that is not all. They can also be used to save csv data, early stop the training based on a metric or even change the learning rate. It’s an extremely useful tool especially for those who don’t to write Tensorflow code and prefer the simplicity of Keras


