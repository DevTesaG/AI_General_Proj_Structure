# Data preprocessing for deep learning: How to build an efficient big data pipeline

Generally speaking, data prepossessing consists of two steps: Data engineering and feature engineering.

- Data engineering is the process of converting raw data into prepared data, which can be used by the ML model.
- Feature engineering creates the features expected by the model.

When we deal with a small number of data points, building a pipeline is usually straightforward. But that’s almost never the case with Deep Learning. Here we play with very very large datasets (I’m talking about GBs or even TBs in some cases). And manipulating those is definitely not a piece of cake. But dealing with difficult software challenges is what this article series is all about. 

## ETL: Extract, Transform, Load
In the wonderful world of databases, there is this notion called ETL. As you can see in the headline ETL is an acronym of Extract, Transform, Load. These are the 3 building blocks of most data pipelines.

- Extraction involves the process of extracting the data from multiple homogeneous or heterogeneous sources.
- Transformation refers to data cleansing and manipulation in order to convert them into a proper format.
- Loading is the injection of the transformed data into the memory of the processing units that will handle the training (whether this is CPUs, GPUs or even TPUs)

When we combine these 3 steps, we get the notorious data pipeline. However, there is a caveat here. It’s not enough to build the sequence of necessary steps. It’s equally important to make them fast. Speed and performance are key parts of building a data pipeline.

Imagine that each training epoch of our model, it’s taking 10 minutes to complete. What happens if ETL of the segment of the required data can’t be finished in less than 15 minutes? The training will remain idle for 5 minutes. And you may say fine, it’s offline, who cares? But when the model goes online, what happens if the processing of a datapoint takes 2 minutes? The user will have to wait for 2 minutes plus the inference time. Let me tell you that 2 minutes in browser response time is simply unacceptable for good user experience.

let’s see how things work in practice. Before we dive into the details, let’s see some of the problems we want to address when constructing an input pipeline. Because it’s not just speed (if only it was). We also care about throughput, latency, ease of implementation and maintenance. In more details, we might need to solve problems such as:

- Data might not fit into memory.
- Data might not even fit into the local storage.
- Data might come from multiple sources.
- Utilize hardware as efficiently as possible both in terms of resources and idle time.
- Make processing fast so it can keep up with the accelerator’s speed.
- The result of the pipeline should be deterministic (or not).
- Being able to define our own specific transformations.
- Being able to visualize the process.

## Data Reading
Data reading or extracting is the step in which we get the data from the data source and convert them from the format they are stored into our desired one. You may wonder where the difficulty is. We can just run a “pandas.read_csv()”. Well not quite. In the research phase of machine learning, we are used to having all the data in our local disk and playing with them. But in a production environment, the data might be stored in a database (like MySQL or MongoDB), or in an object storage cloud service (like AWS S3 or Google cloud storage), or in a data warehouse (like Amazon Redshift or Google BigQuery) or of course in a simple storage unit locally. And each storage option has its own set of rules on how to extract and parse data.

## Loading from multiple sources
That’s why we need to be able to consolidate and combine all these different sources into a single entity that can be passed into the next step of the pipeline. And of course, each source has a specific format to store them, so we need a way to decode it as well. Here is a boilerplate code that uses tf.data (the standard data manipulation library of tf)

```python
files = tf.data.Dataset.list_files(file_pattern)
dataset = tf.data.TFRecordDataset(files)

# or load from amazon s3

filenames = ["s3://bucketname/path/to/file1.tfrecord",
             "s3://bucketname/path/to/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)

```

The data should be in a format and in a source supported by tensorflow. What we want to do here is to manually download, unzip and convert the raw images into tf.Records or another compatible with Tensorflow format. Luckily we can use the tensorflow datasets library (tf.tdfs) for that, which wraps all this functionality and returns a ready to use tf.Dataset.

# Parallel data extraction
In cases where data are stored remotely, loading them can become a bottleneck in the pipeline since it can significantly increase the latency and lower our throughput. The time that takes for the data to be extracted from the source and travel into our system is an important factor to take into consideration. What can we do to tackle this bottleneck?

The answer that comes first to mind is parallelization. Since we haven’t touched upon parallel processing so far in the series, let’s give a quick definition:
```
Parallel processing is a type of computation in which many calculations or the execution of processes are carried out simultaneously.
```
Modern computing systems include multiple CPU cores, so why not take advantage of all of them (remember efficient hardware utilization?). For example, my laptop has 4 cores. Wouldn’t it be reasonable to assign each core to a different data point and load four of them at the same time? Luckily it is as easy as this:

```python
tf.data.Dataset.list_files(file_pattern)
tf.data.interleave(TFRecordDataset,num_calls=4)
```

The “interleave()” function will load many data points concurrently and interleave the results so we don’t have to wait for each one of them to be loaded.
Because parallel processing is a highly complicated topic, let’s talk about it more extensively later in the series. For now, all you need to remember is that we can extract many data at the same time utilizing our system resources efficiently.

## Data Processing
Well well where are we? We loaded our data in parallel from all the sources and now we are ready to apply some transformations into them. In this step, we are running the most computationally intense functions such as image manipulation, data decoding and literally anything you can code (or find a ready solution for). In the image segmentation example that we are using, this will simply be resizing our images, flip a portion of them to introduce variance in our dataset, and finally normalize them. Although let me introduce another new concept before that, starting from functional programming

```
Functional programming is a programming paradigm in which we build software by stacking pure functions, avoiding to share state between them and using immutable data. In functional programming, the logic and data flows through functions, inspired by the mathematics
```

Let’s give an example:
```python
df.rename(columns={"titanic_survivors": "survivors"})\
  .query("survivors_age > 14 and survivors_gender == "female"")\
  .sort_values("survivors_age", ascending=False)\
```
Notice how we chained the methods so each function is called after the previous one. Also notice that we don’t share information between functions and that the original dataset flows throughout the chain. That way we don’t need to have for-loops, or reassign variables over and over or create a new dataframe every time we apply a new transformation. Plus, it is so freaking easy to parallelize this code. Remember the trick above in the interleave function where we add a num_calls argument? Well, the reason we are able to do that so effortlessly is functional programming.

Functional programming supports many different functions such as “filter()”, “sort()” and more. But the most important one is called “map()”. With “map()” we can apply whatever (almost) function we may think of.

```python
@staticmethod
def preprocess_data(dataset, batch_size, buffer_size,image_size):
    """ Preprocess and splits into training and test"""
    train = dataset['train'].map(lambda image: DataLoader._preprocess_train(image,image_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train.shuffle(buffer_size)

    test = dataset['test'].map(lambda image: DataLoader._preprocess_test(image, image_size))
    test_dataset = test.shuffle(buffer_size)

    return train_dataset, test_dataset
```
As you can see, we have two different pipelines. One for the train dataset and one for the test dataset. See how we first apply the “map()” function and sequentially the “shuffle()”. The map function will apply the “_preprocess_train“ in every single datapoint. And once the preprocessing finished it will shuffle the dataset. That’s functional programming babe. No share of objects between functions, no mutability of objects, no unnecessary side effects. We just declare our desired functionality and that’s it. Again I’m sure some of you don’t understand all the terms and that’s fine. The key thing is to understand the high-level concept.

Notice also the “num_parallel_calls” argument. Yup, we will run the function in parallel. And thanks to TensorFlow’s built-in autotuning, we don’t even have to worry about setting the number of calls. It will figure it out by itself, based. How cool is that?

For completion’s sake, we also need to mention that besides “map()”, tf.data also supports many other useful functions such as:

- filter() : filtering dataset based on condition
- shuffle(): randomly shuffle dataset
- skip(): remove elements from the pipeline
- concatenate(): combines 2 or more datasets
- cardinality(): returns the number of elements in the dataset

And of course it contains some extremely powerful functions like “batch()”, “prefetch()”, “cache()”, “reduce()”, which are the topic of the next in line article. The original plan was to have them here as well but it will surely compromise the readability of this article and it will definitely give you a headache. So stay tuned. You can also subscribe to our newsletter to make sure that you won’t miss it.