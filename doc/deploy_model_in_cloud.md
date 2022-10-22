# How to train a deep learning model in the cloud
When it comes to training a large Deep Learning model, there are many obstacles that we need to overcome. First, we need to acquire lots and lots of data. Training relies on other steps such as fine-tuning our architecture and hyperparameters, the accuracy, and building some sort of logging and visualization. Obviously, we should keep track of progress and many many more. However, one of the most frequent struggles is the lack of hardware resources. Most of us don't own an NVIDIA Titan RTX or we don't have access to a cluster of PCs, so we are forced to wait hours and hours on each training iteration to evaluate our model.

I also understand that we are tempted to buy a high-end GPU. And to be honest with you I have already tried it. Nevertheless, there is a much easier way to train our models and I'm sure you're aware of that. It’s called Cloud. Cloud providers such as Google cloud, Amazon Web Services and Microsoft Azure are excellent examples of low cost, high-end infrastructure. Cloud is usually targeted to machine learning applications.

In today's article, we will take our previously built Unet model that performs image segmentation, deploy it in the Google cloud and run a full training job there. If you remember from our last article of the series, we developed a custom training loop in Tensorflow. The goal is to take that code almost unchanged and run it in a Google cloud instance.

What is an instance you may think? We will get there,don't worry.

Regarding the structure of the article, I think I should take it step by step and explain important topics and details while I'm outlining all the necessary instructions to reach our end goal.

## Cloud computing
I'm sure you know what cloud computing is but for consistency reasons, let's give a high-level definition.

```
Cloud computing is the on-demand delivery of IT resources via the internet. Instead of buying and maintaining physical servers and data centers, we can access infrastructure such as computer power and storage from cloud providers.
```

About 90% of all the companies in the world use some form of cloud service today. I hope that's enough to convince you about the power of the Cloud. And the most astonishing thing is that you literally have access to a huge variety of different systems and applications that would be unimaginable to maintain on your own.

For our use case, we're gonna need only one of all the services called Compute Engine. Compute Engine let us use virtual machine instances hosted in the Google servers and maintained by them.

```
A virtual machine (VM) is an emulation of a computer system. Virtual machines are based on computer architectures and provide functionality of a physical computer. Their implementations may involve specialized hardware, software, or a combination of them.
```

So, in essence we borrow a small PC in Google servers, install whatever operating system and software we may want (aka build a Virtual Machine) and do remotely whatever we might do in our own local laptop. It’s that simple.

OK now that we know the basics let's proceed with a hands-on approach. If you haven't created a Google account by now, feel free to do that. All you have to do is go here, register an account (a credit card is required for security reasons but it won't be charged for at least a year or if you surpass the free quotas) and enjoy a full 300$ free credit (at least at the time of writing this post).

One thing I forgot to mention is that in most cases the cloud follows a pay as you go pricing model, meaning that we get charged depending on how many resources we use.

## Creating a VM instance
Once you have a Google cloud account, it's time to create a brand new project to host our application by clicking on the top left “New project” and naming it whatever you like.


When the project is initialized, we can navigate to: Compute Engine > VM instances from the sidebar on the left and create a new VM.

As you can see you can customize the instance in any fashion you like. You can choose your CPU, your RAM, you can add a GPU and you can make it as high performant as you need. I'm gonna keep things simple here and select a standard CPU with 3.75 GB of memory and an Nvidia Tesla K80. And of course you can pick your own OS. I will use Ubuntu’s 20.04 minimal image with a 10GB disk size.

You can also select some other things like deploying a container to the instance or allow traffic from the Internet, but let's not deal with them right now.

## Connecting to the VM instance
OK great, we have our instance up and running. It's time to start using it. We can connect to it using standard SSH by clicking the SSH button. This will open a new browser window and give us access to the Google machine.

As you can see we literally have access through a terminal to a remote machine in Google servers and as you can imagine we can install what we want as we would normally do in our laptop.

The next step is to transfer our data and code into the remote machine from our laptop. Google cloud uses a specialized command called “gcloud” that handles many things such as authentication (using SHH under the hood). A similar command exists in almost all cloud providers so the following steps have pretty much the same logic.

```
SHH is a protocol that uses encryption to secure the connection between a client and a server and allow us to safely connect to a remote machine.
```

