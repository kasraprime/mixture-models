# mixture-models
Implementing mixture models on toy datasets such as MNIST
batch size is 32, and when we load data the shape is 2\*32. 2 is because we have images and labels, so data[1] is a list of labels. data[0] is a list of length 32 in which every element is tensor of size 28\*28 (which is the size of images in MNIST)

In order to run the program type the following shell command:
```python bmm.py --experiment_name K10 --K_mixture 10```

Note that all the arguments are optional. Default value for K is 10, and number of epochs is 10.
