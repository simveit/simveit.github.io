# Multi chip performance in JAX

The larger the models we use get the more it becomes necessary to be able to perform training of machine learning models over multiple chips.
In this blog post we will explain how to efficently use Googles TPU. TPUs are especially convenient as they are designed especially for machine learning and easily deployable on Google cloud. For an introduction on how to deploy your own TPU with Google cloud [see this excellent documentation](https://github.com/ayaka14732/tpu-starter?tab=readme-ov-file#2-introduction-to-tpu).

In this tutorial we will take a simple layerwise matrix multiplication of activations with weights as our running example. The workload may be visualized like this:

![Layer-wise Matrix Multiplication](assets/LayerMatMul.png)