---
title: "High performance Inference on TPUs using Maxtext"
categories:
  - Cloud computing
  - LLM
tags:
  - Googlecloud
  - TPU
---

![Picture](/assets/maxtext_inference/cloud.png)

### High Throughput Inference on TPUs
Performing training and inference of LLMs on TPUs with the "standard toolbox" (i.e. huggingface transformers) of data scientist is not straightforward.
According to their [github repo](https://github.com/google/maxtext) *MaxText is a high performance, highly scalable, open-source LLM written in pure Python/Jax and targeting Google Cloud TPUs and GPUs for training and inference. MaxText achieves high MFUs and scales from single host to very large clusters while staying simple and "optimization-free" thanks to the power of Jax and the XLA compiler.*
In this blogpost I will show how to perform a simple text completion example using Maxtext and a model from the huggingface.
There is no example on how to do this in the Maxtext repo so I thought it might be beneficial to write a short blog post on this.
### Download model from the hub
We will use [Hermes 2 Pro - Mistral 7B](https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B) as a model.
Clone the model with `git clone https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B`. 
### Prepare Model bucket
We want to upload our model in a next step to a google cloud bucket. We will use this later to access the model files in an easy way from our TPU VM. 
To learn how to create a cloud bucket see [this link](https://cloud.google.com/storage/docs/creating-buckets). After creating the bucket we can simply create a folder inside this bucket and transfer per drag and drop the content of our local folder into this folder.
### Creating a TPU VM and setting it up
There is an excellent guide on how to create a TPU VM [on this github repo](https://github.com/ayaka14732/tpu-starter) if you are not familar with this. 
After setting up the TPU VM we can SSH into it like described in the guide above and use a convenient IDE like `VSCode` for development. In my example i used a `TPU` of the type `v4-8` which stands for *fourth generation* and has *8/2 = 4 chips*.
The Setup commands are as follows:

This will do the basic setup and create a virtual environment for python:
```
sudo apt-get update -y -qq
sudo apt-get upgrade -y -qq
sudo apt-get install -y -qq golang neofetch zsh byobu
```
```
sudo apt-get install -y -qq software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y -qq python3.10-full python3.10-dev
```
```
python3.11 -m venv ~/venv
```
To avoid interactive mode in needrestart for ubuntu we use:
```
sed "/#\$nrconf{restart} = 'i';/s/.*/\$nrconf{restart} = 'a';/" /etc/needrestart/needrestart.conf
```
We then clone the Maxtext repo:
```
git clone https://github.com/google/maxtext.git
```
Cd into it and execute the setup bash file
```
cd maxtext
bash setup.sh
```
### Adjusting the convert file
To run a model using Maxtext we need it to be in the Maxtext compatible format.
Conversion from huggingface format is not the standard so we need to adjust the default conversion script.
Thanks to user @Froggy111 from the JAX LLM discord channel for kindly providing this script to me :).
You can copy the script from [this gist](https://gist.github.com/simveit/53f59b682c54172620726aa5609f6cb6). Be aware that you might adjust settings like I did with the hermes entry in the config (here the vocab size was different to the original mistral).
We will take this file and use it to replace `MaxText/llama_or_mistral_ckpt.py` inside the maxtext repo. We should also adjust the vocab_size in `MaxText/configs/base.yml` and `MaxText/configs/models/mistral-7b.yml` if needed.

### Running inference
Now we are ready to run inference.
We modify the content of to be:
```
#!/bin/bash

set -ex
idx="0"

export MODEL_PATH=gs://YOUR_BUCKET_PATH
export PROMPT="<|im_start|>system\nYou are the Pokemon trainer Ash Ketchum.<|im_end|>\n<|im_start|>user\nHello, who are you?<|im_end|>\n<|im_start|>assistant\n"

# Download checkpoint, convert it to MaxText, and run inference
pip3 install torch
pip3 install jax-smi
pip3 install safetensors
## The next 2 commands only need to be done ONCE!
gsutil -m cp -r ${MODEL_PATH} /tmp
python3 MaxText/llama_or_mistral_ckpt.py --base-model-path /tmp/Hermes-2-Pro-Mistral-7B --model-size hermes --maxtext-model-path ${MODEL_PATH}/test/${idx}/decode-ckpt-maxtext/
python3 MaxText/decode.py MaxText/configs/base.yml load_parameters_path=${MODEL_PATH}/test/${idx}/decode-ckpt-maxtext/0/items run_name=runner_direct_${idx} per_device_batch_size=1 model_name='mistral-7b' tokenizer_path=${MODEL_PATH}/tokenizer.model prompt="${PROMPT}" max_target_length=92
```
Please be aware of the fact that you will need to adjust the Model path to point to your model folder inside the bucket created above.
After running this bash script we recieve
```
Input `<|im_start|>system\nYou are the Pokemon trainer Ash Ketchum.<|im_end|>\n<|im_start|>user\nHello, who are you?<|im_end|>\n<|im_start|>assistant\n` -> `Hello! I am Ash Ketchum, a Pokemon trainer on a journey to become the ultimate master of all the Pokemon out there`
```
### What's next?
After we have learned how to convert a huggingface model to a Maxtext compatible format and performing inference the next step would be to train a model. Maxtext has very high Model flops utilization and can scale to extremly large number of chips (reported on the repo is a training job with **51K** chips!)

### Conclusion
I hope you liked this blog post. Feel free to share it with friends which have access to TPUs or are interested in large scale training jobs. Maxtext might be a good fit for them.
The experiments in this blog were supported by [Google's TPU Research Cloud program](https://sites.research.google/trc/about/).