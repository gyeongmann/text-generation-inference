![BlockNote image](https://camo.githubusercontent.com/816a23feff65e234bf4393d453a7b3c8d24092f8630c2ed131b8b500ddb16dc1/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f4e617273696c2f7467695f6173736574732f7265736f6c76652f6d61696e2f7468756d626e61696c2e706e67)

# Text Generation Inference

![BlockNote image](https://camo.githubusercontent.com/afaad1cd3eb9f616fa5f8764f047603f53587984b922c9198ec1ab2895d9aae1/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f68756767696e67666163652f746578742d67656e65726174696f6e2d696e666572656e63653f7374796c653d736f6369616c)![BlockNote image](https://camo.githubusercontent.com/f0fe98e6afc151ec5523b758b023431bacb2de779486f7488953b503f5643b88/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4150492d537761676765722d696e666f726d6174696f6e616c)

A Rust, Python and gRPC server for text generation inference. Used in production at [HuggingFace](https://huggingface.co/) to power Hugging Chat, the Inference API and Inference Endpoint.

## Table of contents

*   [Get Started](https://github.com/gyeongmann/text-generation-inference#get-started)

    *   [API Documentation](https://github.com/gyeongmann/text-generation-inference#api-documentation)
    *   [Using a private or gated model](https://github.com/gyeongmann/text-generation-inference#using-a-private-or-gated-model)
    *   [A note on Shared Memory](https://github.com/gyeongmann/text-generation-inference#a-note-on-shared-memory-shm)
    *   [Distributed Tracing](https://github.com/gyeongmann/text-generation-inference#distributed-tracing)
    *   [Local Install](https://github.com/gyeongmann/text-generation-inference#local-install)
    *   [CUDA Kernels](https://github.com/gyeongmann/text-generation-inference#cuda-kernels)

*   [Optimized architectures](https://github.com/gyeongmann/text-generation-inference#optimized-architectures)

*   [Run Mistral](https://github.com/gyeongmann/text-generation-inference#run-a-model)

    *   [Run](https://github.com/gyeongmann/text-generation-inference#run)
    *   [Quantization](https://github.com/gyeongmann/text-generation-inference#quantization)

*   [Develop](https://github.com/gyeongmann/text-generation-inference#develop)

*   [Testing](https://github.com/gyeongmann/text-generation-inference#testing)

Text Generation Inference (TGI) is a toolkit for deploying and serving Large Language Models (LLMs). TGI enables high-performance text generation for the most popular open-source LLMs, including Llama, Falcon, StarCoder, BLOOM, GPT-NeoX, and [more](https://huggingface.co/docs/text-generation-inference/supported_models). TGI implements many features, such as:

*   Simple launcher to serve most popular LLMs

*   Production ready (distributed tracing with Open Telemetry, Prometheus metrics)

*   Tensor Parallelism for faster inference on multiple GPUs

*   Token streaming using Server-Sent Events (SSE)

*   Continuous batching of incoming requests for increased total throughput

*   Optimized transformers code for inference using [Flash Attention](https://github.com/HazyResearch/flash-attention) and [Paged Attention](https://github.com/vllm-project/vllm) on the most popular architectures

*   Quantization with :

    *   [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
    *   [GPT-Q](https://arxiv.org/abs/2210.17323)
    *   [EETQ](https://github.com/NetEase-FuXi/EETQ)
    *   [AWQ](https://github.com/casper-hansen/AutoAWQ)

*   [Safetensors](https://github.com/huggingface/safetensors) weight loading

*   Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)

*   Logits warper (temperature scaling, top-p, top-k, repetition penalty, more details see [transformers.LogitsProcessor](https://huggingface.co/docs/transformers/internal/generation_utils#transformers.LogitsProcessor))

*   Stop sequences

*   Log probabilities

*   [Speculation](https://huggingface.co/docs/text-generation-inference/conceptual/speculation) ~2x latency

*   [Guidance/JSON](https://huggingface.co/docs/text-generation-inference/conceptual/guidance). Specify output format to speed up inference and make sure the output is valid according to some specs..

*   Custom Prompt Generation: Easily generate text by providing custom prompts to guide the model's output

*   Fine-tuning Support: Utilize fine-tuned models for specific tasks to achieve higher accuracy and performance

### Hardware support

*   [Nvidia](https://github.com/huggingface/text-generation-inference/pkgs/container/text-generation-inference)
*   [AMD](https://github.com/huggingface/text-generation-inference/pkgs/container/text-generation-inference) (-rocm)
*   [Inferentia](https://github.com/huggingface/optimum-neuron/tree/main/text-generation-inference)
*   [Intel GPU](https://github.com/huggingface/text-generation-inference/pull/1475)
*   [Gaudi](https://github.com/huggingface/tgi-gaudi)
*   [Google TPU](https://huggingface.co/docs/optimum-tpu/howto/serving)

## Get Started

### Docker

For a detailed starting guide, please see the [Quick Tour](https://huggingface.co/docs/text-generation-inference/quicktour). The easiest way of getting started is using the official Docker container:

```javascript
model=HuggingFaceH4/zephyr-7b-beta
# share a volume with the Docker container to avoid downloading weights every run
volume=$PWD/data

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:2.1.0 --model-id $model
```

And then you can make requests like

```javascript
curl 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```

**Note:** To use NVIDIA GPUs, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). We also recommend using NVIDIA drivers with CUDA version 12.2 or higher. For running the Docker container on a machine with no GPUs or CUDA support, it is enough to remove the `--gpus all` flag and add `--disable-custom-kernels`, please note CPU is not the intended platform for this project, so performance might be subpar.

**Note:** TGI supports AMD Instinct MI210 and MI250 GPUs. Details can be found in the [Supported Hardware documentation](https://huggingface.co/docs/text-generation-inference/supported_models#supported-hardware). To use AMD GPUs, please use `docker run --device /dev/kfd --device /dev/dri --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.1.0-rocm --model-id $model` instead of the command above.

To see all options to serve your models (in the [code](https://github.com/huggingface/text-generation-inference/blob/main/launcher/src/main.rs) or in the cli):

```javascript
text-generation-launcher --help
```

### API documentation

You can consult the OpenAPI documentation of the `text-generation-inference` REST API using the `/docs` route. The Swagger UI is also available at: <https://huggingface.github.io/text-generation-inference>.

### Using a private or gated model

You have the option to utilize the `HF_TOKEN` environment variable for configuring the token employed by `text-generation-inference`. This allows you to gain access to protected resources.

For example, if you want to serve the gated Llama V2 model variants:

1.  Go to <https://huggingface.co/settings/tokens>
2.  Copy your cli READ token
3.  Export `HF_TOKEN=<your cli READ token>`

or with Docker:

```javascript
model=meta-llama/Llama-2-7b-chat-hf
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
token=<your cli READ token>

docker run --gpus all --shm-size 1g -e HF_TOKEN=$token -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id $model
```

### A note on Shared Memory (shm)

`NCCL` is a communication framework used by `PyTorch` to do distributed training/inference. `text-generation-inference` make use of `NCCL` to enable Tensor Parallelism to dramatically speed up inference for large language models.

In order to share data between the different devices of a `NCCL` group, `NCCL` might fall back to using the host memory if peer-to-peer using NVLink or PCI is not possible.

To allow the container to use 1G of Shared Memory and support SHM sharing, we add `--shm-size 1g` on the above command.

If you are running `text-generation-inference` inside `Kubernetes`. You can also add Shared Memory to the container by creating a volume with:

```javascript
- name: shm
  emptyDir:
   medium: Memory
   sizeLimit: 1Gi
```

and mounting it to `/dev/shm`.

Finally, you can also disable SHM sharing by using the `NCCL_SHM_DISABLE=1` environment variable. However, note that this will impact performance.

### Distributed Tracing

`text-generation-inference` is instrumented with distributed tracing using OpenTelemetry. You can use this feature by setting the address to an OTLP collector with the `--otlp-endpoint` argument. The default service name can be overridden with the `--otlp-service-name` argument

### Architecture

![BlockNote image](https://camo.githubusercontent.com/865b15b83e926b08c3ce2ad186519ad520bce2241b89095edcf7416d2be91aba/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f68756767696e67666163652f646f63756d656e746174696f6e2d696d616765732f7265736f6c76652f6d61696e2f5447492e706e67)

### Local install

You can also opt to install `text-generation-inference` locally.

First [install Rust](https://rustup.rs/) and create a Python virtual environment with at least Python 3.9, e.g. using `conda`:

```javascript
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

conda create -n text-generation-inference python=3.11
conda activate text-generation-inference
```

You may also need to install Protoc.

On Linux:

```javascript
PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP
```

On MacOS, using Homebrew:

```javascript
brew install protobuf
```

Then run:

```javascript
BUILD_EXTENSIONS=True make install # Install repository and HF/transformer fork with CUDA kernels
text-generation-launcher --model-id mistralai/Mistral-7B-Instruct-v0.2
```

**Note:** on some machines, you may also need the OpenSSL libraries and gcc. On Linux machines, run:

```javascript
sudo apt-get install libssl-dev gcc -y
```

## Optimized architectures

TGI works out of the box to serve optimized models for all modern models. They can be found in [this list](https://huggingface.co/docs/text-generation-inference/supported_models).

Other architectures are supported on a best-effort basis using:

`AutoModelForCausalLM.from_pretrained(<model>, device_map="auto")`

or

`AutoModelForSeq2SeqLM.from_pretrained(<model>, device_map="auto")`

## Run locally

### Run

```javascript
text-generation-launcher --model-id mistralai/Mistral-7B-Instruct-v0.2
```

### Quantization

You can also quantize the weights with bitsandbytes to reduce the VRAM requirement:

```javascript
text-generation-launcher --model-id mistralai/Mistral-7B-Instruct-v0.2 --quantize
```

4bit quantization is available using the [NF4 and FP4 data types from bitsandbytes](https://arxiv.org/pdf/2305.14314.pdf). It can be enabled by providing `--quantize bitsandbytes-nf4` or `--quantize bitsandbytes-fp4` as a command line argument to `text-generation-launcher`.

## Develop

```javascript
make server-dev
make router-dev
```

## Testing

```javascript
# python
make python-server-tests
make python-client-tests
# or both server and client tests
make python-tests
# rust cargo tests
make rust-tests
# integration tests
make integration-tests
```



