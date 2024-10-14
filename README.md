# Nudging: Inference-time Alignment via Model Collaboration

This is the code for Nudging: Inference-time Alignment via Model Collaboration.
 * [Project Page](https://fywalter.github.io/nudging/)  (coming soon)
 * [Paper](https://fywalter.github.io/nudging/) (coming soon)

Currently we provide an API-based implementation of nudging that uses [vllm](https://github.com/vllm-project/vllm) to host the models and provide API access.
With slight modifications the code could also work with any API service providers that provide openai-compatible interfaces like [Together AI](https://www.together.ai/) and [Fireworks AI](https://fireworks.ai/).
We are working on a [speculative docoding](https://arxiv.org/abs/2211.17192) style fast local implementation of nudging that will be released soon.
 
## Setup
The code can be run under any environment with Python 3.11 and above.
(It may run with lower versions, but we have not tested it).

We recommend using [conda](https://anaconda.org/anaconda/conda) to set up the environment:

    conda create --name nudging python=3.11

Next, install the vllm package that would also install the necessary dependencies:

    pip install vllm==0.6.2

## Hosting models using vllm
We use vllm to host the models and provide [API access](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html). For example, one can host a model using the following command:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model model_path \
    --tensor-parallel-size num_gpus \
    --max-logprobs 100 \
    --port 8000 \
    --max_model_len 2048 \
```
Then one can use the openai-compatible API to interact with the model:

```python
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
completion = client.completions.create(model=model_path,
                                      prompt="San Francisco is a")
print("Completion result:", completion)
```

Please refer to the [vllm documentation](https://docs.vllm.ai/en/v0.2.7/getting_started/quickstart.html) for more details including setting host and port numbers.

## Running the code
To run the experiment for a dataset, say GSM8K, run the following command:

For base model only:
```bash
python run_api.py --dataset_name gsm8k --num_sample 100 --exp base_only --base_model base_model_path --base_host base_model_host_url --rerun --num_threads 20
```

For nudging model only
```bash
python run_api.py --dataset_name gsm8k --num_sample 100 --exp nudging_only --nudging_model nudging_model_path --nudging_host nudging_model_host_url --rerun --num_threads 20
```
For Nudging
```bash
python run_api.py --dataset_name gsm8k --num_sample 100 --exp nudging --base_model base_model_path --base_host base_model_host_url  --nudging_model nudging_model_path --nudging_host nudging_model_host_url --rerun --num_threads 20 --top_prob_thres 0.4
```

For Baselines (proxy_tuning)
```bash
python run_api.py --dataset_name gsm8k --num_sample 100 --exp baseline --baseline_method proxy_tuning --base_model base_model_path --base_host base_model_host_url --proxy_chat_model proxy_chat_path --proxy_base_model proxy_base_path --proxy_base_host proxy_base_host_url --proxy_nudging_host proxy_nudging_host_url --rerun --num_threads 20 --top_prob_thres 0.4 
```

We use concurrent.futures to parallelize the inference process. The `num_threads` argument specifies the number of threads to use for parallelization.

## Citing
If you find our work useful, please consider citing:
```BibTeX
@article{park2021nerfies
  author    = {Park, Keunhong 
               and Sinha, Utkarsh 
               and Barron, Jonathan T. 
               and Bouaziz, Sofien 
               and Goldman, Dan B 
               and Seitz, Steven M. 
               and Martin-Brualla, Ricardo},
  title     = {Nerfies: Deformable Neural Radiance Fields},
  journal   = {ICCV},
  year      = {2021},
}
```