# Nudging: Inference-time Alignment of LLMs via Guided Decoding

This is the code for Nudging: Inference-time Alignment of LLMs via Guided Decoding.
 * [Project Page](https://fywalter.github.io/nudging/)
 * [Demo](https://huggingface.co/spaces/fywalter/nudging_align)
 * [Paper](https://arxiv.org/abs/2410.09300)

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
## Datasets
We use the following 13 datasets for our experiments: 
[gsm8k](https://huggingface.co/datasets/gsm8k), 
[svamp](https://huggingface.co/datasets/gsm8k), 
[multiarith](https://huggingface.co/datasets/ChilleD/MultiArith), 
[mmlu](https://huggingface.co/datasets/cais/mmlu), 
[arc_challenge](https://huggingface.co/datasets/allenai/ai2_arc), 
[strategyqa](https://huggingface.co/datasets/ChilleD/StrategyQA), 
[csqa](https://huggingface.co/datasets/tau/commonsense_qa), 
[sports](https://huggingface.co/datasets/hails/bigbench/viewer/sports_understanding_zero_shot), 
[date](https://huggingface.co/datasets/hails/bigbench/viewer/date_understanding_zero_shot), 
[coin_flip](https://huggingface.co/datasets/skrishna/coin_flip), 
[last_letter_concat](https://huggingface.co/datasets/ChilleD/LastLetterConcat), 
[justeval](https://huggingface.co/datasets/re-align/just-eval-instruct), 
[justeval_safe](https://huggingface.co/datasets/re-align/just-eval-instruct/viewer/judgements_safety).


The download of datasets is handle automatically by the code from the [Huggingface datasets](https://huggingface.co/docs/datasets/) library.

## Models
Our experiments are based on three model families: 
[Llama-2](https://huggingface.co/meta-llama/Llama-2-7b),
[Gemma-2](https://huggingface.co/google/gemma-2-2b),
[OLMo](https://huggingface.co/allenai/OLMo-1B-0724-hf). One can use any instruct model from all three families to nudge any base model from any of the three families. For example, using Llama-2-7b-chat to nudge Gemma-2-27b.

To run nudging on your own model family, one need to add the instruct template to the `apply_instruct_template()` function in the `utils.py` file. One can check the code for the three model families to see how to add a new model family.


## Running the code
To run the experiment for a dataset, say GSM8K, run the following commands. We use concurrent.futures to parallelize the inference process. The `num_threads` argument specifies the number of threads to use for parallelization. The `dataset_name` should be one of: `gsm8k`, `svamp`, `multiarith`, `mmlu`, `arc_challenge`, `strategyqa`, `csqa`, `sports`, `date`, `coin_flip`, `last_letter_concat`, `justeval`, `justeval_safe`.

For base model only with base model `base_model_path` hosted at `base_model_host_url`:
```bash
python run_api.py --dataset_name gsm8k \
    --num_sample 100 \
    --exp base_only \
    --base_model base_model_path \
    --base_host base_model_host_url \
    --rerun --num_threads 20
```

For nudging model only with nudging model `nudging_model_path` hosted at `nudging_model_host_url`:
```bash
python run_api.py --dataset_name gsm8k \
    --num_sample 100 \
    --exp nudging_only \
    --nudging_model nudging_model_path \
    --nudging_host nudging_model_host_url \
    --rerun --num_threads 20
```
For nudging with top prob threshold $\gamma=0.4$ with base model `base_model_path` hosted at `base_model_host_url` and nudging model `nudging_model_path` hosted at `nudging_model_host_url`:
```bash
python run_api.py --dataset_name gsm8k \
    --num_sample 100 \
    --exp nudging \
    --base_model base_model_path \
    --base_host base_model_host_url \
    --nudging_model nudging_model_path \
    --nudging_host nudging_model_host_url \
    --rerun --num_threads 20 \
    --top_prob_thres 0.4
```

For baselines (proxy_tuning) with base model `base_model_path` hosted at `base_model_host_url`, proxy chat model `proxy_chat_path`, proxy base model `proxy_base_path` hosted at `proxy_base_host_url`:
```bash
python run_api.py --dataset_name gsm8k \
    --num_sample 100 \
    --exp baseline \
    --baseline_method proxy_tuning \
    --base_model base_model_path \
    --base_host base_model_host_url \
    --proxy_chat_model proxy_chat_path \
    --proxy_base_model proxy_base_path \
    --proxy_base_host proxy_base_host_url \
    --proxy_nudging_host proxy_nudging_host_url \
    --rerun --num_threads 20
```

## Citing
If you find our work useful, please consider citing:
```BibTeX
@misc{fei2025nudginginferencetimealignmentllms,
      title={Nudging: Inference-time Alignment of LLMs via Guided Decoding}, 
      author={Yu Fei and Yasaman Razeghi and Sameer Singh},
      year={2025},
      eprint={2410.09300},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.09300}, 
}
```
