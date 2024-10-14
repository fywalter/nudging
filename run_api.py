
from tqdm import tqdm
import os
import argparse
import json
import concurrent.futures   # for parallel processing of the samples

from openai import OpenAI

from utils import apply_instruct_template, completion_with_nudging, completion_with_baseline
from dataset_utils import extract_ans, parse_pred_ans, get_dataset, PROMPTS

def exp_nudging(
    client_base: OpenAI,
    client_nudging: OpenAI,
    dataset_name: str,
    num_samples: int,
    base_model: str,
    nudging_model: str,
    max_token_total: int,
    input_data: list,
    output_data: list,
    base_temperature: float,
    nudging_temperature: float,
    base_top_p: float,
    answer_start_prompt_base: str = None,
    answer_start_prompt_nudging: str = None,
    print_intermediate_output: bool = False,
    rerun: bool = False,
    exp_prefix: str = "",
    exp: str = "nudging",
    completion_token_num = 16,
    completion_token_num_nudging = 16,
    top_prob_thres: float = 0.3,
    num_threads: int = 10,
):
    print("="*20)
    print(f"{exp} experiments")
    print("experiment settings:")
    print(f"dataset_name: {dataset_name}")
    print(f"num_samples: {num_samples}")
    print(f"base_model: {base_model}")
    print(f"nudging_model: {nudging_model}")
    print(f"max_token_total: {max_token_total}")
    print(f"base_temperature: {base_temperature}")
    print(f"base_top_p: {base_top_p}")
    print(f"nudging_temperature: {nudging_temperature}")
    print(f"num_threads: {num_threads}")
    print(f"top probability threshold: {top_prob_thres}")
    print(f"completion_token_num_base: {completion_token_num}")
    print(f"completion_token_num_nudging: {completion_token_num_nudging}")
    print("="*20)

    base_dir = f'./outputs/{dataset_name}'                      # for saving the txt file that contains the questions and answers
    os.makedirs(base_dir, exist_ok=True)

    all_info_base_dir = f'./outputs/{dataset_name}/all_info'    # for saving the json file that contains all the information
    os.makedirs(all_info_base_dir, exist_ok=True)

    base_model_name = base_model.split('/')[-1]
    nudging_model_name = nudging_model.split('/')[-1]

    if len(exp_prefix)>0 and exp_prefix[-1] != '_':
        exp_prefix += '_'
    save_filename = exp_prefix + f'top_prob_{top_prob_thres}_thres_{exp}_{base_model_name}_{nudging_model_name}_{num_samples}_samples.txt'
    
    save_path = os.path.join(base_dir, save_filename)
    save_path_all_info = os.path.join(all_info_base_dir, save_filename.replace('.txt', '.json'))
    all_info_list = [None] * len(input_data)

    def process_nudging_sample(client_base, client_nudging, base_model, nudging_model, system_prompt_base, system_prompt_nudging, question_prompt, answer_start_prompt_base, answer_start_prompt_nudging, max_token_total, base_temperature, base_top_p, nudging_temperature, print_intermediate_output, top_prob_thres, q, a, dataset_name):
        question = q['input']
        context = q['context']
        all_info = completion_with_nudging(
            base_model=base_model,
            client_base=client_base,
            client_nudging=client_nudging,
            nudging_model=nudging_model,
            system_prompt_base=system_prompt_base,
            system_prompt_nudging=system_prompt_nudging,
            question=question,
            context=context,
            question_prompt=question_prompt,
            answer_start_prompt_base=answer_start_prompt_base,
            answer_start_prompt_nudging=answer_start_prompt_nudging,
            completion_token_num=completion_token_num,
            completion_token_num_nudging=completion_token_num_nudging,
            max_token_total=max_token_total,
            base_temperature=base_temperature,
            top_p=base_top_p,
            nudging_temperature=nudging_temperature,
            print_intermediate_output=print_intermediate_output,
            top_prob_thres=top_prob_thres,
        )

        raw_answer = all_info['raw_answer']
        ans_, scores = extract_ans(dataset_name, raw_answer, input=question, question_start=question_prompt, ans_gold=a)

        all_info['extracted_answer'] = ans_ # Currectly we don't process the answer from the model, so the extracted answer is the same as the raw answer. 
        all_info['scores'] = scores         # GPT-4 scores, if any, for the extracted/raw answer.
        all_info["gold_answer"] = a
        all_info['q_prefix'] = question_prompt
        return all_info

    def process_nudging_chunk(progress_bar, client_base, client_nudging, base_model, nudging_model, system_prompt_base, system_prompt_nudging, question_prompt, answer_start_prompt_base, answer_start_prompt_nudging, max_token_total, base_temperature, base_top_p, nudging_temperature, print_intermediate_output, input_data, output_data, dataset_name):
        chunk_results = []
        for q, a in zip(input_data, output_data):
            all_info = process_nudging_sample(
                client_base=client_base,
                client_nudging=client_nudging,
                base_model=base_model,
                nudging_model=nudging_model,
                system_prompt_base=system_prompt_base,
                system_prompt_nudging=system_prompt_nudging,
                question_prompt=question_prompt,
                answer_start_prompt_base=answer_start_prompt_base,
                answer_start_prompt_nudging=answer_start_prompt_nudging,
                max_token_total=max_token_total,
                base_temperature=base_temperature,
                base_top_p=base_top_p,
                nudging_temperature=nudging_temperature,
                print_intermediate_output=print_intermediate_output,
                top_prob_thres=top_prob_thres,
                q=q, a=a, dataset_name=dataset_name
            )
            chunk_results.append(all_info)
            progress_bar.update(1)
        return chunk_results

    if os.path.exists(save_path) and not rerun:
        print(f"Load saved results from {save_path}")
    else:
        # delete file
        if os.path.exists(save_path):
            os.remove(save_path)
        if os.path.exists(save_path_all_info):
            os.remove(save_path_all_info)
        
        system_prompt_base = PROMPTS[dataset_name]['system']
        system_prompt_nudging = PROMPTS[dataset_name]['system_nudging']
        if answer_start_prompt_base is None:
            answer_start_prompt_base = PROMPTS[dataset_name]["answer_start"]
        if answer_start_prompt_nudging is None:
            answer_start_prompt_nudging = PROMPTS[dataset_name]["answer_start"]
        question_prompt = PROMPTS[dataset_name]['question']

        print(f"system_prompt_base: {system_prompt_base}")
        print(f'system_prompt_nudging: {system_prompt_nudging}')
        print(f"question_prompt: {question_prompt}")
        print(f"answer_start_prompt_base: {answer_start_prompt_base}")
        print(f"answer_start_prompt_nudging: {answer_start_prompt_nudging}")

        chunk_size = (len(input_data) + num_threads - 1) // num_threads
        progress_bar = tqdm(total=len(input_data), desc="Processing samples")
        futures = []
        # Process the samples in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(0, len(input_data), chunk_size):
                chunk_input_data = input_data[i:i + chunk_size]
                chunk_output_data = output_data[i:i + chunk_size]
                futures.append((executor.submit(
                    process_nudging_chunk, progress_bar, client_base, client_nudging, base_model, nudging_model, 
                    system_prompt_base, system_prompt_nudging, question_prompt, answer_start_prompt_base, answer_start_prompt_nudging, 
                    max_token_total, base_temperature, base_top_p, nudging_temperature, print_intermediate_output, 
                    chunk_input_data, chunk_output_data, dataset_name), i))

            for future, start_index in futures:
                result = future.result()
                all_info_list[start_index:start_index + len(result)] = result

        progress_bar.close()

        # save all information
        with open(save_path_all_info, 'w') as f:
            json.dump(all_info_list, f)

        # save the answers and scores
        with open(save_path, 'a') as fd:
            for info in all_info_list:
                fd.write('Input_q: %s\nNudging_words:\n%s\nA_model:\n%s\nScores:\n%s\nA:\n%s\n\n' % (info['context'] + info['question'], info['all_nudging_words'], info['extracted_answer'], json.dumps(info['scores'], indent=4), info['gold_answer']))

    return parse_pred_ans(dataset_name, save_path, print_aggregated_metric=True)

def exp_baseline(
    client_base: OpenAI,
    client_proxy_chat: OpenAI,
    client_proxy_base: OpenAI,
    dataset_name: str,
    num_samples: int,
    base_model: str,
    proxy_chat_model: str,
    proxy_base_model: str,
    max_token_total: int,
    baseline_method: str,
    input_data: list,
    output_data: list,
    rerun: bool = False,
    exp_prefix: str = "",
    temperature: float = 0.0,
    num_threads: int = 100,
):
    print("+"*20)
    print("Baseline experiments")
    print("experiment settings:")
    print(f"baseline_method: {baseline_method}")
    print(f"dataset_name: {dataset_name}")
    print(f"num_samples: {num_samples}")
    print(f"base_model: {base_model}")
    print(f"proxy_chat_model: {proxy_chat_model}")
    print(f"proxy_base_model: {proxy_base_model}")
    print(f"max_token_total: {max_token_total}")
    print(f"temperature: {temperature}")
    print(f"num_threads: {num_threads}")
    print("+"*20)
    base_dir = f'./outputs/{dataset_name}'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    base_model_name = base_model.split('/')[-1]
    proxy_chat_model_name = proxy_chat_model.split('/')[-1]
    if len(exp_prefix)>0 and exp_prefix[-1] != '_':
        exp_prefix += '_'
    save_filename = exp_prefix + f'{baseline_method}_{base_model_name}_{proxy_chat_model_name}_{num_samples}_samples.txt'
    save_path = os.path.join(base_dir, save_filename)
    all_info_base_dir = f'./outputs/{dataset_name}/all_info'
    os.makedirs(all_info_base_dir, exist_ok=True)
    save_path_all_info = os.path.join(all_info_base_dir, save_filename.replace('.txt', '.json'))
    all_info_list = [None] * len(input_data)

    def process_sample(client_base, 
                       client_proxy_chat,
                       client_proxy_base,
                       base_model, 
                       proxy_chat_model,
                       proxy_base_model,
                       baseline_method,
                       max_token_total, 
                       instruction_prompt, 
                       q_prefix, 
                       answer_start_prompt,
                       temperature, 
                       q, a, dataset_name):
        """Function for process a single sample"""
        context = q['context']
        question = q['input']

        all_info = completion_with_baseline(
            client_base=client_base,
            client_proxy_chat=client_proxy_chat,
            client_proxy_base=client_proxy_base,
            base_model=base_model,
            proxy_chat_model=proxy_chat_model,
            proxy_base_model=proxy_base_model,
            baseline_method=baseline_method,
            max_token_total=max_token_total,
            instruction_prompt=instruction_prompt,
            q_prefix=q_prefix,
            answer_start_prompt=answer_start_prompt,
            temperature=temperature,
            context=context,
            question=question,
        )
        raw_answer = all_info['raw_answer']
        ans_, scores = extract_ans(dataset_name, raw_answer, input=question, question_start=q_prefix, ans_gold=a)
        all_info['extracted_answer'] = ans_
        all_info['scores'] = scores
        all_info["gold_answer"] = a
        all_info['q_prefix'] = q_prefix

        return all_info
    
    def process_chunk(progress_bar, client_base, client_proxy_chat, client_proxy_base, base_model, proxy_chat_model, proxy_base_model, baseline_method, max_token_total, instruction_prompt, q_prefix, answer_start_prompt, temperature, input_data, output_data, dataset_name):
        chunk_results = []
        for q, a in zip(input_data, output_data):
            all_info = process_sample(client_base=client_base, 
                                        client_proxy_chat=client_proxy_chat,
                                        client_proxy_base=client_proxy_base,
                                        base_model=base_model,
                                        proxy_chat_model=proxy_chat_model,
                                        proxy_base_model=proxy_base_model,
                                        baseline_method=baseline_method,
                                        max_token_total=max_token_total,
                                        instruction_prompt=instruction_prompt,
                                        q_prefix=q_prefix,
                                        answer_start_prompt=answer_start_prompt,
                                        temperature=temperature,
                                        q=q, a=a, dataset_name=dataset_name)
            chunk_results.append(all_info)
            progress_bar.update(1)
        return chunk_results
    
    if os.path.exists(save_path) and not rerun:
        print(f"Load saved results from {save_path}")
    else:
        # delete file
        if os.path.exists(save_path):
            os.remove(save_path)
        if os.path.exists(save_path_all_info):
            os.remove(save_path_all_info)
        instruction_prompt = PROMPTS[dataset_name]['system']
        answer_start_prompt = PROMPTS[dataset_name]["answer_start"]
        q_prefix = PROMPTS[dataset_name]['question']
        print(f"instruction_prompt: {instruction_prompt}")
        print(f"q_prefix: {q_prefix}")
        print(f"answer_start_prompt: {answer_start_prompt}")

        chunk_size = len(input_data) // num_threads
        progress_bar = tqdm(total=len(input_data), desc="Processing samples")
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(0, len(input_data), chunk_size):
                chunk_input_data = input_data[i:i + chunk_size]
                chunk_output_data = output_data[i:i + chunk_size]
                futures.append((executor.submit(
                    process_chunk, progress_bar, client_base, client_proxy_chat, client_proxy_base, base_model, proxy_chat_model, proxy_base_model, baseline_method, max_token_total, instruction_prompt, q_prefix, answer_start_prompt, temperature, chunk_input_data, chunk_output_data, dataset_name), i))

            for future, start_index in futures:
                result = future.result()
                all_info_list[start_index:start_index + len(result)] = result

        # Save results
        with open(save_path, 'a') as fd:
            for info in all_info_list:
                fd.write('Input_q: %s\nA_model:\n%s\nScores:\n%s\nA:\n%s\n\n' % (info['context'] + info['question'], info['extracted_answer'], json.dumps(info['scores'], indent=4), info['gold_answer']))

        with open(save_path_all_info, 'w') as f:
            json.dump(all_info_list, f)
        # Close the progress bar
        progress_bar.close()
    return parse_pred_ans(dataset_name, save_path, print_aggregated_metric=True)

def exp_single_model(
    client: OpenAI,
    dataset_name: str,
    num_samples: int,
    model: str,
    max_token_total: int,
    input_data: list,
    output_data: list,
    rerun: bool = False,
    exp_prefix: str = "",
    temperature: float = 0.0,
    top_p: float = 0.9,
    num_threads: int = 100,
    model_type: str = "nudging",
):
    print("*"*20)
    print(f"{model_type} only experiments")
    print("experiment settings:")
    print(f"dataset_name: {dataset_name}")
    print(f"num_samples: {num_samples}")
    print(f"{model_type}_model: {model}")
    print(f"max_token_total: {max_token_total}")
    print(f"temperature: {temperature}")
    print(f"top_p: {top_p}")
    print(f"num_threads: {num_threads}")
    print("*"*20)

    base_dir = f'./outputs/{dataset_name}'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    model_name = model.split('/')[-1]
    if len(exp_prefix)>0 and exp_prefix[-1] != '_':
        exp_prefix += '_'
    filename = exp_prefix + f'{model_name}_{num_samples}_samples.txt'
    save_path = os.path.join(base_dir, filename)
    all_info_base_dir = f'./outputs/{dataset_name}/all_info'
    os.makedirs(all_info_base_dir, exist_ok=True)
    save_path_all_info = os.path.join(all_info_base_dir, filename.replace('.txt', '.json'))
    all_info_list = [None] * len(input_data)

    def process_sample(client, model, max_token_total, instruction_prompt, q_prefix, answer_start_prompt, temperature, q, a, dataset_name):
        all_info = {}
        prompt_q = q['context'] + q_prefix + q['input']

        # apply the instruction template for the instruct models, for base models the function concatenates the prompts with "\n"
        prompt = apply_instruct_template(model_name=model, system_prompt=instruction_prompt, instruct_prompt=prompt_q, response_prompt=answer_start_prompt) 
        
        response = client.completions.create(
            model=model,
            max_tokens=max_token_total,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
        )
        ans_model = response.choices[0].text

        all_info['raw_answer'] = ans_model
        ans_, scores = extract_ans(dataset_name, ans_model, input=q['input'], question_start=q_prefix, ans_gold=a)

        all_info['question'] = q['input']
        all_info['context'] = q['context']
        all_info['q_prefix'] = q_prefix
        all_info['prompted_question'] = prompt
        all_info['system_prompt_nudging'] = instruction_prompt
        all_info['full_prompt'] = prompt
        all_info['answer_start_prompt'] = answer_start_prompt
        all_info['extracted_answer'] = ans_
        all_info['scores'] = scores
        all_info["gold_answer"] = a

        return all_info

    def process_chunk(progress_bar, client, model, max_token_total, instruction_prompt, q_prefix, answer_start_prompt, temperature, input_data, output_data, dataset_name):
        chunk_results = []
        for q, a in zip(input_data, output_data):
            all_info = process_sample(client, model, max_token_total, instruction_prompt, q_prefix, answer_start_prompt, temperature, q, a, dataset_name)
            chunk_results.append(all_info)
            progress_bar.update(1)
        return chunk_results
    
    if os.path.exists(save_path) and not rerun:
        print(f"Load saved results from {save_path}")
    else:
        # delete file
        if os.path.exists(save_path):
            os.remove(save_path)

        instruction_prompt = PROMPTS[dataset_name]['system_nudging'] if model_type == 'nudging' else PROMPTS[dataset_name]['system']
        answer_start_prompt = PROMPTS[dataset_name]["answer_start"]
        q_prefix = PROMPTS[dataset_name]['question']
        print(f"instruction_prompt: {instruction_prompt}")
        print(f"q_prefix: {q_prefix}")
        print(f"answer_start_prompt: {answer_start_prompt}")

        chunk_size = len(input_data) // num_threads
        progress_bar = tqdm(total=len(input_data), desc="Processing samples")
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(0, len(input_data), chunk_size):
                chunk_input_data = input_data[i:i + chunk_size]
                chunk_output_data = output_data[i:i + chunk_size]
                futures.append((executor.submit(
                    process_chunk, progress_bar, client, model, max_token_total, instruction_prompt, q_prefix,
                    answer_start_prompt, temperature, chunk_input_data, chunk_output_data, dataset_name), i))

            for future, start_index in futures:
                result = future.result()
                all_info_list[start_index:start_index + len(result)] = result

        # Save results
        with open(save_path, 'a') as fd:
            for info in all_info_list:
                fd.write('Input_q: %s\nA_model:\n%s\nScores:\n%s\nA:\n%s\n\n' % (info['context'] + info['question'], info['extracted_answer'], json.dumps(info['scores'], indent=4), info['gold_answer']))

        with open(save_path_all_info, 'w') as f:
            json.dump(all_info_list, f)
        # Close the progress bar
        progress_bar.close()
    return parse_pred_ans(dataset_name, save_path, print_aggregated_metric=True)

def main(
    dataset_name: str,
    num_samples: int,
    num_threads: int,
    base_model: str,
    nudging_model: str,
    proxy_base_model: str,      # for the baseline methods (proxy tuning)
    proxy_nudging_model: str,   # for the baseline methods (proxy tuning)
    max_token_total: int,
    base_temperature: float,
    nudging_temperature: float,
    base_top_p: float,
    nudging_top_p: float,
    print_intermediate_output: bool,
    exp: str,
    exp_prefix: str,
    split: str,
    rerun: bool,
    baseline_method: str,
    top_prob_thres: float,
    completion_token_num: int,
    completion_token_num_nudging: int,
    base_host: str = None,
    nudging_host: str = None,
    proxy_base_host: str = None,
    proxy_nudging_host: str = None,
    use_local_host: bool = True,
):
    exp_prefix = f"split_{split}_" + exp_prefix
    ############################
    # For deploying the model using API providers like Together AI or Fireworks AI
    # Load the API keys
    # with open('togetherai-key.txt', 'r') as f:
    #     togetherai_api_key = f.read().strip()
    # with open('fireworks-key.txt', 'r') as f:
    #     fireworks_key = f.read().strip()
    # client_together_ai = OpenAI(
    #     api_key=togetherai_api_key,
    #     base_url="https://api.together.xyz/v1",
    # )
    # client_fireworks = OpenAI(
    #     base_url = "https://api.fireworks.ai/inference/v1",
    #     api_key=fireworks_key,
    # )
    ############################

    input_data, output_data, input_key, output_key = get_dataset(
        dataset_name=dataset_name,
        split=split,
        num_sample=num_samples
    )

    # print one example
    print("\nSample example: ")
    if input_key is not None:
        print(f"{input_key}:\n"+ input_data[0]['input'])
    else:
        print(input_data[0]['input'])
    if output_key is not None:
        print(f"{output_key}:\n"+ output_data[0])
    else:
        print(output_data[0])
    
    # set up the clients for the base model
    if use_local_host:   # local server
        openai_api_key = "EMPTY"
        client_base = OpenAI(
            api_key=openai_api_key,
            base_url=base_host,
        )
    ############################
    # Change here for deploying the model using API providers like Together AI or Fireworks AI
    # else:
    #     if base_model.startswith('accounts'):
    #         client_base = client_fireworks
    #     else:
    #         client_base = client_together_ai
    ############################

    # set up the clients for the nudging model
    if use_local_host:   # local server
        openai_api_key = "EMPTY"
        client_nudging = OpenAI(
            api_key=openai_api_key,
            base_url=nudging_host,
        )
    ############################
    # Change here for deploying the model using API providers like Together AI or Fireworks AI
    # else:
    #     if nudging_model.startswith('accounts'):
    #         client_nudging = client_fireworks
    #     else:
    #         client_nudging = client_together_ai
    ############################
    
    # set up the clients for the proxy models
    if proxy_base_model is not None:
        if use_local_host:   # local server
            openai_api_key = "EMPTY"
            client_proxy_base = OpenAI(
                api_key=openai_api_key,
                base_url=proxy_base_host,
            )
            client_proxy_nudging = OpenAI(
                api_key=openai_api_key,
                base_url=proxy_nudging_host,
            )
        ############################
        # Change here for deploying the model using API providers like Together AI or Fireworks AI
        # else:
        #     if proxy_base_model.startswith('accounts'):
        #         client_proxy_base = client_fireworks
        #         client_proxy_nudging = client_fireworks
        #     else:
        #         client_proxy_base = client_together_ai
        #         client_proxy_nudging = client_together_ai
        ############################
    if exp == 'nudging':
        exp_nudging(
            client_base=client_base,
            client_nudging=client_nudging,
            dataset_name=dataset_name,
            num_samples=num_samples,
            base_model=base_model,
            nudging_model=nudging_model,
            max_token_total=max_token_total,
            base_temperature=base_temperature,
            nudging_temperature=nudging_temperature,
            base_top_p=base_top_p,
            input_data=input_data,
            output_data=output_data,
            print_intermediate_output=print_intermediate_output,
            rerun=rerun,
            exp_prefix=exp_prefix,
            exp=exp,
            completion_token_num=completion_token_num,
            completion_token_num_nudging=completion_token_num_nudging,
            top_prob_thres=top_prob_thres,
            num_threads=num_threads,
        )
    elif exp == 'base_only':
        exp_single_model(
            client=client_base,
            dataset_name=dataset_name,
            num_samples=num_samples,
            model=base_model,
            max_token_total=max_token_total,
            input_data=input_data,
            output_data=output_data,
            rerun=rerun,
            exp_prefix=exp_prefix,
            num_threads=num_threads,
            temperature=base_temperature,
            top_p=base_top_p,
            model_type="base",
        )
    elif exp == 'nudging_only':
        exp_single_model(
            client=client_nudging,
            dataset_name=dataset_name,
            num_samples=num_samples,
            model=nudging_model,
            max_token_total=max_token_total,
            input_data=input_data,
            output_data=output_data,
            rerun=rerun,
            exp_prefix=exp_prefix,
            num_threads=num_threads,
            temperature=nudging_temperature,
            top_p=nudging_top_p,
            model_type="nudging",
        )
    elif exp == 'baseline':
        exp_baseline(
            client_base=client_base,
            client_proxy_chat=client_proxy_nudging,
            client_proxy_base=client_proxy_base,
            dataset_name=dataset_name,
            num_samples=num_samples,
            base_model=base_model,
            proxy_base_model=proxy_base_model,
            proxy_chat_model=proxy_nudging_model,
            max_token_total=max_token_total,
            baseline_method=baseline_method,
            temperature=base_temperature,
            input_data=input_data,
            output_data=output_data,
            rerun=rerun,
            exp_prefix=exp_prefix,
            num_threads=num_threads,
        )
    else:
        raise ValueError(f"Unknown experiment {exp}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="gsm8k", help="The name of the dataset")
    parser.add_argument("--num_samples", type=int, default=None, help="The number of samples")
    parser.add_argument("--num_threads", type=int, default=20, help="The number of threads")
    parser.add_argument("--base_model", type=str, default=None, help="The base model to use")
    parser.add_argument("--nudging_model", type=str, default=None, help="The nudging model to use")
    parser.add_argument("--proxy_base_model", type=str, default=None, help="The base model to use")
    parser.add_argument("--proxy_nudging_model", type=str, default=None, help="The nudging model to use")
    parser.add_argument("--completion_token_num", type=int, default=16, help="The number of token to complete in each nudging round using the base model")
    parser.add_argument("--completion_token_num_nudging", type=int, default=16, help="The number of token to complete in each nudging round using the nudging model")
    parser.add_argument("--max_token_total", type=int, default=512, help="The maximum number of tokens")
    parser.add_argument("--base_temperature", type=float, default=0.0, help="The temperature for the base model")
    parser.add_argument("--base_top_p", type=float, default=0.9, help="The top p for the base model")       # by default, we have temperature = 0 so top_p is not used by default
    parser.add_argument("--nudging_temperature", type=float, default=0.0, help="The temperature for the nudging model")
    parser.add_argument("--nudging_top_p", type=float, default=0.9, help="The top p for the nudging model") # by default, we have temperature = 0 so top_p is not used by default
    parser.add_argument("--baseline_method", type=str, choices=["ensemble", 'proxy_tuning'], default='ensemble', help="The baseline method name")
    parser.add_argument("--top_prob_thres", type=float, default=0.3, help="The top-1 token probability threshold for top prob nudging")
    parser.add_argument("--exp", type=str, choices=['nudging_only', "base_only", 'nudging', 'baseline'], default='nudging_only', help="The experiment to run")
    parser.add_argument("--exp_prefix", type=str, default="", help="The prefix for the experiment, e.g. complex_fewshot_")
    parser.add_argument("--split", type=str, default='test', help="The split to test")
    parser.add_argument("--base_host", type=str, default=None, help="The base host for local models")
    parser.add_argument("--nudging_host", type=str, default=None, help="The nudging host for local models")
    parser.add_argument("--proxy_base_host", type=str, default=None, help="The proxy base host for local models")
    parser.add_argument("--proxy_nudging_host", type=str, default=None, help="The proxy nudging host for local models")
    parser.add_argument("--rerun", action='store_true', help="Whether to rerun the experiment")
    parser.add_argument("--print_intermediate_output", action='store_true', help="Whether to print intermediate output")
    args = parser.parse_args()
    args = vars(args)

    main(**args)