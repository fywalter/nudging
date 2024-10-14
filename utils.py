import numpy as np

import tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo") # use gpt3.5 tokenizer for token number controlling, so we don't need to load the actual tokenizer for API models

NUM_LOGPROBS = {
    'top_prob': 1,
}

def apply_instruct_template(model_name, system_prompt, instruct_prompt, response_prompt, add_bos=False):
    model_name = model_name.lower()

    if "chat" in model_name and "llama" in model_name and "2" in model_name:
        return llama_2_chat_template(system_prompt=system_prompt, instruct_prompt=instruct_prompt, response_prompt=response_prompt, add_bos=add_bos)
    elif "instruct" in model_name and "llama" in model_name and "3" in model_name:
        if "3.1" in model_name: # for llama-3.1 models, add knowledge cut in system prompmt
            return llama_3_instruct_template(system_prompt=system_prompt, instruct_prompt=instruct_prompt, response_prompt=response_prompt, add_bos=add_bos, add_knowledge_cut=True)
        else:
            return llama_3_instruct_template(system_prompt=system_prompt, instruct_prompt=instruct_prompt, response_prompt=response_prompt, add_bos=add_bos)
    elif "it" in model_name and "gemma" in model_name:
        return gemma_instruct_template(system_prompt=system_prompt, instruct_prompt=instruct_prompt, response_prompt=response_prompt, add_bos=add_bos)
    elif "instruct" in model_name and "olmo" in model_name:
        return olmo_instruct_template(system_prompt=system_prompt, instruct_prompt=instruct_prompt, response_prompt=response_prompt, add_bos=add_bos) 
    else:
        return f"{system_prompt}\n{instruct_prompt}\n{response_prompt}" # non-instruct model or models with unknown template

def llama_2_chat_template(system_prompt, instruct_prompt, response_prompt, add_bos=False):
    """
    Convert the input and output into the template used for the llama-2 chat models training.
    """
    prefix = "<s>" if add_bos else ""
    return prefix + f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruct_prompt} [/INST] {response_prompt.lstrip()}"  # for most servers that add <s> automatically so we don't need to add it here

def llama_3_instruct_template(system_prompt, instruct_prompt, response_prompt, add_bos=False, add_knowledge_cut=False):
    """
    Convert the input and output into the template used for the llama-3 instruct models training.
    """
    # print("applying llama-3 instruct template")
    prefix = "<|begin_of_text|>" if add_bos else ""
    if add_knowledge_cut:
        system_prompt = f"Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"+ system_prompt
    return prefix + f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruct_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response_prompt}"

def gemma_instruct_template(system_prompt, instruct_prompt, response_prompt, add_bos=False):
    """
    Convert the input and output into the template used for the gemma instruct models training.
    <bos><start_of_turn>user
    Write a hello world program<end_of_turn>
    <start_of_turn>model
    """
    prefix = "<bos>" if add_bos else ""
    return prefix + f"<start_of_turn>user\n{system_prompt}\n{instruct_prompt}<end_of_turn>\n<start_of_turn>model\n{response_prompt}"

def olmo_instruct_template(system_prompt, instruct_prompt, response_prompt, add_bos=False):
    """
    Convert the input and output into the template used for the olmo instruct models training.
    """
    return f"<|endoftext|><|user|>\n{system_prompt}\n{instruct_prompt}\n<|assistant|>\n{response_prompt}"

def find_longest_repeated_suffix(s):
    
    # Helper function to check if a substring repeats
    def has_repeated(s, length):
        if length < 30:
            return False
        # Extract the suffix of length 'length'
        suffix = s[-length:]
        # Check the rest of the string for another occurrence
        # return s[:-length].find(suffix) != -1
        return s[:-length].endswith(suffix)

    left, right = 0, len(s)
    result = 0

    # Binary search for the longest repeated suffix
    while left <= right:
        mid = (left + right) // 2
        if has_repeated(s, mid):
            result = mid  # Store the longest length found
            left = mid + 1  # Try for a longer suffix
        else:
            right = mid - 1  # Try for a shorter suffix

    # Return the longest repeated suffix
    if result > 0:
        return s[-result:]
    return None  # Return an empty string if no repetition is found

def remove_redundant_repetitions(s):
    s = s.strip()
    # Find the longest repeated suffix
    longest_repeated_suffix = find_longest_repeated_suffix(s)
    while longest_repeated_suffix:
        # Remove the longest repeated suffix
        s = s[:-len(longest_repeated_suffix)]
        # Find the longest repeated suffix again
        longest_repeated_suffix = find_longest_repeated_suffix(s)
    return s

def repetition_check(new_completion, full_prefix, subseq_len=5):
    words = new_completion.split(" ")
    if len(words) > subseq_len and new_completion in full_prefix:
        return True
    return False

def check_need_nudging(nudging_method,
                        base_token_id,
                        current_base_info, 
                        thresholds,
):
    if nudging_method == 'top_prob':
        # check if the token prob is below the threshold
        sorted_base_top_logprobs = {k: v for k, v in sorted(current_base_info["top_logprobs"][base_token_id].items(), key=lambda item: item[1], reverse=True)}
        base_top_prob = np.exp(list(sorted_base_top_logprobs.values())[0])
        need_nudging = base_top_prob < thresholds['top_prob']
    else:
        raise ValueError(f"Unknown nudging method {nudging_method}")
    return need_nudging

def complete_with_base(nudging_method='top_prob',
                        base_model="davinci-002",
                        full_prefix_base="",
                        output="",
                        current_base_info=None,
                        max_completion_token=256,
                        completion_token_num=16,
                        client_base=None,
                        thresholds=None,
                        temperature=0.0,
                        top_p=0.9,
                        ):
    completion_base = "" if len(current_base_info["completion"]) == 0 else current_base_info["tokens"][0]   # accept the first token from the 1st round which is the acc token from the first stage
    completion_all = "" if len(current_base_info["completion"]) == 0 else current_base_info["tokens"][0]    # completion_all records all the tokens from the base model including the tokens that are not accepted in the last round, for debugging and visualization
    found_nudging_token = False
    response = None
    has_acc_token_stage_1 = True if len(current_base_info["completion"]) > 0 else False                     # if the current_base_info["completion"] is not empty, it means the first token in base completion is accepted from the 1st stage
    EMPTY_INFO_DICT = {
        "completion": "",
        "tokens": [],
        "top_logprobs": [],
        "stop_reason": None, 
        "num_logprobs": NUM_LOGPROBS[nudging_method],
    }
    next_nudging_info = EMPTY_INFO_DICT     # for nudging methods that compute nudging info during base completion, we can save the info for the next round, currently not used for top_prob nudging
    while len(encoding.encode(completion_base)) < max_completion_token and not found_nudging_token:
       
        if current_base_info["completion"] == "":
            # complete the sentence using the base model
            response = client_base.completions.create(
                model=base_model,
                prompt=full_prefix_base + output + completion_base,
                max_tokens=completion_token_num,
                temperature=temperature,
                logprobs=current_base_info["num_logprobs"],
                top_p=top_p,
                )
            current_base_info["tokens"] = response.choices[0].logprobs.tokens
            current_base_info["top_logprobs"] = response.choices[0].logprobs.top_logprobs
            current_base_info["completion"] = response.choices[0].text

        if has_acc_token_stage_1:
            # pop the first token from the 1st round as it is already accepted from stage 1
            current_base_info["tokens"] = current_base_info["tokens"][1:]
            current_base_info["top_logprobs"] = current_base_info["top_logprobs"][1:]
            current_base_info["completion"] = "".join(current_base_info["tokens"])
            has_acc_token_stage_1 = False

        completion = current_base_info["completion"]
        tokens = current_base_info["tokens"]

        if completion in completion_base:
            break   # repeated completion, break

        nudging_position = -1

        # find the first token that violates the nudging criteria
        for base_idx in range(len(tokens)):
            found_nudging_token = check_need_nudging(nudging_method=nudging_method, base_token_id=base_idx, current_base_info=current_base_info, thresholds=thresholds)
            if found_nudging_token:
                nudging_position = base_idx
                break
        
        if nudging_position == -1:
            new_completion= "".join(tokens)
        else:
            new_completion = "".join(tokens[:nudging_position])   # include the last agreed token
        # avoid repetition in answer
        if repetition_check(new_completion, output + completion_base):
            break
        else:
            completion_base += new_completion

        if found_nudging_token: # if found the nudging token, break the loop, concat the last base completion to completion_all
            completion_all += completion
        else:
            completion_all += new_completion

        next_nudging_info = EMPTY_INFO_DICT
        if response is not None and response.choices[0].finish_reason == "stop":
            break

        # reset the current_base_info
        current_base_info['completion'] = ""
        current_base_info['tokens'] = []
        current_base_info['top_logprobs'] = []

    return completion_base, completion_all, next_nudging_info

def completion_with_nudging(
        base_model="davinci-002",
        nudging_model="gpt-3.5-turbo",
        system_prompt_base="Answer the question by walking through the reasoning step by step.",
        system_prompt_nudging="Answer the question by walking through the reasoning step by step.",
        question="",
        context="",
        question_prompt="Question: ",
        answer_start_prompt_base="Answer: ",
        answer_start_prompt_nudging="Answer: ",
        completion_token_num=16,
        completion_token_num_nudging=16,
        max_token_total=256,
        print_intermediate_output=False,
        client=None,                # default client
        client_base=None,
        client_nudging=None,
        max_round=150,
        nudging_temperature=0.0,    # deterministic for nudging
        base_temperature=0.0,       # deterministic for base model
        nudging_method='top_prob',
        top_prob_thres=0.3,
        top_p=0.9,
        ):
    if client_base is None:
        client_base = client
    if client_nudging is None:
        client_nudging = client

    if nudging_method not in NUM_LOGPROBS.keys():
        raise ValueError(f"nudging method {nudging_method} number of logprobs not defined")

    full_prefix_base = apply_instruct_template(base_model, system_prompt_base, context + question_prompt + question, answer_start_prompt_base)  # for base model this function just adds newlines
    full_prefix_nudging = apply_instruct_template(nudging_model, system_prompt_nudging, context + question_prompt + question, answer_start_prompt_nudging)

    thresholds = {
        'top_prob': top_prob_thres,
    }

    output = ""
    nudging_round = 0
    all_nudging_words = []
    all_nudging_and_completions = []
    current_nudging_info = {
        "completion": "",
        "tokens": [],
        "top_logprobs": [],
        "stop_reason": None,
        "num_logprobs": NUM_LOGPROBS[nudging_method],
    }
    stop_reason = None
    repeat_nudging_word = 0
    last_nudging_word = ""
    while len(encoding.encode(output)) < max_token_total and nudging_round < max_round:    # use the number of gpt-3.5 token to approximately control the length
        nudging_round += 1
        if current_nudging_info["completion"] == "":
            response = client_nudging.completions.create(
                model=nudging_model,
                prompt=full_prefix_nudging + output,
                max_tokens=completion_token_num_nudging,
                temperature=nudging_temperature,
                logprobs=current_nudging_info["num_logprobs"],
                )
            current_nudging_info["completion"] = response.choices[0].text
            current_nudging_info["tokens"] = response.choices[0].logprobs.tokens
            current_nudging_info["top_logprobs"] = response.choices[0].logprobs.top_logprobs
            current_nudging_info["stop_reason"] = response.choices[0].finish_reason

        # if finish_reason is stop, break the loop, also handles nudging completion from previous round
        if current_nudging_info["stop_reason"] == "stop":
            stop_reason = "nudging_model_stop"
            if len(current_nudging_info["completion"]) > 0:
                all_nudging_words.append(current_nudging_info["completion"])
                all_nudging_and_completions.append(current_nudging_info["completion"])
                output += current_nudging_info["completion"]
            break

        # ===================================================================
        # Stage 1: use base model to find the first token that violates the nudging criteria (no need to nudge)
        # ===================================================================
        found_acc_token = False
        current_base_info = {   # will be passed to the next stage
            "completion": "",
            "tokens": [],
            "top_logprobs": [],
            "num_logprobs": NUM_LOGPROBS[nudging_method],
        }
        nudging_text = current_nudging_info["completion"]
        num_whitespaces = len(nudging_text) - len(nudging_text.lstrip(" "))
        space_prefix = " " * num_whitespaces
        current_nudging_words = nudging_text.lstrip(" ").split(" ")     # token leads to some unexpected behaviors, still use nudging word
        nudging_word_id = 0 if len(current_nudging_words) > 1 else 1    # if only one word, always accept the word and go to the next round: it won't go into the loop and found_acc_token will be False
        while not found_acc_token and nudging_word_id < len(current_nudging_words) - 1:
            nudging_word_id += 1                # always accept the first word
            nudging_gen_prefix = space_prefix + " ".join(current_nudging_words[:nudging_word_id])
            current_nudging_word = " " + current_nudging_words[nudging_word_id]  # add a leading space to the current nudging word since the nudging words a split by space
            if current_nudging_word == " ":     # skip the multiple space
                continue
            prefix = full_prefix_base + output + nudging_gen_prefix
            response = client_base.completions.create(
                model=base_model,
                prompt=prefix,
                max_tokens=completion_token_num,
                temperature=base_temperature,
                logprobs=current_base_info["num_logprobs"],
                top_p=top_p,
                )
            current_base_info["tokens"] = response.choices[0].logprobs.tokens
            current_base_info["top_logprobs"] = response.choices[0].logprobs.top_logprobs
            current_base_info["completion"] = response.choices[0].text

            # look for the first token that meets the nudging criteria
            first_base_token = current_base_info["tokens"][0]            
            if current_nudging_word.startswith(first_base_token): # check if the current nudging word is the same or starts with the first base token
                found_acc_token = True
            else: 
                found_acc_token = not check_need_nudging(nudging_method,    # check if the token violates the nudging criteria (no need to nudge)
                                                         base_token_id=0,
                                                         current_base_info=current_base_info, 
                                                         thresholds=thresholds)
                
        # here we have either prefix_idx == len(current_nudging_info["tokens"]):    if no token meets the nudging criteria, use the current nudging completion
        # or found_acc_token == True:    if a token violates the nudging criteria, we use the prefix as nudging tokens
        
        nudging_words = space_prefix +  " ".join(current_nudging_words[:nudging_word_id])
        
        # Heuristic: if the nudging words are the same as the last one for three rounds, break the loop
        if nudging_words == last_nudging_word:
            repeat_nudging_word += 1
            if repeat_nudging_word >= 3:
                stop_reason = "repeated_nudging_words"
                break
        else:
            last_nudging_word = nudging_words
            repeat_nudging_word = 0
        all_nudging_words.append(nudging_words)
        output += nudging_words

        if not found_acc_token: # if no base token can be accepted, use the current nudging completion and go to the next round
            all_nudging_and_completions.append(nudging_words)
            # reset the current nudging info and continue to the next round
            current_nudging_info = {
                "completion": "",
                "tokens": [],
                "logprobs": [],
                "stop_reason": None,
                "num_logprobs": NUM_LOGPROBS[nudging_method],
            }
            continue
        if current_base_info["completion"] == "":   # the base model thinks the completion is done, go to the next round. Make sure current_base_info["completion"] is not empty if proceed to the next stage
            all_nudging_and_completions.append(nudging_words)
            current_nudging_info = {
                "completion": "",
                "tokens": [],
                "logprobs": [],
                "stop_reason": None,
                "num_logprobs": NUM_LOGPROBS[nudging_method],
            }
            continue

        # ===================================================================
        # Stage 2: use nudging model to find the first token that meets the nudging criteria (need to nudge)
        # ===================================================================
        max_completion_token = max_token_total - len(encoding.encode(output))
        completion_base, completion_base_all, current_nudging_info = complete_with_base(nudging_method=nudging_method,
                                                                                        base_model=base_model,
                                                                                        full_prefix_base=full_prefix_base,
                                                                                        output=output,
                                                                                        current_base_info=current_base_info,
                                                                                        max_completion_token=max_completion_token,
                                                                                        completion_token_num=completion_token_num,
                                                                                        client_base=client_base,
                                                                                        thresholds=thresholds,
                                                                                        temperature=base_temperature,
                                                                                        top_p=top_p,
                                                                                        )
        # print(f"next_nudging_info: {current_nudging_info}") # debug

        output += completion_base
        all_nudging_and_completions.append(nudging_words + completion_base) # the generated tokens in each round, concating all completion would be the final output
        if print_intermediate_output:
            print(f"************nudging round {nudging_round}************")
            print(f"****nudging words from {nudging_model}****: {nudging_words}")
            print(f"****nudging text****: {nudging_text}")
            print(f"****completion from {base_model}****: {completion_base}")
            print(f"****all completion from {base_model}****: {completion_base_all}")
            print(f"****output****: {output}")
    
    if nudging_round >= max_round and not stop_reason:
        stop_reason = "round"
    if len(encoding.encode(output)) >= max_token_total and not stop_reason:
        stop_reason = "length"
    output = remove_redundant_repetitions(output)
    if print_intermediate_output:
        print(f"************final output************")
        print(f"****output****: {output}")

    all_info = {
        "question": question,
        "context": context,
        "raw_answer": output,
        "all_nudging_words": all_nudging_words,
        "all_completions": all_nudging_and_completions,
        "stop_reason": stop_reason,
        "system_prompt_base": system_prompt_base,
        "system_prompt_nudging": system_prompt_nudging,
        "full_prefix_base": full_prefix_base,
        "full_prefix_nudging": full_prefix_nudging,
    }
    return all_info

############################################################################################################
# Baseline completion
############################################################################################################
def completion_baseline_ensemble(
        base_model="davinci-002",
        proxy_chat_model="gpt-3.5-turbo",
        client_base=None,
        client_proxy_chat=None,
        max_token_total=256,
        full_prefix_base="",
        full_prefix_proxy_chat="",
        temperature=0.0,
        completion_token_num=16,
        logprobs=5,
        ):
    output = ''
    while len(encoding.encode(output)) < max_token_total:
        response_base = client_base.completions.create(
            model=base_model,
            prompt=full_prefix_base + output,
            max_tokens=completion_token_num,
            temperature=temperature,
            logprobs=logprobs,
            )
        response_proxy_chat = client_proxy_chat.completions.create(
            model=proxy_chat_model,
            prompt=full_prefix_proxy_chat + output,
            max_tokens=completion_token_num,
            temperature=temperature,
            logprobs=logprobs,
            )
        base_tokens = response_base.choices[0].logprobs.tokens
        proxy_chat_tokens = response_proxy_chat.choices[0].logprobs.tokens
        base_logprobs = response_base.choices[0].logprobs.top_logprobs
        proxy_chat_logprobs = response_proxy_chat.choices[0].logprobs.top_logprobs
        # stop criteria: if chat model finish reason is stop, break the loop
        if response_proxy_chat.choices[0].finish_reason == "stop":
            output += "".join(proxy_chat_tokens)
            break
        acc_tokens = []
        for i in range(len(base_tokens)):
            if base_tokens[i] == proxy_chat_tokens[i]:
                acc_tokens.append(base_tokens[i])
            else:
                base_top_logprobs = {k: v for k, v in sorted(base_logprobs[i].items(), key=lambda item: item[1], reverse=True)}
                proxy_chat_top_logprobs = {k: v for k, v in sorted(proxy_chat_logprobs[i].items(), key=lambda item: item[1], reverse=True)}
                all_keys = set(base_top_logprobs.keys()).union(proxy_chat_top_logprobs.keys())
                ensemble_top_probs = {}
                for key in all_keys:
                    base_prob = np.exp(base_top_logprobs.get(key, -1e6))
                    proxy_chat_prob = np.exp(proxy_chat_top_logprobs.get(key, -1e6))
                    ensemble_top_probs[key] = base_prob + proxy_chat_prob
                ensemble_top_probs = {k: v for k, v in sorted(ensemble_top_probs.items(), key=lambda item: item[1], reverse=True)}
                acc_tokens.append(list(ensemble_top_probs.keys())[0])
                break
        new_completion = "".join(acc_tokens)
        if len(new_completion) == 0:
            # if no token is accepted, add the first non-empty token from the base model
            for i in range(len(base_tokens)):
                if len(base_tokens[i]) > 0:
                    acc_tokens.append(base_tokens[i])
                    break
            # if base model has no token, add the first token from the proxy chat model
            new_completion = "".join(acc_tokens)
            if len(new_completion) == 0:
                for i in range(len(proxy_chat_tokens)):
                    if len(proxy_chat_tokens[i]) > 0:
                        acc_tokens.append(proxy_chat_tokens[i])
                        break
        output += "".join(acc_tokens)
    return output

def completion_baseline_proxy_tuning(
        base_model="davinci-002",
        proxy_chat_model="gpt-3.5-turbo",
        proxy_base_model="davinci-002",
        client_base=None,
        client_proxy_chat=None,
        client_proxy_base=None,
        max_token_total=256,
        full_prefix_base="",
        full_prefix_proxy_chat="",
        full_prefix_proxy_base="",
        temperature=0.0,
        completion_token_num=16,
        logprobs=100,
        ):
    output = ''
    while len(encoding.encode(output)) < max_token_total:
        response_base = client_base.completions.create(
            model=base_model,
            prompt=full_prefix_base + output,
            max_tokens=completion_token_num,
            temperature=temperature,
            logprobs=logprobs,
            )
        response_proxy_chat = client_proxy_chat.completions.create(
            model=proxy_chat_model,
            prompt=full_prefix_proxy_chat + output,
            max_tokens=completion_token_num,
            temperature=temperature,
            logprobs=logprobs,
            )
        response_proxy_base = client_proxy_base.completions.create(
            model=proxy_base_model,
            prompt=full_prefix_proxy_base + output,
            max_tokens=completion_token_num,
            temperature=temperature,
            logprobs=logprobs,
            )
        base_tokens = response_base.choices[0].logprobs.tokens
        proxy_chat_tokens = response_proxy_chat.choices[0].logprobs.tokens
        proxy_base_tokens = response_proxy_base.choices[0].logprobs.tokens
        base_logprobs = response_base.choices[0].logprobs.top_logprobs
        proxy_chat_logprobs = response_proxy_chat.choices[0].logprobs.top_logprobs
        proxy_base_logprobs = response_proxy_base.choices[0].logprobs.top_logprobs
        # stop criteria: if chat model finish reason is stop, break the loop
        if response_proxy_chat.choices[0].finish_reason == "stop":
            output += "".join(proxy_chat_tokens)
            break
        acc_tokens = []
        for i in range(len(base_tokens)):
            base_top_logprobs = {k: v for k, v in sorted(base_logprobs[i].items(), key=lambda item: item[1], reverse=True)}
            proxy_chat_top_logprobs = {k: v for k, v in sorted(proxy_chat_logprobs[i].items(), key=lambda item: item[1], reverse=True)}
            proxy_base_top_logprobs = {k: v for k, v in sorted(proxy_base_logprobs[i].items(), key=lambda item: item[1], reverse=True)}
            shared_keys = set(base_top_logprobs.keys()).intersection(proxy_chat_top_logprobs.keys()).intersection(proxy_base_top_logprobs.keys())
            rescaled_probs = {}
            for key in shared_keys:
                base_prob = np.exp(base_top_logprobs[key])
                proxy_chat_prob = np.exp(proxy_chat_top_logprobs[key])
                proxy_base_prob = np.exp(proxy_base_top_logprobs[key])
                rescaled_probs[key] = base_prob * proxy_chat_prob / proxy_base_prob
            rescaled_probs = {k: v for k, v in sorted(rescaled_probs.items(), key=lambda item: item[1], reverse=True)}
            if len(rescaled_probs) == 0:
                acc_tokens.append(base_tokens[i])
                top_1_token = None
            else:
                top_1_token = list(rescaled_probs.keys())[0]
                acc_tokens.append(top_1_token)
            if top_1_token == base_tokens[i] and top_1_token == proxy_chat_tokens[i] and top_1_token == proxy_base_tokens[i]:   # if all models agree, continue to the next token
                continue
            else:
                break
        new_completion = "".join(acc_tokens)
        if len(new_completion) == 0:
            # if no token is accepted, add the first non-empty token from the base model
            for i in range(len(base_tokens)):
                if len(base_tokens[i]) > 0:
                    acc_tokens.append(base_tokens[i])
                    break
            # if base model has no token, add the first token from the proxy chat model
            new_completion = "".join(acc_tokens)
            if len(new_completion) == 0:
                for i in range(len(proxy_chat_tokens)):
                    if len(proxy_chat_tokens[i]) > 0:
                        acc_tokens.append(proxy_chat_tokens[i])
                        break
        output += "".join(acc_tokens)
    return output

def completion_with_baseline(client_base,
                            client_proxy_chat,
                            client_proxy_base,
                            base_model,
                            proxy_chat_model,
                            proxy_base_model,
                            baseline_method,
                            max_token_total=512,
                            instruction_prompt="",
                            q_prefix="Question: ",
                            answer_start_prompt="",
                            temperature=0,
                            context="",
                            question="",
                            completion_token_num=16,
                            debug=False,
                            ):
    all_info = {}
    full_prefix_base = apply_instruct_template(base_model, instruction_prompt, context + q_prefix + question, answer_start_prompt)
    full_prefix_proxy_chat = apply_instruct_template(proxy_chat_model, instruction_prompt, context + q_prefix + question, answer_start_prompt)
    full_prefix_proxy_base = apply_instruct_template(proxy_base_model, instruction_prompt, context + q_prefix + question, answer_start_prompt)

    if baseline_method == 'ensemble':
        ans_model = completion_baseline_ensemble(
            base_model=base_model,
            proxy_chat_model=proxy_chat_model,
            client_base=client_base,
            client_proxy_chat=client_proxy_chat,
            max_token_total=max_token_total,
            full_prefix_base=full_prefix_base,
            full_prefix_proxy_chat=full_prefix_proxy_chat,
            temperature=temperature,
            completion_token_num=completion_token_num,
            debug=debug,
        )
    elif baseline_method == 'proxy_tuning':
        ans_model = completion_baseline_proxy_tuning(
            base_model=base_model,
            proxy_chat_model=proxy_chat_model,
            proxy_base_model=proxy_base_model,
            client_base=client_base,
            client_proxy_chat=client_proxy_chat,
            client_proxy_base=client_proxy_base,
            max_token_total=max_token_total,
            full_prefix_base=full_prefix_base,
            full_prefix_proxy_chat=full_prefix_proxy_chat,
            full_prefix_proxy_base=full_prefix_proxy_base,
            temperature=temperature,
            completion_token_num=completion_token_num,
            debug=debug,
        )
    else:
        raise ValueError(f"Unknown baseline method {baseline_method}")
    all_info['raw_answer'] = ans_model
    all_info['question'] = question
    all_info['context'] = context
    all_info["full_prefix_proxy_chat"] = full_prefix_proxy_chat
    return all_info