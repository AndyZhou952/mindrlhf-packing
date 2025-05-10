# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from .math_eval_toolkit.parser import extract_answer
from .math_eval_toolkit.grader import math_equal
import re 

from multiprocessing import Process, Queue

def qwen_math_equal_subprocess(prediction, reference,  timeout_seconds=10):

    def worker(q, prediction, reference):
        result = math_equal(prediction=prediction, reference=reference, timeout=False)
        q.put(result)

    q = Queue()
    p = Process(target=worker, args=(q, prediction, reference))
    p.start()
    
    p.join(timeout=timeout_seconds)
    
    if p.is_alive():
        p.terminate()
        p.join()  
        return False
        
    try:
        return q.get_nowait()
    except:
        return False   

def preprocess_box_response_for_qwen_prompt(sequences, answers, **kwargs):
    scores = []

    for sequence, answer in zip(sequences, answers):
        model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', sequence, flags=re.DOTALL,count = 1)
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
        for stop_word in stop_words:
            if stop_word in model_output:
                model_output = model_output.split(stop_word)[0].strip()
        ext_answer = extract_answer(model_output, data_name="math")

        if qwen_math_equal_subprocess(prediction=ext_answer, reference=answer):
            box_match = 1.0
        else:
            box_match = -0.5
            
        if "boxed" not in model_output:
            box_match = -1.0

        scores.append(box_match)
        
    return scores

def format_reward(sequences, **kwargs):
    """
    Reward function that checks if the completion has a specific format.

    Args:
        sequences: A list of sequences, where each completion is a tuple containing a list of dictionaries.
                     Each dictionary should have a "content" key with the text to be checked.

    Returns:
        A list of floats, where each float is 1.0 if the corresponding completion matches the required format,
        and 0.0 otherwise.

    Raises:
        ValueError: If the input sequences are not in the expected format.
    """
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"

    if not isinstance(sequences, list):
        raise ValueError("Input sequences must be a list.")

    rewards = []
    for completion in sequences:
        if re.match(pattern, completion, re.DOTALL | re.MULTILINE):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards

def reasoning_steps_reward(sequences, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    matches = [len(re.findall(pattern, content)) for content in sequences]

    return [min(1.0, count / 3) for count in matches]

def func_from_jiaoda(sequences, answers):

    class RuleBasedRMProxy:
        def __init__(self):
            self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
            self.english_pattern = re.compile(r'[a-zA-Z]')
            self.boxed_pattern = re.compile(r"\\boxed\{((?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{[^{}]*\}))*\}))*\}))*\})")
            self.valid_char_pattern = re.compile(r'[a-zA-Z0-9\s\.,!?"\'\(\)\{\}\[\]_\-+=<>/@#$%^&*\\|:;~`\u2200-\u22FF]')
            self.repeat_pattern = re.compile(r'(.{5,}?)\1{4,}')

        def check_mixed_languages(self, text):
            chinese_chars = len(self.chinese_pattern.findall(text))
            english_chars = len(self.english_pattern.findall(text))
            return chinese_chars >= 20 and english_chars >= 20
        
        def check_garbled_characters(self, text):
            valid_chars = self.valid_char_pattern.sub('', text)
            if not text: 
                return False
            invalid_ratio = len(valid_chars) / len(text)
            return invalid_ratio > 0.3
        
        def has_repeated_patterns(self, text):
            return bool(self.repeat_pattern.search(text))
        
        def correctness_score(self, label, response):
            matches = self.boxed_pattern.findall(response)
            if not matches:
                # print('--->not mathched, return -1', label, response)
                return -1.0
            
            pred = matches[-1][:-1]
            return 1.0 if math_equal(label, pred) else -0.5
        
        def split_and_score(self, response, label):
            # if "<|im_end|>" not in response and "<|endoftext|>" not in response:
            #     return -1.0
            if "\\boxed" not in response or response.count("\\boxed")>=5:
                # print('box-->enter', "\\boxed" not in response, response, response.count("\\boxed")>=5)
                return -1.0
            
            # if self.check_mixed_languages(response):
            #     return -1.0
                
            if self.check_garbled_characters(response):
                # print('garbled-->enter')
                return -1.0
            
            if self.has_repeated_patterns(response):
                # print('repeat-->repeat')
                return -1.0
            
            return self.correctness_score(label, response)
        
        def get_reward(self, queries):
            scores = []
            for query in queries:
                score = self.split_and_score(query)
                scores.append(score)
            return scores
        
    checker = RuleBasedRMProxy()
    res = []
    # print(len(sequences), len(labels))
    for r, l in zip(sequences, answers):
        res.append(checker.split_and_score(r, l))
    return res

def _test_rule_verifier():
    text = ["""Assistant\nWanda guesses that the Fermat point is at $P = (4,2)$, so the sum of the distances from $P$ to the vertices of $\\triangle ABC$ is $PA + PB + PC = \\sqrt{4^2 + 2^2} + \\sqrt{(4-10)^2 + 3^2 + (5-2)^2 = 2\\sqrt5 + \\sqrt{29}$, so $5 + 2\\sqrt{10}$. Therefore, $m = \\boxed{5}$ and $n = \\boxed{2}$, and $m + n = \\boxed{7}$."""]
    label = ["5"]

    print('reward_verifier=',preprocess_box_response_for_qwen_prompt(text, label))
    print('format_verifier=',format_reward(text))
    print('step_verifier=',reasoning_steps_reward(text))
    print('jiaoda=',func_from_jiaoda(text, label))
    

if __name__ == "__main__":
    _test_rule_verifier()
