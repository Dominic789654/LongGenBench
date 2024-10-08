import re
import time
import argparse
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json 
from datasets import load_dataset
import pandas as pd
import os


TASKS = [
        'abstract_algebra',
        'anatomy',
        'astronomy',
        'business_ethics',
        'clinical_knowledge',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_medicine',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'econometrics',
        'electrical_engineering',
        'elementary_mathematics',
        'formal_logic',
        'global_facts',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_european_history',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_mathematics',
        'high_school_microeconomics',
        'high_school_physics',
        'high_school_psychology',
        'high_school_statistics',
        'high_school_us_history',
        'high_school_world_history',
        'human_aging',
        'human_sexuality',
        'international_law',
        'jurisprudence',
        'logical_fallacies',
        'machine_learning',
        'management',
        'marketing',
        'medical_genetics',
        'miscellaneous',
        'moral_disputes',
        'moral_scenarios',
        'nutrition',
        'philosophy',
        'prehistory',
        'professional_accounting',
        'professional_law',
        'professional_medicine',
        'professional_psychology',
        'public_relations',
        'security_studies', 
        'sociology',
        'us_foreign_policy',
        'virology',
        'world_religions'
]

choices = ["A", "B", "C", "D"]


def test_answer(pred_str, ans_str):
    pattern = 'A|B|C|D'
    # pattern = '-?\d+\.?\d*'
    # pred_str = pred_str.replace(",","")
    pred = re.findall(pattern, pred_str)
    if(len(pred) >= 1):
        # print(pred_str)
        pred = pred[-1]
        gold = re.findall(pattern, ans_str)
        # print(ans_str)
        gold = gold[-1]
        return pred == gold
    else: return False


def parse_pred_ans(num_questions, filename):
    with open(filename) as fd: lines = fd.readlines()
    am, a = None, None
    num_q, acc = 0, 0
    current_mode = 'none'
    questions = []
    ans_pred = []
    ans_gold = []
    for l in lines:
        if(l.startswith('Question:')):
            if(am is not None and a is not None):
                questions.append(q)
                ans_pred.append(am)
                ans_gold.append(a)
                if(test_answer(am, a)):
                    acc += 1
            current_mode = 'q'
            q = l
            num_q += 1
        elif(l.startswith('A_model:')):
            current_mode = 'am'
            am = l
        elif(l.startswith('A_gold:')):
            current_mode = 'a'
            a = l
        else:
            if(current_mode == 'q'): q += l
            elif(current_mode == 'am'): am += l
            elif(current_mode == 'a'): a += l
            else:
                raise ValueError(current_mode)
                
    questions.append(q)
    ans_pred.append(am)
    ans_gold.append(a)
    if(test_answer(am, a)):
        acc += 1
    print('num_q %d correct %d ratio %.4f' % (num_questions, acc, float(acc / num_questions)))
    return questions, ans_pred, ans_gold, num_q, acc

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_batch_example(df, k, include_answer=True, include_question_mark=True):
    # result = []
    prompt = ""
    for i in range(k):
        if include_question_mark:
            prompt += "Question_{}:\n".format(i+1)
        prompt += df.iloc[i, 0]
        prompt += "\n"
        for j in range(df.shape[1] - 2):
            prompt += "({}) {}".format(choices[j], df.iloc[i, j+1])
        prompt += "\n\n"
    prompt += "\n\n"
    if include_answer:
        for i in range(k):
            prompt += "Answer_{}:\n".format(i+1)
            prompt += "{}\n\n".format(df.iloc[i, df.shape[1] - 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    sys_prompt = "Answer each question step by step, adhering to the format shown in the examples provided. Start each response with 'Answer_' . Do not repeat the question. Ensure that you respond to all the questions presented, regardless of their number. The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    prompt_q = format_batch_example(train_df, k, True,True)
    return [sys_prompt, prompt_q]

def extract_ans(ans_model):
    # breakpoint()
    ans_model = ans_model.split('Answer_')
    ans = []
    residual = []
    for li, al in enumerate(ans_model):
        ans.append(al)

    residual = list(ans_model[li + 1:])
    ans = '\n'.join(ans)
    residual = '\n'.join(residual)
    return ans, residual


def process_question(args, q_list, a_list, prompt_original, Prompt_TOKENS, Completion_TOKENS, Max_Prompt_LEN, Max_Completion_LEN, Max_Total_LEN):
    sys_prompt, prompt_q = prompt_original
    prompt_q = 'Examples: \n' + prompt_q + 'Following Question: \n'
    for i in range(len(q_list)):
        prompt_q = prompt_q + f'Question_{i+6}:\n'+ q_list[i] + '\n\n'

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_endpoint
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt_q},
    ]

    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=0,
            seed=42,
            max_tokens=4096
        )
        ans_model = response.choices[0].message.content
    except Exception as e:
        print(f"Error processing article {e}")
        time.sleep(3)
        return process_question(args, q_list, a_list, prompt_original, Prompt_TOKENS, Completion_TOKENS, Max_Prompt_LEN, Max_Completion_LEN, Max_Total_LEN)

    Prompt_TOKENS += response.usage.prompt_tokens
    Completion_TOKENS += response.usage.completion_tokens
    Max_Prompt_LEN = max(Max_Prompt_LEN, response.usage.prompt_tokens)
    Max_Completion_LEN = max(Max_Completion_LEN, response.usage.completion_tokens)
    Max_Total_LEN = max(Max_Total_LEN, response.usage.prompt_tokens + response.usage.completion_tokens)

    ans_, residual = extract_ans(ans_model)
    ans_ = re.findall(r'(?<=:\n)(.*?)(?=\n\n|$)', ans_, re.DOTALL)
    ans_list = [cur.strip() for cur in ans_]    
    result_str = ""

    pre_length = len(ans_list)
    if pre_length >= len(a_list):
        for i in range(len(a_list)):
            result_str += f"Question: {q_list[i]}\nA_model:Q_{i}\n{ans_list[i]}\nA_gold:\n{a_list[i]}\n\n"
    elif pre_length < len(a_list):
        for i in range(len(ans_list)):
            result_str += f"Question: {q_list[i]}\nA_model:Q_{i}\n{ans_list[i]}\nA_gold:\n{a_list[i]}\n\n"
        q_list = q_list[len(ans_list):]
        a_list = a_list[len(ans_list):]
        return process_question(args, q_list, a_list, prompt_original, Prompt_TOKENS, Completion_TOKENS, Max_Prompt_LEN, Max_Completion_LEN, Max_Total_LEN)
    return result_str, Prompt_TOKENS, Completion_TOKENS, Max_Prompt_LEN, Max_Completion_LEN, Max_Total_LEN


def parallel_process(args, questions, answers, prompt_original, output_file):
    total_tokens = {'Prompt_TOKENS': 0, 'Completion_TOKENS': 0, 'Max_Prompt_LEN': 0, 'Max_Completion_LEN': 0, 'Max_Total_LEN': 0}

    with ThreadPoolExecutor(max_workers=20) as executor:
        # Map each question and answer to a future
        future_to_question = {executor.submit(process_question, args, question, answer, prompt_original, total_tokens['Prompt_TOKENS'], total_tokens['Completion_TOKENS'], total_tokens['Max_Prompt_LEN'], total_tokens['Max_Completion_LEN'], total_tokens['Max_Total_LEN']): question for question, answer in zip(questions, answers)}

        # Process the futures as they complete
        for future in tqdm(as_completed(future_to_question), total=len(future_to_question), desc='Processing Questions'):
            result_str, prompt_tokens, completion_tokens, max_prompt_len, max_completion_len, max_total_len = future.result()
            total_tokens['Prompt_TOKENS'] += prompt_tokens
            total_tokens['Completion_TOKENS'] += completion_tokens
            total_tokens['Max_Prompt_LEN'] = max(total_tokens['Max_Prompt_LEN'], max_prompt_len)
            total_tokens['Max_Completion_LEN'] = max(total_tokens['Max_Completion_LEN'], max_completion_len)
            total_tokens['Max_Total_LEN'] = max(total_tokens['Max_Total_LEN'], max_total_len)

            # Write each result to the output file as soon as it's ready
            with open(output_file, 'a') as fd:
                fd.write(result_str)

    # Return the token statistics
    return total_tokens['Prompt_TOKENS'], total_tokens['Completion_TOKENS'], total_tokens['Max_Prompt_LEN'], total_tokens['Max_Completion_LEN'], total_tokens['Max_Total_LEN']


def batched_prompt(args, prompt_original):
    prompt_list = prompt_original.split('\n\nQ: ')[1:]
    question_list = []
    answer_list = []
    for i in range(len(prompt_list)):
        question_list.append(prompt_list[i].split('\nA: ')[0])
        answer_list.append(prompt_list[i].split('\nA: ')[1])
    batched_prompt = ""
    for i in range(len(question_list)):
        batched_prompt += f"Question_{i+1}:\n{question_list[i]}\n\n"
    for i in range(len(question_list)):
        batched_prompt += f"Answer_{i+1}:\n{answer_list[i]}\n\n"
    return batched_prompt

def main(args):
    output_file = args.output_path
    k = args.k  # Number of questions to process in parallel
    total_tokens = {'Prompt_TOKENS': 0, 'Completion_TOKENS': 0}
    total_num_q = 0
    total_acc = 0

    run_results = {}
    total_questions = 0
    for task in TASKS:
        new_question = []
        new_answer = []

        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", task + "_dev.csv"), header=None)
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", task + "_test.csv"), header=None)
        questions =  format_batch_example(test_df, test_df.shape[0], False, False).split('\n\n')
        answers = list(test_df.iloc[:, test_df.shape[1] - 1])

        prompt_original =  json.load(open('./data/LongGenBench_MMLU_prompt/LongGenBench_prompt.json'))[task]
        prompt_original = batched_prompt(args,prompt_original)
        prompt_original = (gen_prompt(dev_df, task, args.ntrain)[0], prompt_original)
        for i in range(0,test_df.shape[0],k):
            test_questions = questions[i:i+k]
            test_answers = answers[i:i+k]
            new_question.append(test_questions)
            new_answer.append(test_answers)
            total_questions += min(len(test_questions), k)

            break
        

        task_output_file = output_file+f"_{task}"
        with open(task_output_file, 'w') as fd:
            fd.write('')
        tokens_info = parallel_process(args, new_question, new_answer, prompt_original, task_output_file)
        print("*"*50)
        print(f'Starting task {task}...{TASKS.index(task)+1}/{len(TASKS)}')
        print("Prompt_TOKENS: ", tokens_info[0])
        print("Completion_TOKENS: ", tokens_info[1])
        total_tokens['Prompt_TOKENS'] += tokens_info[0]
        total_tokens['Completion_TOKENS'] += tokens_info[1]
        print("Max_Prompt_LEN: ", tokens_info[2])
        print("Max_Completion_LEN: ", tokens_info[3])
        print("Max_Total_LEN: ", tokens_info[4])
        print("*"*50)

        _, _, _, num_q, acc = parse_pred_ans(min(len(test_questions), k), task_output_file)

        total_num_q += num_q
        total_acc += acc
    print("Total Prompt_TOKENS: ", total_tokens['Prompt_TOKENS'])
    print("Total Completion_TOKENS: ", total_tokens['Completion_TOKENS'])
    print("Total num_q: ", total_questions)
    print("Total correct: ", total_acc)        
    print("Total acc ratio: ", total_acc/total_questions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--k', type=int, default=4, help='Number of questions to process in parallel')
    parser.add_argument('--data_dir', type=str, default='./data/MMLU/data')
    parser.add_argument('--output_path', type=str, default='./LongGenBench/outputs/LongGenBench_MMLU/')
    parser.add_argument('--api_key', type=str, default='your-api-key-here', help='OpenAI API Key')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI Model')
    parser.add_argument('--api_endpoint', type=str, default='https://api.openai.com/v1/', help='OpenAI API Endpoint')
    parser.add_argument('--ntrain', type=int, default=5)

    args = parser.parse_args()
    main(args)
    # pass