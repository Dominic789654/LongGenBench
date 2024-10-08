import re
import argparse
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from contextlib import contextmanager
from typing import Generator, List


@contextmanager
def get_openai_client(args) -> Generator[OpenAI, None, None]:
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_endpoint
    )
    try:
        yield client
    finally:
        client.close()



# Function to test if the predicted answer matches the gold answer
def test_answer(pred_str, ans_str):
    pattern = '\d*\.?\d+'
    pred_str = pred_str.replace(",", "")
    pred = re.findall(pattern, pred_str)
    if len(pred) >= 1:
        pred = pred[-1]
        gold = re.findall(pattern, ans_str)
        gold = gold[-1]
        return pred == gold
    else:
        return False

# Function to parse predictions and answers from a file and calculate accuracy
def parse_pred_ans(filename):
    with open(filename) as fd:
        lines = fd.readlines()
    am, a = None, None
    num_q, acc = 0, 0
    current_mode = 'none'
    questions, ans_pred, ans_gold = [], [], []
    for l in lines:
        if l.startswith('Question:'):
            if am is not None and a is not None:
                questions.append(q)
                ans_pred.append(am)
                ans_gold.append(a)
                if test_answer(am, a):
                    acc += 1
            current_mode = 'q'
            q = l
            num_q += 1
        elif l.startswith('A_model:'):
            current_mode = 'am'
            am = l
        elif l.startswith('A_gold:'):
            current_mode = 'a'
            a = l
        else:
            if current_mode == 'q':
                q += l
            elif current_mode == 'am':
                am += l
            elif current_mode == 'a':
                a += l
            else:
                raise ValueError(current_mode)
                
    questions.append(q)
    ans_pred.append(am)
    ans_gold.append(a)
    if test_answer(am, a):
        acc += 1
    print(f'num_q {num_q} correct {acc} ratio {acc / num_q:.4f}')
    return questions, ans_pred, ans_gold

def extract_ans(ans_model):
    ans_model = ans_model.split('Answer_')
    ans = []
    for al in ans_model:
        ans.append(al)
    residual = list(ans_model[len(ans):])
    ans = '\n'.join(ans)
    residual = '\n'.join(residual)
    return ans, residual

def process_question(args, q_list, a_list, prompt_original):
    prompt_q = 'Examples: \n' + prompt_original + '\n\nFollowing Question: \n'
    for i, q in enumerate(q_list, start=9):
        prompt_q += f'Question_{i}:\n{q}\n'
    prompt_q += '\n'

    with get_openai_client(args) as client:
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "Answer each question step by step, adhering to the format shown in the examples provided. Start each response with 'Answer_' and introduce the final response with 'The answer is'. Do not repeat the question. Ensure that you respond to all the questions presented, regardless of their number."},
                    {"role": "user", "content": prompt_q}
                ],
                temperature=0,
                max_tokens=4096
                # response_format={ "type": "json_object" }
            )
            # breakpoint()
            ans_model = completion.choices[0].message.content
        except Exception as e:
            print(f"Error processing article {e}")
            print(completion)
            return process_question(args, q_list, a_list, prompt_original)

    ans_, _ = extract_ans(ans_model)
    ans_ = re.findall(r'(?<=:\n)(.*?)(?=\n\n|$)', ans_, re.DOTALL)
    ans_list = [cur.strip() for cur in ans_]

    result_str = ""
    for i, (q, a, ans) in enumerate(zip(q_list, a_list, ans_list)):
        result_str += f"Question: {q}\nA_model:Q_{i}\n{ans}\nA_gold:\n{a}\n\n"

    return result_str

def parallel_process(args, questions, answers, prompt_original, output_file):
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_question = {executor.submit(process_question, args, question, answer, prompt_original): question 
                              for question, answer in zip(questions, answers)}

        with open(output_file, 'a') as fd:
            for future in tqdm(as_completed(future_to_question), total=len(future_to_question), desc='Processing Questions'):
                result_str = future.result()
                fd.write(result_str)
                fd.flush()  # Ensure the content is written immediately

def main(args):
    prompt_original = open(args.prompt_path).read()
    output_file = args.output_path

    # Load GSM8K dataset
    gsm8k = load_dataset('gsm8k', 'main')
    questions = gsm8k['test']['question'][:args.question_limit]
    answers = gsm8k['test']['answer'][:args.question_limit]

    with open(output_file, 'w') as fd:
        fd.write('')

    # Split questions and answers into batches of size k
    batched_questions = [questions[i:i+args.k] for i in range(0, len(questions), args.k)]
    batched_answers = [answers[i:i+args.k] for i in range(0, len(answers), args.k)]

    parallel_process(args, batched_questions, batched_answers, prompt_original, output_file)
    
    print("*" * 50)
    print("Running configuration:")
    print("Number of questions to process in parallel: ", args.k)
    print("Prompt file path: ", args.prompt_path)
    print("Output file path: ", args.output_path)
    print("Dataset: GSM8K")
    _, _, _ = parse_pred_ans(output_file)
    print("*" * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process GSM8K questions in parallel.')
    parser.add_argument('--k', type=int, default=4, help='Number of questions to process in parallel')
    parser.add_argument('--prompt_path', type=str, default='./data/LongGenBench_GSM8K_prompt/LongGenBench_prompt.txt', help='Path to the prompt file')
    parser.add_argument('--output_path', type=str, default='./outputs/LongGenBench_GSM8K/LongGenBench_GSM8K_demo.txt', help='Path to the output file')
    parser.add_argument('--question_limit', type=int, default=1319, help='Limit to the number of questions to process')
    parser.add_argument('--api_key', type=str, default='your-api-key-here', help='OpenAI API Key')
    parser.add_argument('--model', type=str, default='gpt-3.5-0125', help='Model name')
    parser.add_argument('--api_endpoint', type=str, default='https://api.openai.com/v1/', help='API endpoint for processing questions')
    args = parser.parse_args()
    main(args)