from langchain import PromptTemplate, LLMChain
from kg_rag.utility import *
import torch

QUESTION_PATH = config_data["MCQ_PATH"]
#QUESTION_PATH = "/workspaces/KG_RAG/data/benchmark_data/mcq_questions_test2.csv"
SYSTEM_PROMPT = system_prompts["MCQ_QUESTION_PROMPT_BASED"]
SAVE_PATH = config_data["SAVE_RESULTS_PATH"]
MODEL_NAME = config_data["LLAMA_MODEL_NAME"]
BRANCH_NAME = config_data["LLAMA_MODEL_BRANCH"]
CACHE_DIR = config_data["LLM_CACHE_DIR"]



save_name = "_".join(MODEL_NAME.split("/")[-1].split("-"))+"_prompt_based_two_hop_mcq_from_monarch_and_robokop_response.csv"
#save_name = "_".join(MODEL_NAME.split("/")[-1].split("-"))+"_prompt_based_two_hop_mcq_from_monarch_and_robokop_response_test.csv"

INSTRUCTION = "Question: {question}"



def main():
    start_time = time.time()
    question_df = pd.read_csv(QUESTION_PATH)
    answer_list = []

    # ADDED: Native chat-template path for Llama 3.2 Instruct.
    if MODEL_NAME == "meta-llama/Llama-3.2-1B-Instruct":
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            revision=BRANCH_NAME,
            cache_dir=CACHE_DIR
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            revision=BRANCH_NAME,
            cache_dir=CACHE_DIR,
            device_map="auto",
            torch_dtype=torch.float32
        )
        model.eval()

        for index, row in question_df.iterrows():
            question = row["text"]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": INSTRUCTION.format(question=question)}
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            new_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
            output = tokenizer.decode(new_tokens, skip_special_tokens=True)
            answer_list.append((question, row["correct_node"], output))

    # Original GitHub code path for other models.
    else:
        llm = llama_model(MODEL_NAME, BRANCH_NAME, CACHE_DIR)
        template = get_prompt(INSTRUCTION, SYSTEM_PROMPT)
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        for index, row in question_df.iterrows():
            question = row["text"]
            output = llm_chain.run(question)
            answer_list.append((row["text"], row["correct_node"], output))

    answer_df = pd.DataFrame(answer_list, columns=["question", "correct_answer", "llm_answer"])
    answer_df.to_csv(os.path.join(SAVE_PATH, save_name), index=False, header=True)
    print("Completed in {} min".format((time.time()-start_time)/60))

    

if __name__ == "__main__":
    main()
