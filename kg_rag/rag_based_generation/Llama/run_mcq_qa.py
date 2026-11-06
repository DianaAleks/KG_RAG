
'''
This script takes the MCQ style questions from the csv file and save the result as another csv file.
This script makes use of Llama model.
Before running this script, make sure to configure the filepaths in config.yaml file.
'''

from langchain import PromptTemplate, LLMChain
from kg_rag.utility import *
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#QUESTION_PATH = config_data["MCQ_PATH"]
QUESTION_PATH = "data/benchmark_data/mcq_questions_test2.csv"
SYSTEM_PROMPT = system_prompts["MCQ_QUESTION"]
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]
SAVE_PATH = config_data["SAVE_RESULTS_PATH"]
MODEL_NAME = config_data["LLAMA_MODEL_NAME"]
BRANCH_NAME = config_data["LLAMA_MODEL_BRANCH"]
CACHE_DIR = config_data["LLM_CACHE_DIR"]

#save_name = "_".join(MODEL_NAME.split("/")[-1].split("-"))+"_kg_rag_based_mcq_from_monarch_and_robokop_response.csv"
save_name = "_".join(MODEL_NAME.split("/")[-1].split("-"))+"_kg_rag_based_mcq_from_monarch_and_robokop_response_test.csv"

INSTRUCTION = "Context:\n\n{context} \n\nQuestion: {question}"

print("[SETUP] Loading Chroma vector store...", flush=True)
vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
print("[SETUP] Chroma vector store loaded.", flush=True)

print("[SETUP] Loading context embedding model...", flush=True)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
print("[SETUP] Context embedding model loaded.", flush=True)

print("[SETUP] Loading node context CSV...", flush=True)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)
print(f"[SETUP] Loaded {len(node_context_df)} node context rows.", flush=True)

edge_evidence = False


def main():
    start_time = time.time()
    print(f"[MODEL] Starting MCQ KG-RAG with {MODEL_NAME}", flush=True)

    # Use direct Transformers generation for Llama 3.2 1B.
    if MODEL_NAME == "meta-llama/Llama-3.2-1B-Instruct":
        print("[MODEL] Loading tokenizer...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            revision=BRANCH_NAME,
            cache_dir=CACHE_DIR
        )

        print("[MODEL] Loading Llama model weights...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            revision=BRANCH_NAME,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float32
        )
        model.eval()
        print("[MODEL] Llama model loaded successfully.", flush=True)
    else:
        # Preserve the original pipeline for other models.
        print("[MODEL] Loading Llama through original LangChain pipeline...", flush=True)
        llm = llama_model(MODEL_NAME, BRANCH_NAME, CACHE_DIR)
        print("[MODEL] Llama pipeline loaded.", flush=True)

        print("[MODEL] Preparing prompt and LLM chain...", flush=True)
        template = get_prompt(INSTRUCTION, SYSTEM_PROMPT)
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        print("[MODEL] LLM chain ready.", flush=True)

    print(f"[DATA] Loading questions from {QUESTION_PATH}...", flush=True)
    question_df = pd.read_csv(QUESTION_PATH)
    print(f"[DATA] Loaded {len(question_df)} questions.", flush=True)

    answer_list = []

    for index, row in question_df.iterrows():
        question = row["text"]
        print(f"\n[QUESTION {index + 1}/{len(question_df)}] Starting...", flush=True)

        print(f"[QUESTION {index + 1}] Retrieving KG context...", flush=True)
        context = retrieve_context(
            question,
            vectorstore,
            embedding_function_for_context_retrieval,
            node_context_df,
            CONTEXT_VOLUME,
            QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
            QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY,
            edge_evidence,
            api=True,
            local_entity_extraction=True
        )
        print(
            f"[QUESTION {index + 1}] Context retrieved: {len(context)} characters.",
            flush=True
        )

        print(f"[QUESTION {index + 1}] Generating answer...", flush=True)

        if MODEL_NAME == "meta-llama/Llama-3.2-1B-Instruct":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": INSTRUCTION.format(
                        context=context,
                        question=question
                    )
                }
            ]

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            )
            inputs = {
                key: value.to(model.device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    use_cache=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            new_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
            output = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True
            ).strip()
        else:
            output = llm_chain.run(context=context, question=question)

        print(f"[QUESTION {index + 1}] Answer generated: {output}", flush=True)
        answer_list.append((row["text"], row["correct_node"], output))

    print("\n[SAVE] Writing results to CSV...", flush=True)
    answer_df = pd.DataFrame(
        answer_list,
        columns=["question", "correct_answer", "llm_answer"]
    )
    output_path = os.path.join(SAVE_PATH, save_name)
    answer_df.to_csv(
        output_path,
        index=False,
        header=True
    )
    print(f"[SAVE] Results saved to {output_path}", flush=True)
    print("Completed in {} min".format((time.time() - start_time) / 60), flush=True)


if __name__ == "__main__":
    main()
