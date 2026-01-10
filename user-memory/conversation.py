import ollama
import logging

memory_file = "memory.txt"
model = "llama3.2" # make sure ollama is running and llama3.2 is available

logger = logging.getLogger(__name__)

def get_user_input(): 
    inp = input("You: ")
    return inp.strip()

def save_to_memory(memory_object): 
    parsed_list = eval(memory_object.strip())

    with open(memory_file, "r") as rf, open(memory_file, "a+") as f: 
        content = rf.read()
        for item in parsed_list:
            if item in content:
                continue
            f.write(item + "\n")

def load_memory():
    with open(memory_file, "r") as f: 
        content = f.read()

    content = content.strip("\n")

    return content.replace("\n", ",")


def main()
    messages = [
        {"role": "system", "content": f"You are a helpful assistant that can answer questions and help with tasks. Always keep short and brief answers. Take a look at memory from previous conversations if needed. Memory: {load_memory()}. Don't treat these pieces of memory as literals in your conversation, they are just information you have about the user."},
    ]

    memory_queue = [
        {"role": "system", "content": "You are a helpful assistant to store the key details of the conversation in a memory. In each turn, you will be given a conversation history and you are expected to extract key details from the conversation history to be used later either in the form of a graph representation or key words that you can pick up or whichever method you think is the most relevant for agent memory just output that format. Don't lose the key information from the conversation history and keep it brief. Return your answer in a list. The format example could be something like ```['likes apples', 'has never rock climbed', ...]```. Only include elements items in the list if the item teaches us something about the user, don't include generic information that will not be useful later no. Essentially, the whole idea is to keep some previous memory of the user based on the conversation they had. Don't include any additional note or information apart from this list as I want it to be easily parsible. Just provide the list and that's fine."}
    ]

    while True: 
        # --------------------------------------------------------
        # Let's implement the basic conversation logic here
        # --------------------------------------------------------

        try: 
            prompt = get_user_input()
            messages.append({
                "role": "user", "content": prompt
            })
            response = ollama.chat(model=model, messages=messages)
            model_answer = response.message.content.replace("\n\n", "\n")
            messages.append({
                "role": "assistant", "content": model_answer
            })
            logger.info(f"Model: {model_answer}\n")
        except Exception as e: 
            logger.error(f"Model couldn't answer: {e}")
            logger.error(f"Response: {response}")


        # --------------------------------------------------------
        # Let's implement user memory logic here
        # --------------------------------------------------------

        try: 
            memory_queue.append({"role": "user", "content": str(messages)})
            mem_res = ollama.chat(model=model, messages=memory_queue)
            memory_object = mem_res.message.content
            save_to_memory(memory_object)
        except Exception as e: 
            logger.error(f"Memory couldn't be saved: {e}")
            logger.error(f"Memory response: {mem_res}")


if __name__ == "__main__": 
    main()