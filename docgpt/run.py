#!/usr/bin/env python3

# modules
pass


# functions
def configure():
    """
    Load documents, split them & create database.
    """

    return


def configure_model():
    """ / """

    # enable verbose mode for LLM callback to standard out
    # callbacks = [StreamingStdOutCallbackHandler()]
    callbacks = list()

    llm = GPT4All(model="models/ggml-gpt4all-j-v1.3-groovy.bin", max_tokens=1_000, backend='gptj', n_batch=8, callbacks=callbacks, verbose=False)

    return llm


def summarize():
    """ Summarize documents using docgpt. """

    return


def retrieval_qa(db):
    """ Interact with documents using docgpt. """

    # set up db as retriever to look for K amount of sources from
    # embeddings that will be used to answer a question
    retriever = db.as_retriever(search_kwargs={"k": 4})
    
    # create a question-answer retrieval chain to interact thru queries
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=False)

    while True:
        print("> Query:")
        query = input()
    
        if query.lower() == "exit":
            break
            
        if query.strip() == "":
            print("\nPlease enter a valid query...\n\n")
            continue
    
        start = time.time()
    
        # get answer using RetrievalQA chain
        output = qa(query)
        answer = output.get("result")
    
        end = time.time()
    
        # display answer
        print("\n> Answer:")
        print(answer)
        print(f"\nETA: {round(end - start, 2)} (s)\n")

    return


if __name__ == "__main__":
    setup()
    summarization()
    retrieval()