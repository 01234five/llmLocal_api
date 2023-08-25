

import torch
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
import Constants as _constants
import GlobalFunctions as global_functions
import Server.Api as ServerApi

def main(_device_type, show_sources):
    print(f"Running on: {_device_type}")
    print(f"Display Source Documents set to: {show_sources}")
    # load the vectorstore
    db = global_functions.DatabaseReturn(_constants.PERSIST_DIRECTORY,HuggingFaceInstructEmbeddings(model_name=_constants.EMBEDDING_MODEL_NAME, model_kwargs={"device": _device_type}),_constants.CHROMA_SETTINGS) 
    retriever = db.as_retriever()
    model_id = "TheBloke/Llama-2-7B-Chat-GGML"
    model_basename = "llama-2-7b-chat.ggmlv3.q4_0.bin"
    llm = global_functions.LoadModel(_device_type, model_id=model_id, model_basename=model_basename)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    instance_server_api = ServerApi.API(qa)
    instance_server_api.ServerStart()



#To use on a local terminal instead of an API use this:
def QuestionsAndAnswersLoop(_qa,_show_sources):
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        # Get the answer from the chain
        res = _qa(query)
        answer, docs = res["result"], res["source_documents"]

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        if _show_sources: 
            print("----------------------------------SOURCE DOCUMENTS---------------------------")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")

if __name__ == "__main__":
    device_type=""
    if torch.cuda.is_available():
        device_type="cuda"  
    else: 
        device_type="cpu",
    main(device_type,True)

