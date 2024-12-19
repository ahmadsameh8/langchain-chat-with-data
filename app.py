from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

llm = HuggingFacePipeline.from_model_id(
    model_id="tiiuae/Falcon3-1B-Instruct",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 1000},
    # device = "cpu"
    
    
    # device_map='cpu'
)

# Initialize the vector database
persist_directory = 'docs/chroma/'
embedding = HuggingFaceEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Initialize Conversational Retrieval Chain
retriever = vectordb.as_retriever(search_kwargs={"k": 1})
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
)

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        # Get the question from the request
        data = request.get_json()
        question = data.get("question", "")

        if not question:
            return jsonify({"error": "Question is required."}), 400

        # Get the answer using the Conversational Retrieval Chain
        result = qa({"question": question})
        output = result["answer"]
        answer = output.split("Helpful Answer:")[-1].strip()

        # Return the answer as JSON
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
