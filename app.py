import os
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage
)
from llama_parse import LlamaParse
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from dash import (
    Dash,
    html,
    Input,
    Output,
    callback,
    dcc
)

load_dotenv()

# You can use a different llm/embedded model and see if it works better

llm = Groq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

PERSIST_DIR = "./storage"

if not os.path.exists(PERSIST_DIR):
    documents = LlamaParse().load_data("./data/Registered-Cats-By-Breed.2022.pdf")
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()
# response = query_engine.query("What are the top 5 popular breeds?")

# print(response)


# This is a pretty simple and janky Dash setup, works for now
app = Dash()

app.layout = html.Div(
    [
        dcc.Input(id="input1", type="text", placeholder="Enter prompt", debounce=True),
        html.Div(id="output"),
    ]
)

@callback(
    Output("output", "children"),
    Output("input1", "value"), # https://stackoverflow.com/q/72351437
    Input("input1", "value"),
)
def update_output(input1):
    response = ""
    if input1:
        response = query_engine.query(input1)

    return f"Response: {response}", ""

if __name__ == '__main__':
    app.run(debug=True)
