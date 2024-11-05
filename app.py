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
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.chat_engine.types import ChatMode

from dash import (
    Dash,
    html,
    Input,
    Output,
    callback,
    dcc,
    State
)

load_dotenv()

llm = Groq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

PERSIST_DIR = "./storage"

if not os.path.exists(PERSIST_DIR):
    documents = LlamaParse().load_data([
        "./data/Registered-Cats-By-Breed.2022.pdf",
        "./data/bonebase bvtv.docx"
    ])
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# chat_store = SimpleChatStore.from_persist_path(
#     persist_path="chat_store.json"
# )

memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    # chat_store=chat_store,
    chat_store_key="user1",
)

chat_engine = index.as_chat_engine(
    chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT,
    memory=memory,
    llm=llm,
    system_prompt=(
        "1. Always try to respond by outputting it in CSV format. Do not use quotes"
        "2. Sort the data based on the prompt."
        "3. You are allowed to say 1 sentence that pertains to the CSV and prompt."
        "4. If you can't respond with CSV, respond with 1 to 2 short sentences."
    ),
    verbose=False
)

app = Dash()

conv_hist = []

app.layout = html.Div([
    html.H3("BoneGPT", style={"text-align": "center"}),
    html.Div([
        html.Div([
            html.Br(),
            html.Div(id="conversation")],
                 id="chat",
                 style={"width": "400px", "margin": "0 auto"}
                 ),
        html.Div([
            html.Table([
                html.Tr([
                    html.Td([dcc.Input(id="prompt", placeholder="Enter a prompt", type="text")],
                            style={"valign": "middle"}),
                    html.Td([html.Button("Send", id="send_button", type="submit")],
                            style={"valign": "middle"})
                    ])
                ])],
                id="user-input",
                style={"width": "325px", "margin": "0 auto"}),
    ])
])

@callback(
    Output(component_id='conversation', component_property='children'),
    Input(component_id='send_button', component_property='n_clicks'),
    State(component_id='prompt', component_property='value')
)
def update_conversation(click, text):
    global conv_hist

    if click and click > 0:
        prompt = [html.H5(text, style={"text-align": "right"})]
        response = [html.H5(str(chat_engine.chat(text)), style={"text-align": "left"})]

        conv_hist += prompt + response

        # chat_store.persist(persist_path="chat_store.json")

        return conv_hist
    else:
        return ""

@callback(
    Output(component_id='prompt', component_property='value'),
    Input(component_id='conversation', component_property='children')
)
def clear_prompt(_):
    return ""

if __name__ == "__main__":
    app.run(debug=True)
