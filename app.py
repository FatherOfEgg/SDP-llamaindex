import os
from io import StringIO
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage
)
from llama_parse import LlamaParse
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
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

import plotly.graph_objects as go
import pandas as pd

load_dotenv()

llm = Groq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
    # model_name="google/tapas-base-finetuned-wtq"
)
# Settings.embed_model = HuggingFaceInferenceAPI(
#     model_name="HuggingFaceH4/zephyr-7b-alpha", token=os.getenv("HUGGING_FACE_TOKEN")
# )

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
        "Instructions:"
        "1. Always double check your answer to make sure there wasn't a more correct answer."
        " Use the following to double check your answer:\n"
        "{context_str}"
        "2. Always try to respond by outputting it in proper CSV format that only has 2 columns."
        "3. Must Include labels."
        "4. Don't preface the CSV with anything, only have the CSV."
        "5. Delimit using commas. Do not use quotes."
        " Each row in the CSV is separated by new lines only."
        "6. Make sure the CSV only has 2 columns. Sort the data based on the prompt."
        "7. If you want to say 1 sentence that pertains to the prompt, then"
        " have the CSV first, and the sentence second. As well as"
        " delimiting the CSV and the sentence with |D|."
        "8. If you can't respond with CSV, respond with 1 to 2 short sentences."
        "9. The CSV must contain only 2 columns, no more no less."
        "10. Include labels for the CSV in relevant responses."
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

        res = str(chat_engine.chat(text)).split("|D|")

        if len(res) == 1:
            res.append("")

        print(res[0])

        data = StringIO(res[0])

        df = pd.read_csv(data)

        labels = res[0].splitlines()[0]
        print(labels)
        [x_col, y_col] = labels.split(",")

        response = [
            html.H5(res[1], style={"text-align": "left"}),
            dcc.Graph(
                figure=go.Figure(
                    data=[
                        go.Scatter(
                            x=df[x_col],
                            y=df[y_col]
                        )
                    ]
                )
            )
        ]

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
