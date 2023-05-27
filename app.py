from flask import Flask, request
from flask_socketio import SocketIO, send, emit

import os

from tree_sitter import Language, Parser
import pandas as pd
import openai
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity


EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*" )

@socketio.on('query')
def query(data):
    df = pd.read_pickle('./embeddings/%s.pkl' % data['projectId'])
    answer = ask(df, data['query'], lambda x: updatestep(emit, x), n=11)
    emit('query', {
        'type': 'answer',
        'data': {
            "message": answer
        },
        'is_final': True,
    })


# @app.route('/ask',  methods=['GET'])
# def ask_query():
    # args = request.args
    # query = args.get("query")
    # df = pd.read_parquet('./embeddings/metalink_mobile.parquet')
    # return ask(df, query, n=10)

def updatestep(emit, message):
    emit('query', {
        'type': 'reasoning_step',
        'data': {
            "message": message
        },
        'is_final': False,
    })

def ask(df, query, updatestep, n=3):
    updatestep("UNDERSTANDING QUERY")
    response = openai.ChatCompletion.create(
        temperature=0,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You respoond with code search that pertains to asked question."},
            {"role": "user", "content": "Respond with code search only: What framework does this appear to use?"},
            {"role": "assistant", "content": "framework and tools"},
            {"role": "user", "content": "Respond with code search only: How does the app start up?"},
            {"role": "assistant", "content": "Start initialization"},
            {"role": "user", "content": "Respond with code search only: What javascript framework is being used?"},
            {"role": "assistant", "content": "Javascript framework used"},
            {"role": "user", "content": "Respond with code search only: How is user authentication handled?"},
            {"role": "assistant", "content": "User authentication"},
            {"role": "user", "content": "Respond with code search only: How is routing handled in the application?"},
            {"role": "assistant", "content": "Routing"},
            {"role": "user", "content": "Respond with code search only: How does the application handle errors and exceptions?"},
            {"role": "assistant", "content": "Errors and exception handling"},
            {"role": "user", "content": "Respond with code search only: " + query},
        ],
    )
    query_search = response["choices"][0]["message"]["content"]
    updatestep("QUERY UNDERSTOOD: " + query_search)
    updatestep("SEARCHING CODEBASE")
    query_embedding = get_embedding(query_search, engine=EMBEDDING_MODEL)
    df['similarities'] = df.source_embedding.apply(
        lambda x: cosine_similarity(x, query_embedding))
    res = df.sort_values('similarities', ascending=False).head(n)

    introduction = 'Use the below code snippets to answer questions about the project. If the answer cannot be found, write "I could not find an answer.". \n \n'
    question = f"\n\nQuestion: {query}. Provide relevant code snippets."
    message = introduction

    for r in res.iterrows():
        message += r[1]['node_source'] + "\n"

    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions about codebases"},
        {"role": "user", "content": message + question},
    ]

    updatestep("TALKING TO CHAT GPT")
    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
