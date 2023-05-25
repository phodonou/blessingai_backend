import os
from langchain.embeddings import OpenAIEmbeddings

import pprint
from tree_sitter import Language, Parser
import pandas as pd
import openai
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
pp = pprint.PrettyPrinter(indent=4)

Language.build_library(
  # Store the library in the `build` directory
  'build/my-languages.so',

  # Include one or more languages
  [
    'vendor/tree-sitter-dart',
#     'vendor/tree-sitter-yaml',
#     'vendor/tree-sitter-json',
#     'vendor/tree-sitter-markdown'
  ]
)

DART_LANGUAGE = Language('build/my-languages.so', 'dart')
# YAML_LANGUAGE = Language('build/my-languages.so', 'yaml')
# JSON_LANGUAGE = Language('build/my-languages.so', 'json')
# MARKDOWN_LANGUAGE = Language('build/my-languages.so', 'markdown')

parser = Parser()

def read_file(path):
    with open(path, encoding="ISO-8859-1") as f:
        contents = f.read()
        return contents
    
def chunk_source(path, rows = []):
    dir_list = os.listdir(path)
    
    for dir_path in dir_list:
        full_path = path + "/" + dir_path
        if(os.path.isfile(full_path)):
            source_code = str(read_file(full_path))
            
            if(dir_path.endswith('.dart')):
                parser.set_language(DART_LANGUAGE)
#             elif(dir_path.endswith('yaml')):
#                 print("yaml " + dir_path)
#                 parser.set_language(YAML_LANGUAGE)
#             elif(dir_path.endswith('json')):
#                 print("json " + dir_path)
#                 parser.set_language(JSON_LANGUAGE)
#             elif(dir_path.endswith('md')):
#                 print("md " + dir_path)
#                 parser.set_language(MARKDOWN_LANGUAGE)
            else:
                continue
            
            tree = parser.parse(bytes(source_code, "utf8"))
            
            # Traverse the AST and extract function information
            for node in tree.root_node.children:
                new_row = [
                    node.type, 
                    source_code[node.start_byte: node.end_byte].strip(),
                    dir_path
                ]
                rows.append(new_row)
            
#             print("saved " + full_path)
    
        else:
            chunk_source(full_path, rows)
    return rows

def embed(df):
    embeddings_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
    )
    embeddings = embeddings_model.embed_documents(df['node_source'])
    return embeddings


# rows = chunk_source("/Users/princehodonou/Development/flutter")
# df = pd.DataFrame(rows, columns=['node_type', 'node_source', 'relative_path'])
# print("DONE CHUNKING. LENGTH IS::: ", len(df))
# embeddings = embed(df)
# print("DONE EMBEDDING. LENGTH IS::: ", len(embeddings))
# df['source_embedding'] = embeddings
# SAVE_PATH = "./embeddings/flutter.pkl"
# df.to_pickle(SAVE_PATH)
# print("EMBEDDING SAVED")

df = pd.read_pickle('./embeddings/flutter.pkl')
