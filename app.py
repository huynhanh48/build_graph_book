from  graph.load_file import readfile_clean,combine_product_script,connect_graph,create_knowledge_graph,get_embedding,add_embedding_product,semantic_search,print_result
import  pandas as pd
from  dotenv import load_dotenv
from py2neo import  Graph,Node,Relationship
import google.generativeai as genai
from ratelimit import limits, sleep_and_retry
import re
import os
import time
from dotenv import load_dotenv
from IPython.display import display
api_key = os.getenv("GOOGLE_API_KEY")
file_path="data/book1.csv"
url= os.getenv("BOLT")
connect= (f"{os.getenv("USER_NEO4J")}",f"{os.getenv("PASSWORD_NEO4J")}")
df=readfile_clean(file_path)
graph=connect_graph(url,connect)
create_knowledge_graph(df,graph)
add_embedding_product(graph)
result= semantic_search(graph,"Kim Bình Mai",n=2)
print_result(result)