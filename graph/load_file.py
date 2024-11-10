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
# load_dotenv()
# file_path="data/book1.csv"
# url= os.getenv("BOLT")
# connect= (f"{os.getenv("USER_NEO4J")}",f"{os.getenv("PASSWORD_NEO4J")}")
# df = pd.read_csv(file_path,encoding="utf-8")
def readfile_clean(filepath: str):
    # Đọc file CSV
    df = pd.read_csv(filepath, encoding="utf-8")
    df['product_cleaned'] = df['product'].str.extract(r'^(\w.*\w)$')
    df['product_cleaned'] = df['product_cleaned'].str.strip()
    df['product_cleaned'] = df['product_cleaned'].apply(lambda x: re.sub(r'\s+', ' ', str(x)) if isinstance(x, str) else str(x))
    df['price_float'] = df['price'].apply(lambda x: float(str(x)) )
    df['descript_cleaned'] = df['descript'].str.strip()
    df['descript_cleaned'] = df['descript_cleaned'].apply(lambda x: re.sub(r'\s+'," ",str(x)) if isinstance(x,str)else x)
    # prodct_cleaned , price_float , descript_clean
    return df 
    #combine embedding
def  combine_product_script(row):
    descript = "Product Title: "+str(row["product_cleaned"])+"\n"
    descript += "Product Descript: "+row["descript_cleaned"]
    return  descript
def connect_graph(file_local,auth:tuple):#success
    graph = Graph(file_local,auth=auth)
    graph.run("MATCH (P) DETACH DELETE P")
    return graph

def create_knowledge_graph(df, graph):
    df['complete_descript'] = df.apply(combine_product_script,axis=1) 
    try:
        # Đảm bảo các constraint đã được tạo
        graph.run("CREATE CONSTRAINT product_id IF NOT EXISTS FOR (p:Product) REQUIRE p.uniq_id IS UNIQUE")
        graph.run("CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE")
    except Exception as e:
        # For Neo4j 4.x, có thể có lỗi nếu đã có constraint
        try:
            graph.run("CREATE CONSTRAINT ON (p:Product) ASSERT p.uniq_id IS UNIQUE")
            graph.run("CREATE CONSTRAINT ON (c:Category) ASSERT c.name IS UNIQUE")
        except Exception as e:
            print(f"Warning: Could not create constraints: {e}")
    
    for _, row in df.iterrows():
        # Tạo node Product
        product = Node("Product",
                       uniq_id=row['unique_id'],
                       name=row['product_cleaned'],
                       factory=row['factory'],
                       price=row['price_float'],
                       author=row['author'],
                       descript=row['descript_cleaned'],
                       complete=row['complete_descript'],
                       description_embedding=None
                       )
        
        factory = Node("Factory", name=row['factory'])
        
        category = Node("Category", name=row['category'])

        graph.merge(product, "Product", "uniq_id")
        graph.merge(factory, "Factory", "name")
        graph.merge(category, "Category", "name")

        factory_in_relationship = Relationship(product, "FACTORY_IN", factory)
        graph.merge(factory_in_relationship)
        
        kind_relationship = Relationship(product, "KIND", category)
        graph.merge(kind_relationship)
        # Create Category nodes from hierarchy
                # categories = row['amazon_category_and_sub_category'].split(' > ')
                # previous_category = None
                # for cat in categories:
                #     category = Node("Category", name=cat)
                #     graph.merge(category, "Category", "name")
                    
                #     if previous_category:
                #         # Create hierarchical relationship between categories
                #         rel = Relationship(previous_category, "HAS_SUBCATEGORY", category)
                #         graph.merge(rel)
                #     previous_category = category
# df = readfile_clean(file_path)
# graph = connect_graph(url,connect) ##  delete  database 
# create_knowledge_graph(df,graph)
# print(df)  
# #----------------------test------------------------ 
def run_query_with_viz(graph,query,title,viz_query=None):
    print("---------------title-----------")
    results = graph.run(query).data()
    dftemp = pd.DataFrame(results)
    display(dftemp)
    viz_results = graph.run(viz_query or query).data()
    print(f"\nNumber of visualization records: {len(viz_results)}")
    print(viz_results)
    for i in viz_results:
            factory_name = i.get('factory')  # Lấy tên nhà máy
            product_name = i.get('product')  # Lấy tên sản phẩm
            price = i.get('price')  # Lấy giá sản phẩm
            print(f"Factory: {factory_name}, Product: {product_name}, Price: {price}")

# query = """
# MATCH (p:Product)-[:FACTORY_IN]->(q:Factory)
# ORDER BY p.price DESC 
# RETURN q.name as factory , p.name as product , p.price as price 
# """
# run_query_with_viz(graph,query,"price  exspesive most")
# api_key = os.getenv("GOOGLE_API_KEY")


@sleep_and_retry
@limits(calls=1500, period=60)
def get_embedding(text):
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    try:
        genai.configure(api_key=api_key)
        result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document",
        title="Embedding of single string")
        return result["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None
def add_embedding_product(graph,batch_size=50):
    total_query="""
    MATCH (p:Product)
    WHERE p.description_embedding IS NULL 
    AND  p.descript IS NOT NULL
    RETURN count(p) AS total
    """
    result_total= graph.run(total_query).data()
    total_process = result_total[0]['total'] if  result_total else 0
    total_processed =0
    print('total row data:',total_process)
    while True:
        query="""
        MATCH(p:Product)
        WHERE p.description_embedding IS NULL 
        AND p.complete IS NOT NULL
        RETURN p.uniq_id as  id  , p.complete as  descript
        LIMIT  $batch_size
        """
        products = graph.run(query,parameters={'batch_size':batch_size}).data()
        if not products:
            break
        for product  in products:
            try:
                if(product['descript']):
                    embedding = get_embedding(product['descript'])
                    if embedding:
                        graph.run("""
                        MATCH (p:Product {uniq_id:$id})
                        SET p.description_embedding = $embedding
                        """,parameters={'id':product['id'],'embedding':embedding})
                    total_processed+=1
            except Exception as e:
                print(f"Error processing product {product['id']}: {e}")
        time.sleep(1)
    print("Total_Processed :",total_processed)
    print("\nVerifying embeddings:")
    result = graph.run("""
    MATCH (p:Product)
    WHERE p.description_embedding IS NOT NULL
    RETURN count(p) as count
    """).data()
    print(f"Products with embeddings: {result[0]['count']}")

# def semantic_search(query_text,n=5):
#     query_embedding = get_embedding(query_text)
#     if not query_embedding:
#         print("Failed to get query embedding")
#         return []
#     print(f"Query embedding length: {len(query_embedding)}")
#     #cosine similarity =  A*B/||A||*||B||
#     ##||A|| = sqrt(a1^2...an)
#     ## 
#     # result = graph.run(
#     #     """
#     #     MATCH (p:Product)
#     #     WHERE  p.complete IS NOT NULL
#     #     WITH p,
#     #     reduce(dot=0.0,i in range(0,size(p.description_embedding))|

#     #         dot + description_ebedding[i] * $ebedding[i]
#     #     )/
#     #     (sqrt(reduce(a=0.0,i in range(0,size(p.description_embedding))|
#     #     a + description_embedding[i]*description_embedding[i]))*
#     #     sqrt(reduce(b=0.0,i in range(0,size(p.$embedding))|
#     #     b + $embedding[i]*$embedding[i]))) AS similarity
#     #     WHERE similarity>0
#     #     RETURN 
#     #         p.name as  name,
#     #         p.descript as descript,
#     #         p.price as price,
#     #         p.description_embedding as embedding,
#     #         similarity as score
#     #     LIMIT $n
#     #     """
#     # ,parameters={"embedding":query_embedding,'n':n}).data()
#     result = graph.run(
#         '''
#         MATCH (p:Product)
#         WHERE p.complete IS NOT NULL
#         WITH p,
#         reduce(dot=0.0, i in range(0, size(p.description_embedding))|
#             dot + description_embedding[i] * $embedding[i]
#         )/
#         (sqrt(reduce(a=0.0, i in range(0, size(p.description_embedding))|
#             a + description_embedding[i] * description_embedding[i])) *
#         sqrt(reduce(b=0.0, i in range(0, size($embedding))|
#             b + $embedding[i] * $embedding[i]))) AS similarity
#         WHERE similarity > 0
#         RETURN 
#             p.name as name,
#             p.descript as descript,
#             p.price as price,
#             p.description_embedding as embedding,
#             similarity as score
#         LIMIT $n
#         '''
#     , parameters={"embedding": query_embedding, 'n': n}).data()

#     return result
def semantic_search(graph,query_text, n=5):
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        print("Failed to get query embedding")
        return []

    print(f"Query embedding length: {len(query_embedding)}")

    # Cypher query to compute cosine similarity between query and product embeddings
    result = graph.run(
        '''
        MATCH (p:Product)
        WHERE p.complete IS NOT NULL AND p.description_embedding IS NOT NULL
        WITH p,
        reduce(dot=0.0, i in range(0, size(p.description_embedding)-1) |
            dot + p.description_embedding[i] * $embedding[i]
        ) /
        (sqrt(reduce(a=0.0, i in range(0, size(p.description_embedding)-1) |
            a + p.description_embedding[i] * p.description_embedding[i])) *
        sqrt(reduce(b=0.0, i in range(0, size($embedding)-1) |
            b + $embedding[i] * $embedding[i]))) AS similarity
        WHERE similarity > 0
        RETURN 
            p.name AS name,
            p.descript AS descript,
            p.price AS price,
            p.description_embedding AS embedding,
            similarity AS score
        ORDER BY similarity DESC
        LIMIT $n
        ''',
        parameters={"embedding": query_embedding, 'n': n}
    ).data()

    print(f"Found {len(result)} products with similarity > 0.")
    return result

def print_result(result):
    for i in  result:
        print(f"\n Product :",i['name'])
        print(f"descript :",i['descript'])
        print(f"price :",i['price'])
        print(f"description_embedding :",i['embedding'][:10],"...TRIMED")
# add_embedding_product(graph)
# result =semantic_search("Bậc Thầy MÔI GIỚI ĐỊA ỐC",n=2)

