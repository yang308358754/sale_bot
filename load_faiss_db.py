from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
import os

import load_faiss_db

class LoadFaiss(object):

    def __init__(self, db_name="faiss_estate_sale_index", source_file="real_estate_sales_data.txt"):
        self.db_name = db_name
        self.source_name = source_file

        if os.path.exists(db_name):
            embeddings = OpenAIEmbeddings()
            self.db = FAISS.load_local(db_name, embeddings)
        else:
            self.import_to_faiss2()

    #初始化把数据导入到faiss
    def import_to_faiss(self):
        # 实例化文档加载器
        loader = TextLoader(self.source_name)
        # 加载文档
        documents = loader.load()

        # 实例化文本分割器
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        # 分割文本
        docs = text_splitter.split_documents(documents)

        # OpenAI Embedding 模型
        embeddings = OpenAIEmbeddings()

        # FAISS 向量数据库，使用 docs 的向量作为初始化存储
        self.db = FAISS.from_documents(docs, embeddings)

        self.db.save_local(self.db_name)

    def import_to_faiss2(self):
        with open(self.source_name) as f:
            real_estate_sales = f.read()
        text_splitter = CharacterTextSplitter(
            separator=r'\d+\.',
            chunk_size=100,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=True,
        )
        docs = text_splitter.create_documents([real_estate_sales])
        self.db = FAISS.from_documents(docs, OpenAIEmbeddings())
        self.db.save_local(self.db_name)


    #实例化faissDb
    def getInstanceDb(self):
        # OpenAI Embedding 模型
        embeddings = OpenAIEmbeddings()
        self.db = FAISS.load_local(self.db_name, embeddings)
        return self.db

    #在faiss中查询数据
    def faiss_query(self, query):
        docs = self.db.similarity_search(query)
        for doc in docs:
            print(doc.page_content + "\n")

    #查询top K答案
    def faiss_query_topk(self, query, k):
        # 实例化一个 TopK Retriever
        topK_retriever = self.db.as_retriever(search_kwargs={"k": k})
        type(topK_retriever)
        docsk = topK_retriever.get_relevant_documents(query)
        for doc in docsk:
            print(doc.page_content + "\n")

    #阀值
    def faiss_query_score(self, query, score):
        # 实例化一个 similarity_score_threshold Retriever
        retriever = self.db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": score}
        )

        docs = retriever.get_relevant_documents(query)
        if len(docs) > 0:
            for doc in docs:
                print(doc.page_content + "\n")
        else:
            print("no information")

if __name__ == '__main__':
   loadFaiss = load_faiss_db.LoadFaiss(db_name="faiss_edu_sale_index", source_file="real_edu_sales_data.txt")

   # loadFaiss.faiss_query("小区吵不吵？")
   loadFaiss.faiss_query_topk("机构有兴趣班吗？", 1)
   #loadFaiss.faiss_query_score("小区吵吗？", 0.8)



