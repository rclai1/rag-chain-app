from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document
import os
from chroma_utils import vectorstore
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

output_parser = StrOutputParser()

qa_prompt = ChatPromptTemplate.from_template(

        """
        <s> [Instructions] You are an assisant for question-answering tasks.
        Use the following pieces of retrieved context to 
        answer the question, as well as any prior knowledge you may have on the topic. Combine the chat history and
        follow up question into a standalone question, if 
        the chat history is relevant to the question. If 
        you don't know the answer, say you don't know. 
        If the chat history is not relevant to the question,
        do not focus on it.
        If you don't know the answer, then reply, No Context availabel for this question {input}. [/Instructions] </s>
        [Instructions] Question: {input}
        Context: {context}
        Chat History: {chat_history}
        Answer: [/Instructions]
        """        
    )

def Rag_chain(model = "llama3" ):
    model = ChatOllama(model=model)
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    return chain