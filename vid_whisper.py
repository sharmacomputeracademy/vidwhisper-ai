from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# ------------------ LOAD ENV ------------------
load_dotenv()

class VidWhisper:
    def __init__(self, video_id):
        self.video_id = video_id
        self.persist_directory = f"./db_{video_id}"
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.full_transcript_path = os.path.join(self.persist_directory, "full_transcript.txt")
        self.full_translated_text = ""
        
        # Initialize LLMs
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # 1. Translation Chain
        translation_prompt = ChatPromptTemplate.from_template(
            "Translate the following text to English, preserving technical terms:\n\n{text}"
        )
        self.translation_chain = translation_prompt | self.llm | StrOutputParser()
        
        # 2. Router Chain (LLM-based intent detection)
        router_prompt = ChatPromptTemplate.from_template(
            """Classify the following user query into one of two categories:
            
            1. 'SUMMARY': If the user is asking for a general overview, a summary of the whole video, or "what is this about".
            2. 'SPECIFIC': If the user is asking a specific question, a technical "why/how", or comparing concepts (e.g. "X vs Y").
            
            Respond only with 'SUMMARY' or 'SPECIFIC'.
            
            Query: {query}
            Intent:"""
        )
        self.router_chain = router_prompt | self.llm | StrOutputParser()
        
        # 3. Contextualize Question Chain (Re-phrasing with History)
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{query}"),
            ]
        )
        self.contextualize_q_chain = contextualize_q_prompt | self.llm | StrOutputParser()

    def is_indexed(self):
        """Check if the video is already indexed."""
        return os.path.exists(self.persist_directory) and os.listdir(self.persist_directory)

    def load_db(self):
        """Load the existing Chroma database and full transcript."""
        if self.is_indexed():
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            if os.path.exists(self.full_transcript_path):
                with open(self.full_transcript_path, "r", encoding="utf-8") as f:
                    self.full_translated_text = f.read()
            return db
        return None

    def process_video(self, progress_callback=None):
        """Fetch transcript, translate, and build vector db."""
        # 1. Fetch Transcript
        if progress_callback: progress_callback("🔍 Fetching transcript...", 0.1)
        try:
            transcript_obj = YouTubeTranscriptApi().fetch(self.video_id, languages=["en", "hi"])
            is_english = transcript_obj.language_code == 'en'
        except Exception as e:
            raise Exception(f"❌ Error fetching transcript: {str(e)}")

        full_text = " ".join(entry.text.strip() for entry in transcript_obj if entry.text.strip())

        # 2. Chunking
        if progress_callback: progress_callback("✂️ Chunking transcript...", 0.3)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(full_text)

        # 3. Translation
        if is_english:
            if progress_callback: progress_callback("✅ Transcript is already in English.", 0.6)
            translated_chunks = chunks
        else:
            if progress_callback: progress_callback(f"🌐 Translating {len(chunks)} chunks in parallel...", 0.4)
            
            # Using the pre-defined translation_chain with concurrency
            responses = self.translation_chain.batch([{"text": chunk} for chunk in chunks])
            translated_chunks = responses
            
            if progress_callback: progress_callback("✅ Translation complete.", 0.7)

        # 4. Store in Chroma
        if progress_callback: progress_callback("📦 Building vector database...", 0.8)
        db = Chroma.from_texts(
            translated_chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        # 5. Save Full Transcript
        self.full_translated_text = "\n\n".join(translated_chunks)
        with open(self.full_transcript_path, "w", encoding="utf-8") as f:
            f.write(self.full_translated_text)

        if progress_callback: progress_callback("✅ Setup complete!", 1.0)
        return db

    def ask(self, query, db, chat_history=None):
        """Ask a question about the video content using LCEL chains and memory."""
        if chat_history is None:
            chat_history = []
            
        if not self.full_translated_text and os.path.exists(self.full_transcript_path):
             with open(self.full_transcript_path, "r", encoding="utf-8") as f:
                self.full_translated_text = f.read()

        # 1. Rephrase query based on history
        if chat_history:
            refined_query = self.contextualize_q_chain.invoke(
                {"chat_history": chat_history, "query": query}
            )
        else:
            refined_query = query

        # 2. Define Common Prompt with history context
        final_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", 
                    """You are a helpful AI assistant. Answer the question using ONLY the provided context from a video transcript.
                    
                    Rules:
                    1. Use the context to provide a comprehensive and accurate answer.
                    2. If the context doesn't contain the answer, but contains related information, provide that and clarify it's related.
                    3. If the context is completely unrelated, say "I don't know".
                    4. Keep your tone professional and helpful."""
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "Context:\n{context}\n\nQuestion: {query}"),
            ]
        )

        # 3. Define Retrieval chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        retriever = db.as_retriever(search_kwargs={"k": 10})
        
        # 4. Define Specialized Chains
        rag_chain = (
            {
                "context": (lambda x: x["query"]) | retriever | format_docs, 
                "query": lambda x: x["query"], 
                "chat_history": lambda x: chat_history
            }
            | final_prompt
            | self.llm
            | StrOutputParser()
        )

        summary_chain = (
            {
                "context": lambda x: self.full_translated_text, 
                "query": lambda x: x["query"], 
                "chat_history": lambda x: chat_history
            }
            | final_prompt
            | self.llm
            | StrOutputParser()
        )

        # 5. Route Execution using the refined query for intent and search
        intent = self.router_chain.invoke({"query": refined_query}).strip().upper()
        
        if "SUMMARY" in intent and self.full_translated_text:
            return summary_chain.invoke({"query": refined_query})
        else:
            return rag_chain.invoke({"query": refined_query})
