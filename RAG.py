import os
import pytesseract
import PyPDF2
import google.generativeai as genai
from pdf2image import convert_from_path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class PDFRAGProcessor:
    def __init__(self, pdf_files, api_key=None):
        self.pdf_files = pdf_files
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API Key must be provided or set in environment variable 'GOOGLE_API_KEY'")

        self.docs = []
        self.vectorstore = None
        self.rag_chain = None
        self.model = None

        self._setup()

    def _load_pdf_with_ocr(self):
        all_text_with_page_numbers = []

        for pdf_file in self.pdf_files:
            pdf_name = os.path.basename(pdf_file)
            pdf_reader = PyPDF2.PdfReader(open(pdf_file, "rb"))

            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()

                if not page_text or not page_text.strip():
                    page_images = convert_from_path(pdf_file, first_page=page_num + 1, last_page=page_num + 1, dpi=300)
                    for page_image in page_images:
                        page_text = pytesseract.image_to_string(page_image)
                        break

                all_text_with_page_numbers.append({
                    "page_number": f"Page {page_num + 1} of {pdf_name}",
                    "text": page_text
                })

        return all_text_with_page_numbers

    def _split_text(self, text_with_page_numbers):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = []

        for page in text_with_page_numbers:
            page_number = page['page_number']
            page_text = page['text']
            chunks = splitter.split_text(page_text)

            for chunk in chunks:
                if chunk:
                    docs.append(Document(page_content=chunk, metadata={'page_number': page_number}))

        return docs

    def _format_docs(self, docs):
        formatted_docs = []
        for doc in docs:
            page_number = doc.metadata.get("page_number", "Unknown")
            formatted_docs.append(f"Page {page_number}: {doc.page_content}")
        return "\n\n".join(formatted_docs)

    def _generate_text(self, text):
        response = self.model.generate_content(text.text)
        return response.text

    def _setup(self):
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash', generation_config={
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 20
        })

        text_pages = self._load_pdf_with_ocr()
        self.docs = self._split_text(text_pages)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=self.api_key)
        self.vectorstore = Chroma.from_documents(documents=self.docs, embedding=embeddings)
        retriever = self.vectorstore.as_retriever()

        prompt = PromptTemplate.from_template(
            """
            You are a Deep Learning Expert. Answer the following question based on the context provided:

            {context}

            Question: {question}

            Display tables in proper table format where available.
            Please include the page number(s) in the format {{page_num}} of pdf_name for any relevant information.
            However, do **not** repeat the page number if it has already been mentioned earlier in the answer.
            """
        )

        self.rag_chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | RunnableLambda(func=self._generate_text)
            | StrOutputParser()
        )

    def query(self, question):
        return self.rag_chain.invoke(question)
