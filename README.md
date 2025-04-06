### 🎯 Overview
IntelliResearch is an AI-powered document analysis system designed to revolutionize how users interact with complex, multilingual PDFs. It supports both Arabic and English and goes beyond text summarization by also generating insightful visualizations and academic-style questions using advanced LLMs like **Gemini 1.5 Pro** , **LLaMA 3.2** and Phi4.

---

### ⚙️ Key Features
- Upload and analyze one or multiple PDF documents.
- Generate structured summaries, plots, graphs, and tables.
- Ask and answer questions in natural language (Arabic/English).
- Extract and understand both text and image content from documents.

---

### 🧠 System Architecture
1. **Document Upload & Parsing**  
   PDFs are uploaded, parsed for both text and images using OCR and other techniques.
   ![Image](https://github.com/user-attachments/assets/a6df7cf3-16ce-4873-892b-30efd68f580c)

3. **Summarization & Q&A**  
   Gemini 1.5 Pro processes the content to produce structured summaries and academic questions.

4. **Multilingual Chat & Visualization**  
   Supports intelligent conversations, visual dashboards, and cross-language understanding.

5. **RAG Integration**  
   Uses a custom Retrieval-Augmented Generation pipeline:
   - Extract text & images
   - Chunk data, embed, and index with ChromaDB
   - Retrieve context with embeddings for LLM querying
   - Apply routing logic and corrective RAG for refined responses

---
![Image](https://github.com/user-attachments/assets/f20a23ca-c1c4-449a-95da-634b1f5ba4ba)
### 🎬 Demo & Impact
Check out our live [**Demo Video**](https://drive.google.com/file/d/1g5LKsKxZzSSCeGrwpXY9e-bo9_mXcfwt/view)  
View the [**Full Presentation**](https://drive.google.com/file/d/1relEJzb5GFxtIVp2kOb8oUDTchjdoyVV/view)

---


### 🙌 Conclusion
IntelliResearch transforms complex PDFs into understandable insights with just a few clicks. It’s your intelligent research partner—capable of processing, reasoning, and visualizing like a human assistant.

