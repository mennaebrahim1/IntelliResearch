import os
import fitz  # PyMuPDF
import pandas as pd
import easyocr
from ollama import chat
from ollama import ChatResponse

# --- PDF to Images ---
def pdf_to_images(pdf_path, output_folder, zoom=2):
    pdf_document = fitz.open(pdf_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = []
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        image_path = os.path.join(output_folder, f"page_{page_number + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
        print(f"Saved: {image_path}")

    pdf_document.close()
    return image_paths

# --- OCR Text Extraction ---
def extract_text_easyocr(image_path):
    reader = easyocr.Reader(['en', 'ar'], gpu=True)
    results = reader.readtext(image_path, paragraph=True)
    text = pd.DataFrame(results, columns=['bbox', 'text'])
    return text

# --- Prompt + Phi4 Model Chat ---
def Model(text_str):
    prompt = '''
     Please analyze the following Arabic and multilingual PDF documents to generate a structured survey paper. The paper should be comprehensive and well-organized, capturing all relevant information from each PDF. The structure of the paper should include the following sections:

        Title: A concise and informative title that accurately reflects the scope of the survey.

        Abstract: A high-level overview summarizing the main research topics, domains, and objectives covered in the survey.

        Keywords: List of relevant keywords capturing each unique domain or area discussed, ensuring all covered topics are reflected.

        Introduction: Present the primary research challenges and themes addressed across the papers. Provide a brief introduction to each domain if the scope covers multiple fields.

        Related Work: A thorough review of existing surveys or studies related to the topics in the PDF. Highlight the contributions of each document to its respective field, emphasizing distinctions between domains if multiple fields are involved.

        Methodologies and Approaches: A detailed explanation of the techniques, models, and methodologies used across studies. Organize this section by domain when multiple fields are present, ensuring clarity by explicitly referring to each methodology and its specific research area.

        Results and Findings: Summarize the key findings of each paper, including comparative analyses where relevant. When tables or figures are present, discuss them thoroughly, specifying the paper each result pertains to. Ensure any tables are formatted correctly and presented in table format for clarity.

        Discussion of Trends: An in-depth discussion on notable trends, common insights, and any key distinctions between domains, where applicable.

        Conclusion and Future Directions: Summarize the main conclusions from the survey and propose directions for future research, distinguishing between domains as needed.

        Please ensure the following:
        Language Consistency: Answer in Arabic when discussing Arabic content, and provide clear language tags for sections in other languages where necessary.
        Tables and Figures: Represent all tables and figures in the correct format, ensuring each is referenced within the "Results and Findings" section.
        Clear Citations: Explicitly reference each paper when discussing methodologies, findings, and trends.
        No External Data: Only use content from the provided PDFs for information extraction and analysis.
        Note: Please avoid non-standard characters or LaTeX commands in the output. Maintain structured and clear formatting throughout the paper, and ensure any distinctions between papers or domains are explicitly noted.

        **Important Notes:**
        - **Language Consistency:** Use the dominant language of the pdf, so if the dominant language is English use English or if the the dominant language of the pdf is Arabic use Arabic.
        - **No External Data:** Rely solely on the content of the uploaded PDFs.
        - **Standard Format:** Avoid non-standard characters or LaTeX commands.

        **Question Generation:**

        - After summarizing the PDF, generate 5 open-ended, in-depth questions suitable for academic or technical discussions based on the content.
          Each question should encourage exploration of alternative approaches to specific challenges addressed in the PDF, considering any relevant factors or constraints mentioned.
          Structure each question to:
        - Encourage critical evaluation of the approaches discussed in the PDF.
        - Explore how alternative methods, technologies, or frameworks might address the challenges highlighted.
        - Avoid numbering or bullets before each question and ensure the questions are written in the same language as the PDF content.

        **Important:**
        - Clearly separate the summary and the generated questions in your response.
        - Use headings or formatting to distinguish them (e.g., "## Summary" and "## Generated Questions")
    '''
    response: ChatResponse = chat(
        model='phi4',
        messages=[
            {'role': 'user', 'content': prompt},
            {'role': 'user', 'content': text_str}
        ]
    )
    return response.message.content

# --- Main Execution Pipeline ---
def process_pdfs_in_directory(directory_path):
    text_str = ""
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            output_folder = os.path.join(directory_path, f"{filename}_images")

            # Convert PDF to images
            images = pdf_to_images(file_path, output_folder)

            # Extract text
            for image in images:
                extracted_text_df = extract_text_easyocr(image)
                extracted_text_str = ' '.join(extracted_text_df['text'].astype(str).tolist())
                text_str += extracted_text_str + " "

    # Generate and print result
    result = Model(text_str)
    print(f"\n\n===== Summary for: {filename} =====\n")
    print(result)
    print("\n=========================================\n")
        # Extract summary and questions from the response
    response_text = result
    if ("## Generated Questions" in response_text) or ("## الأسئلة المُولّدة" in response_text):
        if ("## Generated Questions" in response_text):
            summary_text, questions_section = response_text.split("## Generated Questions", 1)
        else:
            summary_text, questions_section = response_text.split("## الأسئلة المُولّدة", 1)
        # Extract questions individually
        questions = [q.strip() for q in questions_section.splitlines() if q.strip()]
    else:
        # Handle case where separator is missing
        summary_text = response_text  # Assume the whole response is the summary
        questions = []  # No questions were generated

    # Return the structured result
    return {
        "summary": summary_text.strip(),
        "questions": questions
    }

