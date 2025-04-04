import fitz  # PyMuPDF
import re
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import keys

# === Setup ===
model = ChatOpenAI()
output_parser = StrOutputParser()

# === 1. Extract Text from PDF ===
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

# === 2. Split Sections ===
def split_into_sections(text):
    section_titles = [
        "Abstract", "Introduction", "Methodology", "Methods",
        "Materials and Methods", "Results", "Discussion", "Conclusion", "References"
    ]
    pattern = r"\n(?P<header>(" + "|".join(section_titles) + r"))\s*\n"
    splits = re.split(pattern, text, flags=re.IGNORECASE)

    paper_sections = {}
    for i in range(1, len(splits) - 1, 2):
        header = splits[i].strip().capitalize()
        content = splits[i + 1].strip()
        paper_sections[header] = content
    return paper_sections

# === 3. Prompt Template ===
PROMPT_TEMPLATE = """
You are an academic summarizer. Given a section of a research paper, produce a concise summary focusing on key points.

Section Title: {question}
Section Content:
{context}

Summary:
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# === 4. Summarize One Section ===
def summarize_section(section_title, section_content):
    prompt = prompt_template.format(context=section_content[:4000], question=section_title)
    response = model.invoke(prompt)
    return response.content.strip()

# === 5. Final Summary from All Summaries ===
FINAL_PROMPT = """
You are an expert academic assistant. Summarize this entire paper based on the following section-wise summaries:

{context}

Final abstract summary in 4-6 sentences:
"""

final_prompt_template = ChatPromptTemplate.from_template(FINAL_PROMPT)

def generate_final_abstract(all_summaries):
    joined_summaries = "\n\n".join(all_summaries)
    final_prompt = final_prompt_template.format(context=joined_summaries, question="Final Abstract")
    response = model.invoke(final_prompt)
    return response.content.strip()

# === Main Driver ===
def summarize_research_paper(pdf_path):
    print("[*] Extracting text...")
    text = extract_text_from_pdf(pdf_path)

    print("[*] Splitting sections...")
    sections = split_into_sections(text)

    all_summaries = []
    for title, content in sections.items():
        print(f"[+] Summarizing section: {title}")
        summary = summarize_section(title, content)
        all_summaries.append(f"--- {title} ---\n{summary}\n")

    print("[*] Generating final summary...")
    final_summary = generate_final_abstract(all_summaries)

    print("\n=== FINAL SUMMARY ===\n")
    print(final_summary)


# === Run it ===
if __name__ == "__main__":
    path_to_pdf = r"C:\Users\Rahul\Desktop\ResearchPapers\NIPS-2017-attention-is-all-you-need-Paper.pdf"  # Replace with your actual PDF file path
    summarize_research_paper(path_to_pdf)
