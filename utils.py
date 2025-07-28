from googletrans import Translator

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def translate_text(text, target_lang="es"):
    translator = Translator()
    translation = translator.translate(text, dest=target_lang)
    return f"{translation.text} (Translated to {target_lang})"