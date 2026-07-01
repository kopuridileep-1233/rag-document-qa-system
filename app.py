import streamlit as st
from pypdf import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import faiss
import numpy as np

# ----------------------------
# Configure Gemini API
# ----------------------------
genai.configure(api_key=""GEMINI_API_KEY"")

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📄 RAG Document Q&A System")
st.write("Upload a PDF or DOCX file and ask questions about it.")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["pdf", "docx"]
)

# ----------------------------
# Process Uploaded File
# ----------------------------
if uploaded_file is not None:

    st.success(
        f"File uploaded successfully: {uploaded_file.name}"
    )

    # ----------------------------
    # Extract Text
    # ----------------------------
    text = ""

    if uploaded_file.name.endswith(".pdf"):
        pdf = PdfReader(uploaded_file)

        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    elif uploaded_file.name.endswith(".docx"):
        doc = Document(uploaded_file)

        for para in doc.paragraphs:
            text += para.text + "\n"

    # ----------------------------
    # Empty Document Check
    # ----------------------------
    if not text.strip():
        st.error("No text could be extracted.")
        st.stop()

    # ----------------------------
    # Display Extracted Text
    # ----------------------------
    st.subheader("Extracted Text")

    st.text_area(
        "Document Content",
        text,
        height=300
    )

    # ----------------------------
    # Split into Chunks
    # ----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    # ----------------------------
    # Create Embeddings
    # ----------------------------
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = embedding_model.encode(chunks)

    # ----------------------------
    # Store in FAISS
    # ----------------------------
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    st.success(
        f"{len(chunks)} chunks stored in FAISS successfully!"
    )

    # ----------------------------
    # Display Chunks
    # ----------------------------
    st.subheader("Text Chunks")

    for i, chunk in enumerate(chunks):

        st.write(f"Chunk {i + 1}")

        st.text_area(
            f"Chunk {i + 1} Content",
            chunk,
            height=150,
            key=f"chunk_{i}"
        )

    # ----------------------------
    # Ask Question
    # ----------------------------
    st.subheader("Ask Questions")

    question = st.text_input(
        "Enter your question"
    )

    # ----------------------------
    # Search Similar Chunks
    # ----------------------------
    if question:

        question_embedding = embedding_model.encode(
            [question]
        )

        distances, indices = index.search(
            np.array(question_embedding),
            k=3
        )

        # ----------------------------
        # Create Context
        # ----------------------------
        context = "\n".join(
            [chunks[i] for i in indices[0]]
        )

        # ----------------------------
        # Prompt
        # ----------------------------
        prompt = f"""
You are a helpful document assistant.

Answer ONLY using the information present in the context.

If the answer is not present, say:
'I could not find that information in the document.'

Context:
{context}

Question:
{question}
"""

        # ----------------------------
        # Gemini Response
        # ----------------------------
        with st.spinner(
            "Generating answer..."
        ):
            response = gemini_model.generate_content(
                prompt
            )

        # ----------------------------
        # Display Answer
        # ----------------------------
        st.subheader("Answer")
        st.write(response.text)

        # ----------------------------
        # Retrieved Chunks
        # ----------------------------
        with st.expander(
            "Retrieved Chunks"
        ):
            for i in indices[0]:
                st.write(chunks[i])