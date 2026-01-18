import pandas as pd
import numpy as np
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import gradio as gr

from pathlib import Path
import sys

# Allow imports from src/
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from models.sklearn.predict import SklearnCategoryClassifier
from models.hf import ZeroShotCategoryClassifier


load_dotenv()

# Initialize Classifiers
baseline_classifier = SklearnCategoryClassifier()
hf_classifier = ZeroShotCategoryClassifier()


# Load Data
books = pd.read_csv("data/books_with_emotions.csv")
books["isbn13"] = books["isbn13"].astype(str) # Ensure ISBN is string for matching

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover_not_available.jpg",
    books["large_thumbnail"]
)

#Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Vector Store
persist_directory = "./chroma_db"

if os.path.exists(persist_directory):
    # Load existing database 
    db_books = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    # Create new database
    raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
    documents = text_splitter.split_documents(raw_documents)
    db_books = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)

def retrieve_semantic_recommendations(query: str, category: str="All", tone: str="All", initial_top_k: int=50, final_top_k: int=16):
    recs = db_books.similarity_search(query, k=initial_top_k)
    
    # Get the full 
    book_list = [rec.page_content.strip().split()[0] for rec in recs]
    
    # Filter books based on ISBN
    book_recs = books[books["isbn13"].isin(book_list)].copy()

    # Filter by Category
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Suprising": # Matches your dropdown choice
        book_recs.sort_values(by="surprise", ascending=False, inplace=True) # Matches CSV column
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True) # Matches CSV column

    return book_recs.head(final_top_k)

def recommend_books(query: str, category: str, tone: str, category_mode: str):

    if category_mode == "Auto (Sklearn)":
        category = baseline_classifier.predict(query)

    elif category_mode == "HF Zero-Shot":
        predicted, confidence = hf_classifier.predict_with_score(query)
        category = predicted if confidence >= 0.4 else "All"

    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = str(row["description"])
        truncated_description = " ".join(description.split()[:30]) + "..."

        
        authors = str(row["authors"])
        authors_split = authors.split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])} and {authors_split[-1]}"
        else:
            authors_str = authors

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Suprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Book Recommender using Semantics")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description:", placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        category_mode = gr.Radio(
            choices=["Manual", "Auto (Sklearn)", "HF Zero-Shot"],
            value="Manual",
            label="Category Selection Mode"
        )
        tone_dropdown = gr.Dropdown(choices=tones, label="Select a tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended Books", columns=8, rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown, category_mode],
                        outputs=output)

if __name__ == "__main__":
    dashboard.launch()