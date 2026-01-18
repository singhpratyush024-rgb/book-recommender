# Project Structure

```text
book-recommender/
│
├── gradio-dashboard.py                      
├── README.md                   
├── requirements.txt            
├── .env.example                
│
├── data/
│   ├── books_cleaned.csv
│   ├── books_with_emotions.csv
│   └── books_with_categories.csv
│
├── notebooks/
│   ├── vector-search.ipynb     
│   ├── data-exploration.ipynb
│   ├── sentiment-analysis.ipynb
│   └── text-classification.ipynb
│
├── chroma_db/                  # Persisted vector database
│
├── src/
│   ├── models/
│   │   ├── sklearn/
│   │   │   ├── train.py        
│   │   │   └── predict.py      
│   │   │
│   │   └── hf/
│   │       └── zero_shot.py    
│   │
│   ├── rag/                    
│   │   ├── retriever.py        # Retrieve relevant book/context chunks
│   │   ├── generator.py        # LLM-based response generation
│   │   └── pipeline.py         # End-to-end RAG pipeline
│   │
│   └── __init__.py
│
└── .gitignore
