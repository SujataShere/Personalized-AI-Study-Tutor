# 📚 Personalized AI Study Tutor

## 📌 Project Overview

The **Personalized AI Study Tutor** is an AI-powered learning application that helps students understand their study materials, practice questions, and prepare for exams. Users can upload PDF study notes, ask questions, generate multiple-choice quizzes, and create personalized revision plans.

The application uses Large Language Models (LLMs), semantic search, and Retrieval-Augmented Generation (RAG) to provide answers based on uploaded study materials.

## 🎯 Objectives

- Help students understand study materials more effectively.
- Provide answers based on uploaded PDF notes.
- Generate multiple-choice quizzes for self-assessment.
- Create personalized study and revision schedules.
- Retrieve relevant passages from study documents.

## ✨ Features

### 💬 1. AI Study Tutor
- Upload study notes in PDF format.
- Ask questions related to the uploaded notes.
- Receive AI-generated explanations at different difficulty levels.
- View relevant source pages and retrieved passages.

### 📝 2. Quiz Generation
- Generate multiple-choice questions from study materials.
- Select the number of questions.
- Attempt quizzes and check answers.
- View correct answers, explanations, and scores.

### 📅 3. Personalized Revision Planner
- Enter a study topic.
- Set daily study hours and available days.
- Generate a day-by-day revision plan.
- Include learning goals, practice exercises, revision sessions, and self-assessment.

### 📖 4. Semantic Search
- Search uploaded study notes by topic or question.
- Retrieve relevant passages using vector embeddings.
- Explore the source pages associated with retrieved passages.

### 🎚️ 5. Difficulty Settings
Choose from three levels:
- Beginner
- Intermediate
- Advanced

## 🛠️ Technologies Used

- **Python** — Core programming language
- **Streamlit** — Web application interface
- **OpenAI Python SDK** — LLM API client
- **Groq API** — Language model inference
- **Sentence Transformers** — Text embeddings
- **FAISS** — Vector similarity search
- **PyPDF** — PDF text extraction
- **NumPy** — Numerical and vector operations
- **python-dotenv** — Environment variable management

## 🧠 System Workflow

1. The user uploads a PDF containing study notes.
2. The application extracts text from the PDF.
3. The extracted text is divided into smaller chunks.
4. Sentence Transformers converts the chunks into embeddings.
5. FAISS builds an index for semantic similarity search.
6. When the user asks a question, relevant chunks are retrieved.
7. The language model generates an answer using the retrieved context.
8. The user can generate quizzes, check answers, or create a revision plan.

## 📂 Project Structure

Personalized-AI-Study-Tutor/
│
├── app.py
├── Personalized_AI_Study_Tutor.ipynb
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore

## 🔐 Security and Limitations

- A valid Groq API key is required for language model responses.
- The application is designed for text-based PDFs.
- Scanned PDFs may require OCR before their contents can be extracted.
- AI-generated answers and quiz questions may contain errors. Verify important information against the original study materials.
- Uploaded notes are processed in application memory for the current session and are not automatically saved as files by the app.
- Do not commit API keys, passwords, or confidential study documents to a public repository.

## 🚀 Future Enhancements

- Support for DOCX, PPTX, and additional document formats.
- OCR support for scanned documents.
- Student progress tracking and learning analytics.
- Persistent storage of quiz results and study plans.
- Multilingual explanations and regional language support.
- User accounts and saved study sessions.
- Deployment to a cloud hosting platform.

## 🎓 Use Cases

- Exam preparation
- College and university learning
- Self-paced learning
- Revision planning
- Practice tests and self-assessment
- Understanding lecture notes and textbooks

 
