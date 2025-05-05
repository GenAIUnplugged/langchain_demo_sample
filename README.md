# Chatbot with Retrieval-Augmented Generation (RAG)

This project is a **Streamlit-based chatbot** that uses **Retrieval-Augmented Generation (RAG)** to answer user queries. It supports querying data from uploaded files (e.g., `.pdf`, `.docx`, `.txt`) or web URLs. The chatbot leverages **LangChain**, **FAISS**, and **OpenAI's GPT models** for efficient and accurate responses. Additionally, it includes support for **Groq-based models** for faster inference in specific use cases.

---

## Features

- **File Upload Support**: Upload `.pdf`, `.docx`, or `.txt` files for querying.
- **Web URL Support**: Fetch and process content from web pages.
- **Retrieval-Augmented Generation**: Combines document retrieval with OpenAI's GPT models for accurate answers.
- **Interactive UI**: Built with Streamlit for a user-friendly interface.
- **Model Selection**: Choose between different OpenAI models (e.g., `gpt-4-turbo`, `gpt-3.5-turbo`) or Groq-based models.
- **Faster Inference**: Includes optimizations for faster query processing using Groq.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   - Create a `.env` file in the root directory.
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

---

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

2. Open the app in your browser at `http://localhost:8501`.

3. Use the sidebar to:
   - Upload a file or enter a web URL.
   - Select a model (e.g., `gpt-4-turbo`, `gpt-3.5-turbo`, or Groq-based models).

4. Ask questions in the input box, and the chatbot will provide answers based on the uploaded file or web content.

---

## Project Structure

```
sample/
├── main.py                # Main Streamlit app
├── main_groq.py           # Alternative implementation with Groq
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── .gitignore             # Ignored files and directories
└── .env                   # Environment variables (not included in version control)
```

---

## Technologies Used

- **Python**: Core programming language.
- **Streamlit**: For building the interactive UI.
- **LangChain**: For RAG pipeline and document processing.
- **FAISS**: For vector-based similarity search.
- **OpenAI GPT Models**: For generating responses.
- **Groq Models**: For faster inference in specific use cases.
- **BeautifulSoup**: For web scraping.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- [OpenAI](https://openai.com/) for GPT models.
- [LangChain](https://langchain.com/) for the RAG framework.
- [Streamlit](https://streamlit.io/) for the interactive UI.
- [Groq](https://groq.com/) for faster inference capabilities.