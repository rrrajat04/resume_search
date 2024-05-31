# Resume Search

## Introduction

Resume Search is a tool designed to search through a collection of resumes using natural language queries. It utilizes pre-trained models for semantic search and zero-shot classification to retrieve relevant resumes based on user input.

## Installation

### Acrobert

1. Install Acrobert by following the instructions from [here](https://huggingface.co/Lihuchen/AcroBERT).
2. Navigate to the folder where you have cloned Acrobert.
3. Install the required dependencies by running:
   ```
   pip install -r requirements.txt
   ```

### ResumeSearch

1. Clone the repository using Git:
   ```
   git clone https://github.com/rrrajat04/resume_search.git
   ```
2. Navigate to the cloned directory:
   ```
   cd resume_search
   ```
3. Install the required dependencies by running:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the app using the following command:
   ```
   streamlit run resume_search_stream.py
   ```
2. The app will be hosted on localhost:8501 by default.

## Note

Ensure that the Python path is set in the environment variables to use Acrobert in the `resume_search_stream.py` file. Alternatively, you can append the path to GLADIS inference using the following code:

```python
import sys
sys.path.append('path/to/GLADIS/inference')
```

---