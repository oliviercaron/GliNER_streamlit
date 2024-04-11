# GliNER Streamlit Application

Welcome to the GliNER Streamlit application repository! This project is designed to offer a user-friendly interface for Named Entity Recognition (NER) tasks using the GliNER model.

You can find the GLiNER repository here: https://github.com/urchade/GLiNER

and the paper: https://arxiv.org/abs/2311.08526


## Getting Started

How to run the app on your local machine.

### Cloning the Repository

To clone the repository and navigate into it, run the following commands in your terminal:

```bash
git clone https://github.com/oliviercaron/GliNER_streamlit/
cd GliNER_streamlit
```

### Installing Dependencies

Install the project dependencies by running:

```bash
pip install -r requirements.txt
```

### Running the Application

With the dependencies installed, you can run the Streamlit application locally:

```bash
streamlit run app.py
```

Your default web browser should automatically open to the Streamlit application's URL, typically `http://localhost:8501`. If it doesn't, you can manually navigate to the URL displayed in your terminal.

## How It Works

The GliNER Streamlit application allows users to perform NER tasks in a few simple steps:

1. **Upload Your Data**: You can upload text files (CSV or Excel)
2. **Select Your Options**: You can choose the column to analyze, filter it, and specify labels for entity recognition.
3. **View Results**: The application processes the text and displays the identified entities, which users can then review and download.


## License

This project is licensed under the Apache License 2.0 - see the `LICENSE` file for details.
