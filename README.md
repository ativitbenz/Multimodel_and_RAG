# Multimodal Retrieval-Augmented Generation (RAG) System

This repository provides implementations for Multimodal Retrieval-Augmented Generation (RAG) systems, combining various models and techniques to enhance retrieval and generation processes using both text and image data.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Virtual Environment Setup](#virtual-environment-setup)
- [Configuration](#configuration)
- [Running the Code](#running-the-code)
- [Credits](#credits)

## Requirements

Ensure you have the following prerequisites:

- Python 3.10 or later
- pip (Python package installer)

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/ativitbenz/Multimodel_and_RAG.git
    cd Multimodel_and_RAG
    ```

## Virtual Environment Setup

It is recommended to use a virtual environment to manage your dependencies. Follow these steps:

1. **Create a virtual environment:**

    ```sh
    python3 -m venv venv
    ```

2. **Activate the virtual environment:**

    - On macOS and Linux:

        ```sh
        source venv/bin/activate
        ```

    - On Windows:

        ```sh
        venv\Scripts\activate
        ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

## Configuration

Configuration settings are managed via a configuration file located in the `.streamlit` directory. Follow these steps to set it up:

1. **Navigate to the `.streamlit` directory:**

    ```sh
    cd .streamlit
    ```

2. **Rename `secrets.example.toml` to `secrets.toml`:**

    ```sh
    mv secrets.example.toml secrets.toml
    ```

3. **Edit the `secrets.toml` file to include your configuration settings:**

    ```toml
    # secrets.toml
    # OpenAI secrets
    OPENAI_API_KEY = "your_openai_api_key"

    # Astra DB secrets
    ASTRA_ENDPOINT = "your_astra_endpoint"
    ASTRA_TOKEN = "your_astra_token"

    # Optionally: LangSmith secrets for tracing
    LANGCHAIN_TRACING_V2 = "true"
    LANGCHAIN_ENDPOINT = "your_langsmith_endpoint"
    LANGCHAIN_API_KEY = "your_langsmith_api_key"
    LANGCHAIN_PROJECT = "your_langsmith_project"
    ```

4. **Ensure that `secrets.toml` contains the correct paths and API keys:**

    - `OPENAI_API_KEY`: [Obtain your OpenAI API key](https://platform.openai.com/api-keys).
    - `ASTRA_ENDPOINT` and `ASTRA_TOKEN`: [Get your Astra DB credentials](https://docs.datastax.com/en/astra-db-serverless/api-reference/dataapiclient.html).
    - `LANGCHAIN_ENDPOINT`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT`: Refer to the [LangSmith documentation](https://docs.smith.langchain.com/old/tracing) or contact their support for more information.

For more information about each service, refer to the following links:

- [OpenAI Developer Quickstart](https://platform.openai.com/docs/quickstart)
- [DataStax Astra Documentation](https://docs.datastax.com/en/astra-db-serverless/get-started/quickstart.html)
- [LangSmith Documentation](https://docs.smith.langchain.com/old/tracing)

## Running the Code

To run the main script, use the following command:

```sh
streamlit run main.py --server.port 8080
```

## Credits
This project is based on the implementation found in the [ragstack-astradb](https://github.com/michelderu/ragstack-astradb) repository by Michelderu.
