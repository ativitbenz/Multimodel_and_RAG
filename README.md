# Multimodel_and_RAG

This repository provides implementations for Multimodal Retrieval-Augmented Generation (RAG) systems. It combines various models and techniques to enhance the retrieval and generation processes using both text and image data.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Virtual Environment Setup](#virtual-environment-setup)
- [Configuration](#configuration)
- [Running the Code](#running-the-code)
- [Usage](#usage)


## Requirements

Before you start, ensure you have the following:

- Python 3.7 or later
- pip (Python package installer)

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/ativitbenz/Multimodel_and_RAG.git
    cd Multimodel_and_RAG
    ```

## Virtual Environment Setup

It's recommended to use a virtual environment to manage your dependencies. Follow these steps:

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
    model_name = "your_pretrained_model_name"
    data_path = "path/to/your/dataset"
    output_path = "path/to/save/results"
    
    [api_keys]
    service_1 = "your_api_key_for_service_1"
    service_2 = "your_api_key_for_service_2"
    ```

4. **Ensure that `secrets.toml` contains the correct paths and API keys:**

    - `model_name`: Name of the pretrained model you are using.
    - `data_path`: Path to your dataset.
    - `output_path`: Path to save the results.
    - `api_keys`: API keys for any external services required by the scripts.

## Running the Code

To run the main script, use the following command:

```sh
streamlit run main.py --server.port 8080
```
