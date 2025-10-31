#!/bin/bash

# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows, use: .\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Start the Flask server
python app.py