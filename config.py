"""
Configuration file for HOA Bot
Contains API keys and email settings for Streamlit Community Cloud deployment
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Perplexity API Configuration
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY', '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')

# SMTP Email Configuration
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USERNAME = os.getenv('SMTP_USERNAME', 'gobulls1026@gmail.com')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', 'ohkm phsc fcrq lllc')
FROM_EMAIL = os.getenv('FROM_EMAIL', 'gobulls1026@gmail.com')

# Streamlit Community Cloud Environment Variables
# These will be set in the Streamlit Cloud dashboard
STREAMLIT_CLOUD_CONFIG = {
    'PERPLEXITY_API_KEY': PERPLEXITY_API_KEY,
    'SMTP_SERVER': SMTP_SERVER,
    'SMTP_PORT': SMTP_PORT,
    'SMTP_USERNAME': SMTP_USERNAME,
    'SMTP_PASSWORD': SMTP_PASSWORD,
    'FROM_EMAIL': FROM_EMAIL
}
