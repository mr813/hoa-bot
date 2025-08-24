# HOA Bot - Florida Condominium Compliance Assistant

A comprehensive Streamlit application that helps Florida condominium owners audit their association rules against governing documents and Chapter 718, Florida Statutes.

## Features

### 🔐 User Management
- Secure user authentication with email registration
- Password management and account security
- Multi-user support with isolated data

### 🏠 Property Management
- Add and manage multiple properties
- Property type classification (Condo/House)
- Interactive maps with geocoding
- Property-specific document storage

### 📄 Document Processing
- PDF upload with OCR capabilities
- Document type classification (HOA Bylaws vs Other)
- Intelligent text extraction and chunking
- Persistent storage with FAISS vector database

### 🤖 AI-Powered Analysis
- RAG (Retrieval Augmented Generation) chatbot
- Conflict detection between rules and bylaws
- Florida condominium law integration (Chapter 718)
- Multi-step reflection for improved responses
- Perplexity API integration for legal research

### 📊 Reporting & Management
- Document removal and storage management
- Chat history and conversation tracking
- Settings and configuration management

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/mr813/hoa-bot.git
   cd hoa-bot
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv hoa_auditor_env
   source hoa_auditor_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install system dependencies**
   ```bash
   # macOS
   brew install poppler tesseract
   
   # Ubuntu/Debian
   sudo apt-get install poppler-utils tesseract-ocr libtesseract-dev
   ```

5. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and email settings
   ```

6. **Run the application**
   ```bash
   streamlit run app/main_with_auth.py
   ```

### Streamlit Cloud Deployment

1. **Fork this repository** to your GitHub account
2. **Connect to Streamlit Cloud** and deploy from the repository
3. **Set environment variables** in Streamlit Cloud dashboard
4. **Access your deployed app** at the provided URL

## Environment Variables

Create a `.env` file with the following variables:

```env
# Perplexity API for legal research
PERPLEXITY_API_KEY=your_perplexity_api_key

# Email settings for user registration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
FROM_EMAIL=your_email@gmail.com
```

## Usage

1. **Register/Login**: Create an account or sign in
2. **Add Properties**: Enter property details and addresses
3. **Upload Documents**: Upload HOA bylaws and other documents
4. **Chat & Analyze**: Ask questions about conflicts and compliance
5. **Review Findings**: Get detailed analysis of rule discrepancies

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.10+
- **AI/ML**: FAISS, Sentence Transformers, Perplexity API
- **Document Processing**: PyMuPDF, Tesseract OCR
- **Authentication**: Streamlit Authenticator, bcrypt
- **Maps**: Folium, Geocoder
- **Storage**: JSON, YAML, FAISS vectors

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions, please open an issue on GitHub or contact the development team.

---

**Note**: This application is designed to assist with legal research and compliance analysis but does not provide legal advice. Always consult with qualified legal professionals for specific legal matters.
