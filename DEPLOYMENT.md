# HOA Bot - Streamlit Community Cloud Deployment Guide

## Quick Deployment

Your HOA Bot is now ready for Streamlit Community Cloud deployment with pre-configured settings!

## Environment Variables (Already Configured)

The following environment variables are pre-configured in `config.py` and will be automatically loaded:



## Deployment Steps

### 1. Deploy to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `mr813/hoa-bot`
5. Set the main file path to: `streamlit_app.py`
6. Click "Deploy!"

### 2. Environment Variables (Optional)

If you want to override the default settings, you can set environment variables in the Streamlit Cloud dashboard:

1. Go to your deployed app
2. Click the "Settings" button (⚙️)
3. Scroll down to "Secrets"
4. Add any of these variables to override defaults:

```toml
PERPLEXITY_API_KEY = "your-new-api-key"
SMTP_SERVER = "your-smtp-server"
SMTP_PORT = 587
SMTP_USERNAME = "your-email@gmail.com"
SMTP_PASSWORD = "your-app-password"
FROM_EMAIL = "your-email@gmail.com"
```

## Features Available

✅ **User Authentication** - Register and login with email verification
✅ **Property Management** - Add multiple properties with maps
✅ **Document Upload** - Upload HOA bylaws and other documents with OCR
✅ **AI Chatbot** - RAG-powered analysis with conflict detection
✅ **Florida Law Integration** - Chapter 718 compliance checking
✅ **Reflection System** - Multi-step response improvement
✅ **Persistent Storage** - Documents and chat history saved

## Security Notes

- The app uses secure password hashing with bcrypt
- Email passwords are stored as environment variables
- All user data is isolated per user
- No sensitive data is logged or exposed

## Troubleshooting

### Common Issues

1. **Import Errors**: The app should work without any import issues now
2. **Missing Dependencies**: All required packages are in `requirements.txt`
3. **System Dependencies**: Poppler and Tesseract are specified in `packages.txt`

### Support

If you encounter any issues:
1. Check the Streamlit Cloud logs
2. Verify environment variables are set correctly
3. Ensure your GitHub repository is up to date

## Local Development

To run locally with the same configuration:

```bash
# Clone the repository
git clone https://github.com/mr813/hoa-bot.git
cd hoa-bot

# Create virtual environment
python3 -m venv hoa_auditor_env
source hoa_auditor_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

The app will automatically use the settings from `config.py`!
