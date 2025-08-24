# HOA Auditor

A minimal, production-ready MVP Streamlit app that helps Florida condominium owners audit their association rules against their governing documents and Chapter 718, Florida Statutes.

## Overview

HOA Auditor analyzes uploaded PDF documents (Declaration, Bylaws, Rules/Policies) and checks them against Florida Chapter 718 requirements. The app identifies potential conflicts, compliance issues, and provides actionable findings with citations.

## Features

- **Document Upload & Parsing**: Upload up to 5 PDFs (~300 pages total)
- **Auto-classification**: Automatically detects document types (declaration/bylaws/rules/other)
- **Florida Chapter 718 Compliance**: Comprehensive checklist against state statutes
- **Findings Engine**: Identifies hierarchy conflicts, rental/transfer issues, fines/suspensions, meetings/notice, collections, budgets/reserves, elections/recall
- **Research Assistance**: Optional Perplexity API integration for enhanced legal research
- **Reporting**: Generate findings reports and draft board letters
- **Export Options**: Download reports as JSON, PDF, or Markdown

## Installation

### Prerequisites

1. **Python 3.10+**
2. **Tesseract OCR** (for PDF text extraction fallback):
   - macOS: `brew install tesseract`
   - Ubuntu: `sudo apt-get install tesseract-ocr`
   - Windows: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

3. **Poppler** (for PDF processing):
   - macOS: `brew install poppler`
   - Ubuntu: `sudo apt-get install poppler-utils`
   - Windows: Download from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases)

### Setup

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up Perplexity API for research assistance:
   ```bash
   export PERPLEXITY_API_KEY="your_api_key_here"
   ```

## Usage

1. Start the app:
   ```bash
   streamlit run app/main.py
   ```

2. Upload your HOA documents (Declaration, Bylaws, Rules/Policies)

3. Review the document classification and structure

4. Run the audit to generate findings

5. (Optional) Enable research assistance for enhanced analysis

6. Download reports and draft board letters

## Optional Research Setup

To enable AI-powered research assistance:

1. Get a Perplexity API key from [perplexity.ai](https://www.perplexity.ai/)
2. Set the environment variable:
   ```bash
   export PERPLEXITY_API_KEY="your_api_key_here"
   ```
3. Restart the app - research features will automatically appear

The research integration provides:
- Statute section summaries
- Hierarchy principle explanations with Florida case law
- Enhanced findings with sourced information
- Rate-limited API calls with fallback handling

## Project Structure

```
hoa_compliance_manager/
├── app/
│   ├── main.py              # Main Streamlit app
│   ├── parsing.py           # PDF parsing and OCR
│   ├── structure.py         # Document structure detection
│   ├── checklist.py         # Florida Chapter 718 compliance checks
│   ├── findings.py          # Findings engine and normalization
│   ├── reporting.py         # Report and letter generation
│   ├── ui_components.py     # UI components and layouts
│   ├── research.py          # Perplexity API integration
│   └── utils.py             # Utility functions
├── assets/
│   └── state_pack_fl_718.json  # Florida Chapter 718 checklist
├── templates/
│   ├── board_letter.md.j2   # Board letter template
│   └── report.md.j2         # Report template
├── tests/                   # Test files
├── requirements.txt         # Python dependencies
├── .flake8                  # Linting configuration
└── README.md               # This file
```

## Testing

Run tests with:
```bash
python -m pytest tests/
```

## Disclaimers

**IMPORTANT**: This application is for educational purposes only and does not constitute legal advice. 

- All findings are preliminary and should be reviewed by qualified legal counsel
- The app analyzes documents against Florida Chapter 718 but may not capture all legal nuances
- Research assistance is informational only and should not be relied upon for legal decisions
- Users should consult with attorneys for specific legal advice regarding their HOA compliance

## License

MIT License - see LICENSE file for details.

## Support

For issues or questions, please review the code comments and test files for implementation details.
