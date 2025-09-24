# Invoice Data Extraction API

An AI-powered backend API for extracting structured data fields from invoice documents (PDFs and images) using OpenAI Vision models with FastAPI.

## Features

- Extract vendor details, invoice information, line items, and totals.
- Supports PDF and common image formats (JPG, PNG, WEBP).
- Returns data with field-level confidence scores.
- Uses OpenAI GPT-4 Vision-based model for extraction.
- Easy to integrate via REST API.
- PDF processing via `pdf2image` and poppler-utils.
- CORS enabled for frontend integration.

## Requirements

- Python 3.8+
- OpenAI API key with access to Vision GPT models.
- `pdf2image` and `poppler-utils` installed for PDF support (optional).
- Install dependencies in `requirements.txt`.

## Setup

1. Clone the repository:

git clone <repo-url>
cd <repo-directory>



2. Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate


3. Install dependencies:

pip install -r requirements.txt



4. Create a `.env` file in the root directory and add your OpenAI API key:

OPENAI_API_KEY=sk-your-openai-api-key



5. Ensure poppler-utils is installed on your OS if you want PDF support:

- On Ubuntu: 
sudo apt-get install poppler-utils


- On MacOS (with Homebrew):
brew install poppler



## Running the API Server

uvicorn simple_main:app --reload


The API will be available at http://localhost:8000 .

## API Endpoints

- `GET /` - Root endpoint with info.
- `GET /health` - Health check endpoint.
- `POST /extract` - Upload invoice file (PDF/Image) to extract data.

## Usage Example (curl)

curl -X POST "http://localhost:8000/extract" -F "file=@/path/to/invoice.pdf"


## Response Format

JSON containing:

- vendor_details: name, gstin, address (with confidence)
- invoice_details: invoice_number, invoice_date, po_number (with confidence)
- line_items: list of items with description, quantity, unit_price, total (with confidence)
- subtotal, tax, total with confidence scores
- extraction_metadata with file info and processing stats

## Notes

- Supported file types: JPG, JPEG, PNG, WEBP, PDF.
- Max file size: 10 MB.
- OpenAI model used: GPT-4 Vision.
- Temperature parameter fixed to default due to model constraints.

## License

MIT License
