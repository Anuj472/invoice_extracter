import os
import json
import base64
from io import BytesIO
from typing import List, Optional, Dict, Any
import re
from datetime import datetime
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
from dotenv import load_dotenv

try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
    print("✅ PDF support enabled")
except ImportError:
    PDF_SUPPORT = False
    print("❌ PDF support disabled. Install: pip install pdf2image")
    print("   Also install poppler-utils for your OS")

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Please set OPENAI_API_KEY in your .env file")
    print("Example .env file content:")
    print("OPENAI_API_KEY=sk-proj-your_actual_api_key_here")
    exit(1)

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(
    title="Invoice Extraction API",
    description="AI-powered invoice data extraction API using OpenAI Vision",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class ConfidenceField(BaseModel):
    value: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)

class VendorDetails(BaseModel):
    name: ConfidenceField
    gstin: ConfidenceField
    address: ConfidenceField

class InvoiceDetails(BaseModel):
    invoice_number: ConfidenceField
    invoice_date: ConfidenceField
    po_number: ConfidenceField

class LineItem(BaseModel):
    description: ConfidenceField
    quantity: ConfidenceField
    unit_price: ConfidenceField
    total: ConfidenceField

class InvoiceExtraction(BaseModel):
    vendor_details: VendorDetails
    invoice_details: InvoiceDetails
    line_items: List[LineItem]
    subtotal: ConfidenceField
    tax: ConfidenceField
    total: ConfidenceField
    extraction_metadata: Dict[str, Any]

# Helper Functions
def calculate_confidence(value: str, field_type: str) -> float:
    if not value or not value.strip():
        return 0.0
    base_confidence = 0.8
    value = value.strip()
    if field_type == "name":
        if any(char.isdigit() for char in value):
            base_confidence -= 0.3
    elif field_type in ("number", "amount"):
        if value.replace('.', '').replace('%', '').replace(',', '').replace('₹', '').replace('$', '').isdigit():
            base_confidence += 0.1
    elif field_type == "date":
        if re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', value):
            base_confidence += 0.1
    return max(min(base_confidence, 1.0), 0.1)

def create_confidence_field(value: Any, field_type: str) -> ConfidenceField:
    if value is None or value == "null":
        return ConfidenceField(value=None, confidence=0.0)
    value_str = str(value).strip()
    if not value_str:
        return ConfidenceField(value=None, confidence=0.0)
    confidence = calculate_confidence(value_str, field_type)
    return ConfidenceField(value=value_str, confidence=confidence)

def process_pdf_to_images(content: bytes) -> List[str]:
    if not PDF_SUPPORT:
        raise HTTPException(
            status_code=400,
            detail="PDF processing not available. Please install pdf2image and poppler-utils"
        )
    try:
        images = convert_from_bytes(content, dpi=200, first_page=1, last_page=5, fmt='PNG')
        base64_images = []
        for image in images:
            max_size = 800
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            buffer = BytesIO()
            image.save(buffer, format='PNG', optimize=True, quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            base64_images.append(img_base64)
        return base64_images
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

def process_image(content: bytes) -> str:
    try:
        image = Image.open(BytesIO(content))
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        max_size = 800
        if image.width > max_size or image.height > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format='PNG', optimize=True, quality=85)
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def validate_file(file: UploadFile) -> None:
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.pdf'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(allowed_extensions))}"
        )
    if file_extension == '.pdf' and not PDF_SUPPORT:
        raise HTTPException(
            status_code=400,
            detail="PDF processing not available. Install pdf2image and poppler-utils"
        )

def clean_response_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace('``````', '')
    text = text.strip()
    if '{' in text and '}' in text:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            text = text[start:end]
    return text

INVOICE_SYSTEM_PROMPT = """
You are an expert at extracting structured fields from invoice documents.
You will receive multiple images that are pages from a single invoice.
Analyze ALL pages together and extract the following as structured JSON.

Return exactly this JSON structure:
{
  "vendor_details": {
    "name": "vendor name or null",
    "gstin": "vendor GSTIN or null",
    "address": "vendor address or null"
  },
  "invoice_details": {
    "invoice_number": "invoice number or null",
    "invoice_date": "invoice date or null",
    "po_number": "purchase order number or null"
  },
  "line_items": [
    {
      "description": "item/service description or null",
      "quantity": "quantity or null",
      "unit_price": "unit price or null",
      "total": "line item total or null"
    }
  ],
  "subtotal": "subtotal value or null",
  "tax": "tax value or null",
  "total": "invoice total or null"
}
IMPORTANT:
1. Combine information from ALL pages.
2. Return ONLY valid JSON, no explanations.
3. Use null for missing fields.
"""

async def extract_with_openai_multi_page(images_b64: List[str]) -> Dict[str, Any]:
    try:
        content = [
            {
                "type": "text",
                "text": f"Extract all information from this {len(images_b64)}-page invoice document. Analyze all pages together."
            }
        ]
        for img_b64 in images_b64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "high"
                }
            })
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": INVOICE_SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            max_completion_tokens=8000,
        )
        if not response.choices or not response.choices[0].message.content:
            raise HTTPException(status_code=500, detail="Empty response from OpenAI")
        response_content = response.choices[0].message.content.strip()
        clean_content = clean_response_text(response_content)
        if not clean_content:
            raise HTTPException(status_code=500, detail="No valid content in response")
        try:
            parsed_data = json.loads(clean_content)
            return parsed_data
        except json.JSONDecodeError:
            # fallback
            return {
                "vendor_details": {},
                "invoice_details": {},
                "line_items": [],
                "subtotal": None,
                "tax": None,
                "total": None
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

async def extract_with_openai(image_b64: str) -> Dict[str, Any]:
    return await extract_with_openai_multi_page([image_b64])

@app.get("/")
async def root():
    supported_formats = ["JPG", "JPEG", "PNG", "WebP"]
    if PDF_SUPPORT:
        supported_formats.append("PDF")
    return {
        "message": "Invoice Extraction API",
        "status": "active",
        "version": "1.0.0",
        "supported_formats": supported_formats,
        "max_file_size": "10MB",
        "pdf_support": PDF_SUPPORT,
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    supported_formats = ["JPG", "JPEG", "PNG", "WebP"]
    if PDF_SUPPORT:
        supported_formats.append("PDF")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "pdf_support": PDF_SUPPORT,
        "supported_formats": supported_formats
    }

@app.post("/extract", response_model=InvoiceExtraction)
async def extract_invoice(file: UploadFile = File(...)):
    print(f"📄 Processing file: {file.filename}")
    try:
        validate_file(file)
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension == '.pdf':
            images_b64 = process_pdf_to_images(content)
            raw_data = await extract_with_openai_multi_page(images_b64)
            pages_processed = len(images_b64)
        else:
            image_b64 = process_image(content)
            raw_data = await extract_with_openai(image_b64)
            pages_processed = 1

        vendor_raw = raw_data.get("vendor_details", {})
        vendor_details = VendorDetails(
            name=create_confidence_field(vendor_raw.get("name"), "name"),
            gstin=create_confidence_field(vendor_raw.get("gstin"), "general"),
            address=create_confidence_field(vendor_raw.get("address"), "general"),
        )
        invoice_raw = raw_data.get("invoice_details", {})
        invoice_details = InvoiceDetails(
            invoice_number=create_confidence_field(invoice_raw.get("invoice_number"), "number"),
            invoice_date=create_confidence_field(invoice_raw.get("invoice_date"), "date"),
            po_number=create_confidence_field(invoice_raw.get("po_number"), "number"),
        )
        line_items = []
        for item in raw_data.get("line_items", []):
            line_items.append(LineItem(
                description=create_confidence_field(item.get("description"), "general"),
                quantity=create_confidence_field(item.get("quantity"), "number"),
                unit_price=create_confidence_field(item.get("unit_price"), "amount"),
                total=create_confidence_field(item.get("total"), "amount")
            ))
        subtotal = create_confidence_field(raw_data.get("subtotal"), "amount")
        tax = create_confidence_field(raw_data.get("tax"), "amount")
        total = create_confidence_field(raw_data.get("total"), "amount")

        metadata = {
            "filename": file.filename,
            "file_type": file_extension,
            "file_size_mb": round(len(content) / (1024 * 1024), 2),
            "extraction_method": "OpenAI GPT Vision",
            "pages_processed": pages_processed,
            "timestamp": datetime.now().isoformat(),
            "total_line_items": len(line_items),
            "status": "success"
        }
        return InvoiceExtraction(
            vendor_details=vendor_details,
            invoice_details=invoice_details,
            line_items=line_items,
            subtotal=subtotal,
            tax=tax,
            total=total,
            extraction_metadata=metadata
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        },
    )

if __name__ == "__main__":
    uvicorn.run(
        "simple_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
