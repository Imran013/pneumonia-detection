"""
Pneumonia Detection API - Single File Version
==============================================
Senior-level FastAPI backend in a single file.

Usage:
    uvicorn pneumonia_api:app --reload --host 127.0.0.1 --port 8000

Requirements:
    pip install fastapi uvicorn python-multipart torch torchvision pillow
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import time
import logging
import threading

# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════

class Config:
    """Application configuration"""
    # Model
    MODEL_PATH = "checkpoints/best_model_acc_98.08.pth"
    DEVICE = "cpu"  # or "cuda"
    NUM_CLASSES = 2
    CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
    
    # Image processing
    IMAGE_SIZE = (224, 224)
    NORMALIZE_MEAN = (0.485, 0.456, 0.406)
    NORMALIZE_STD = (0.229, 0.224, 0.225)
    
    # Validation
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    
    # Rate limiting
    RATE_LIMIT_REQUESTS = 10
    RATE_LIMIT_PERIOD = 60  # seconds
    
    # API
    API_VERSION = "1.0.0"
    API_TITLE = "Pneumonia Detection API"


config = Config()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════

class PneumoniaCNN(nn.Module):
    """ResNet50 fine-tuned model"""
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3):
        super().__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')
        
        # Freeze early layers
        for param in self.backbone.conv1.parameters():
            param.requires_grad = False
        for param in self.backbone.bn1.parameters():
            param.requires_grad = False
        for param in self.backbone.layer1.parameters():
            param.requires_grad = False
        for param in self.backbone.layer2.parameters():
            param.requires_grad = False
        
        # Replace classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ═══════════════════════════════════════════════════════
# MODEL SERVICE (Singleton)
# ═══════════════════════════════════════════════════════

class ModelService:
    """ML model service - Singleton pattern"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model: Optional[nn.Module] = None
        self.device = torch.device(config.DEVICE)
        self.transform = self._create_transform()
        self.is_loaded = False
        self._initialized = True
    
    def _create_transform(self) -> transforms.Compose:
        """Create preprocessing pipeline"""
        return transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD
            )
        ])
    
    def load_model(self):
        """Load model from checkpoint"""
        try:
            logger.info(f"📦 Loading model from {config.MODEL_PATH}")
            
            # Initialize model
            self.model = PneumoniaCNN(num_classes=config.NUM_CLASSES)
            
            # Load checkpoint
            checkpoint = torch.load(
                config.MODEL_PATH,
                map_location=self.device,
                weights_only=False
            )
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"✅ Loaded from epoch {checkpoint.get('epoch', '?')}")
            else:
                self.model.load_state_dict(checkpoint)
            
            # Set to eval mode
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            
            logger.info("✅ Model ready for inference")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict:
        """Run inference"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        # Preprocess
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Forward pass
        outputs = self.model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        probs_np = probabilities.cpu().numpy()[0]
        
        # Get prediction
        predicted_idx = int(torch.argmax(probabilities, dim=1).item())
        predicted_class = config.CLASS_NAMES[predicted_idx]
        confidence = float(probs_np[predicted_idx]) * 100
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': {
                'NORMAL': float(probs_np[0]),
                'PNEUMONIA': float(probs_np[1])
            },
            'processing_time_ms': processing_time,
            'image_size': image.size
        }


# ═══════════════════════════════════════════════════════
# RATE LIMITER
# ═══════════════════════════════════════════════════════

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """Check if request is allowed"""
        with self.lock:
            current_time = time.time()
            window_start = current_time - config.RATE_LIMIT_PERIOD
            
            # Remove old requests
            self.requests[client_id] = [
                t for t in self.requests[client_id] if t > window_start
            ]
            
            # Check limit
            if len(self.requests[client_id]) >= config.RATE_LIMIT_REQUESTS:
                oldest = min(self.requests[client_id])
                retry_after = int(config.RATE_LIMIT_PERIOD - (current_time - oldest)) + 1
                return False, retry_after
            
            # Add current request
            self.requests[client_id].append(current_time)
            return True, 0


rate_limiter = RateLimiter()


# ═══════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ═══════════════════════════════════════════════════════

class PredictionResponse(BaseModel):
    """Prediction response schema"""
    status: str
    prediction: str
    confidence: float = Field(..., ge=0.0, le=100.0)
    probabilities: Dict[str, float]
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    patient_id: Optional[str] = None
    image_size: Optional[tuple] = None
    warnings: Optional[List[str]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "prediction": "PNEUMONIA",
                "confidence": 88.5,
                "probabilities": {"NORMAL": 0.115, "PNEUMONIA": 0.885},
                "processing_time_ms": 245.3,
                "timestamp": "2024-02-05T12:34:56.789Z",
                "patient_id": "PAT-12345",
                "image_size": [1024, 1024],
                "warnings": None
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    api_version: str
    model_loaded: bool


# ═══════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - load model on startup"""
    logger.info("🚀 Starting Pneumonia Detection API...")
    
    # Load model
    model_service = ModelService()
    model_service.load_model()
    app.state.model_service = model_service
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down...")


# Initialize FastAPI
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="AI-powered pneumonia detection from chest X-rays",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Pneumonia Detection API",
        "version": config.API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(request: Request):
    """Health check endpoint"""
    model_service = request.app.state.model_service
    return {
        "status": "healthy",
        "api_version": config.API_VERSION,
        "model_loaded": model_service.is_loaded
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Chest X-ray image (JPG/PNG)"),
    patient_id: Optional[str] = Form(None, description="Optional patient ID")
):
    """
    **Pneumonia Detection Endpoint**
    
    Upload a chest X-ray image to detect pneumonia.
    
    - **file**: Image file (JPG, JPEG, or PNG)
    - **patient_id**: Optional patient identifier
    
    Returns prediction with confidence score.
    
    ⚠️ **Medical Disclaimer**: For educational purposes only.
    """
    
    # Rate limiting
    client_ip = request.client.host
    allowed, retry_after = rate_limiter.is_allowed(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry in {retry_after}s"
        )
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(400, "No filename provided")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in config.ALLOWED_EXTENSIONS:
            raise HTTPException(
                400,
                f"Invalid file type. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}"
            )
        
        # Read image
        contents = await file.read()
        
        if len(contents) > config.MAX_FILE_SIZE:
            raise HTTPException(413, "File too large (max 10MB)")
        
        image = Image.open(io.BytesIO(contents))
        
        # Validate image
        if image.size[0] < 50 or image.size[1] < 50:
            raise HTTPException(400, "Image too small (min 50x50px)")
        
        # Get model service
        model_service = request.app.state.model_service
        
        # Predict
        result = model_service.predict(image)
        
        # Generate warnings
        warnings = []
        if result['confidence'] < 70:
            warnings.append("Low confidence - manual review recommended")
        if image.size[0] < 224 or image.size[1] < 224:
            warnings.append("Low resolution - higher quality recommended")
        
        # Build response
        response = PredictionResponse(
            status="success",
            prediction=result['prediction'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            processing_time_ms=result['processing_time_ms'],
            patient_id=patient_id,
            image_size=result['image_size'],
            warnings=warnings if warnings else None
        )
        
        logger.info(
            f"✅ Prediction: {result['prediction']} "
            f"({result['confidence']:.1f}%) in {result['processing_time_ms']:.0f}ms"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@app.get("/model/info", tags=["Model"])
async def model_info(request: Request):
    """Get model information"""
    model_service = request.app.state.model_service
    
    if not model_service.is_loaded:
        raise HTTPException(503, "Model not loaded")
    
    num_params = sum(p.numel() for p in model_service.model.parameters())
    
    return {
        "model_name": "ResNet50-FT",
        "version": config.API_VERSION,
        "accuracy": 98.08,
        "num_parameters": num_params,
        "device": str(model_service.device),
        "class_names": config.CLASS_NAMES,
        "image_size": config.IMAGE_SIZE
    }


# ═══════════════════════════════════════════════════════
# RUN SERVER
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🏥 PNEUMONIA DETECTION API")
    print("="*60)
    print(f"📡 Server: http://127.0.0.1:8000")
    print(f"📚 Docs:   http://127.0.0.1:8000/docs")
    print(f"💊 Health: http://127.0.0.1:8000/health")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )