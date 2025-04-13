from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import uuid

app = FastAPI(title="DeepSeek AI API")

# Configure CORS to allow access from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # Use half precision for better performance
    device_map="auto",  # Automatically use GPU if available
)
print("Model and tokenizer loaded successfully!")

class Message(BaseModel):
    role: str
    content: str
    id: Optional[str] = None

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    usage: Dict[str, int]
    choices: List[Dict[str, Any]]

@app.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    try:
        # Format messages for the model
        formatted_messages = []
        for msg in request.messages:
            role = msg.role
            content = msg.content
            
            if role == "user":
                formatted_messages.append(f"Human: {content}")
            elif role == "assistant":
                formatted_messages.append(f"Assistant: {content}")
            else:
                formatted_messages.append(f"{role.capitalize()}: {content}")
        
        # Combine messages for model input
        prompt = "\n".join(formatted_messages)
        prompt += "\nAssistant:"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
            )
        
        # Convert output to text
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract model response (only the new part after "Assistant:")
        assistant_response = full_response.split("Assistant:")[-1].strip()
        
        # Calculate token counts for usage statistics
        prompt_tokens = inputs.input_ids.shape[1]
        completion_tokens = len(tokenizer.encode(assistant_response))
        
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "choices": [
                {
                    "message": {"role": "assistant", "content": assistant_response, "id": f"msg-{uuid.uuid4()}"},
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "model": MODEL_NAME}

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)