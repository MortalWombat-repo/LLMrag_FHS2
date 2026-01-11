from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core import import_google_api, embedding_function, persistent_client_hr, persistent_client_en, get_query


client, gemini_embed_fn, collection = None, None, None
collection_hr = None
collection_en = None
try:
    client = import_google_api()
    gemini_embed_fn = embedding_function(client)
    collection_hr = persistent_client_hr(gemini_embed_fn) 
    collection_en = persistent_client_en(gemini_embed_fn)
    print("Application dependencies initialized successfully.")
except Exception as e:
    print(f"Failed to initialize application dependencies: {e}")

app = FastAPI()

# CORS Configuration to allow frontend access
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------------------------------------------

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def handle_query(req_data: QueryRequest, request: Request):
    """
    Endpoint to process a user query, check the IP rate limit, and return a response.
    """
    # Check if ALL resources are initialized
    if not all([client, gemini_embed_fn, collection_hr, collection_en]):
        raise HTTPException(
            status_code=500,
            detail="Application is not initialized. Please check the server logs."
        )

    client_ip = request.client.host
    print(f"RATE LIMITING KEY: rl:{client_ip}:...")
    
    try:
        response_text = get_query(
            user_query=req_data.query, 
            embed_fn=gemini_embed_fn, 
            collection_hr=collection_hr,
            collection_en=collection_en,
            client=client, 
            request_ip=client_ip
        )
        
        if response_text.startswith("ERROR 429"):
             raise HTTPException(
                 status_code=429,
                 detail=response_text
             )
             
        return {"response": response_text}
        
    except HTTPException as h_e:
        raise h_e 
    except Exception as e:
        print(f"An error occurred during query processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during query processing.")