from fastapi import FastAPI 
from fastapi.responses import StreamingResponse  
from fastapi.middleware.cors import CORSMiddleware  
import uvicorn  
from classification import classify_emotion
from rag import process_user_query

app = FastAPI()  
  
app.add_middleware(  
    CORSMiddleware,  
    allow_origins=["*"],  
    allow_credentials=True,  
    allow_methods=["*"],  
    allow_headers=["*"],  
)  
  
@app.get("/emotion")
async def classify(user_input: str):
    emotion = classify_emotion(user_input)
    return emotion

def rag_generator(user_query):
    yield "event: generatingQuery\ndata: Generating response...\n\n"
    response = process_user_query(user_query)
    yield f"event: queryGenerated\ndata: {response}\n\n"

@app.get("/rag")
async def rag(user_query: str):
    return StreamingResponse(rag_generator(user_query), media_type="text/event-stream")

if __name__ == "__main__":  
    uvicorn.run(app, host="127.0.0.1", port=8000)  
