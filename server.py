from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Union, Optional
from enum import Enum
import uvicorn
from recomendator import MDP
import json
import pickle
import pandas as pd 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

ml_models = {}

def saveJson(name, dictObj):
    with open(name, "w") as f:
        json.dump(dictObj, f)

def loadJson(name):
    with open(name, "r") as f:
        return json.load(f)

def savePickle(name, arrObj):
    with open(name, "wb") as f:
        pickle.dump(arrObj, f)

def loadPickle(name):
    with open(name, "rb") as f:
        return pickle.load(f)
    
def Save(invertedStatesDict,statesDict,probTransitions,categoriasDict,invertedCategoriasDict,arrTransitions):
    savePickle("invertedStatesDict.pkl", invertedStatesDict)
    savePickle("statesDict.pkl", statesDict)
    savePickle("probTransitions.pkl", probTransitions)
    savePickle("categoriasDict.pkl", categoriasDict)
    savePickle("invertedCategoriasDict.pkl", invertedCategoriasDict)
    savePickle("arrTransitions.pkl", arrTransitions)

def load():
    invertedStatesDict = loadPickle("invertedStatesDict.pkl")
    statesDict = loadPickle("statesDict.pkl")
    probTransitions = loadPickle("probTransitions.pkl")
    categoriasDict = loadPickle("categoriasDict.pkl")
    invertedCategoriasDict = loadPickle("invertedCategoriasDict.pkl")
    arrTransitions = loadPickle("arrTransitions.pkl")
    videosIdMapping = pd.read_csv("videosIdMapping.csv")
    mdpTransitions = loadPickle("mdpTransitions.pkl")
    values = loadPickle("values.pkl")
    policy = loadPickle("policy.pkl")
    return invertedStatesDict,statesDict,probTransitions,categoriasDict,invertedCategoriasDict,arrTransitions,videosIdMapping,mdpTransitions, values, policy


def getVideos(model:MDP, videosIdMapping:pd.DataFrame, recomendations:list):
    path = []
    # print(recomendations[0])
    for state, category in recomendations:
        selection = videosIdMapping[videosIdMapping["categoria"] == category]
        # try:
        if not selection.empty:
            idmaxVideo = selection["count"].idxmax()
            # print("1")
            videoMayorCount = selection.loc[idmaxVideo]
            # print("2")
            metrics = model.getMetrics(state)
            # print("3:",videoMayorCount["videoid"])
            path.append({
                "estado": state, 
                "metricas": metrics, 
                "videoId": int(videoMayorCount["videoid"])
            })
        # except ValueError:
        #     print(f"Advertencia: No se encontraron videos para el estado '{state}' y la categoría correspondiente. Se omitirá.")
        #     print("selection:",selection)
        #     print(ValueError)
    return path

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código que se ejecuta ANTES de que la aplicación empiece a aceptar solicitudes
    print("Iniciando carga de modelos...")
    invertedStatesDict, statesDict, probTransitions, categoriasDict, invertedCategoriasDict, arrTransitions, videosIdMapping, mdpTransitions, values, policy = load()
    
    model = MDP(
        invertedStatesDict=invertedStatesDict,
        statesDict=statesDict,
        probTransitions=probTransitions,
        categoriasDict=categoriasDict,
        invertedCategoriasDict=invertedCategoriasDict,
        arrTransitions=arrTransitions,
        preInitialized=True,
        mdpTransitions=mdpTransitions, 
        values=values, 
        policy=policy
    )
    
    # Guarda el modelo en el diccionario para que los endpoints puedan usarlo
    ml_models["recommender"] = model
    ml_models["videos_mapping"] = videosIdMapping
    print("Inicialización completa. La aplicación está lista.")
    
    yield
    
    # Código que se ejecuta al apagar la aplicación (limpieza)
    print("Limpiando recursos...")
    ml_models.clear()

# invertedStatesDict,statesDict,probTransitions,categoriasDict,invertedCategoriasDict,arrTransitions,videosIdMapping,mdpTransitions, values, policy = load()
# model = MDP(
#           invertedStatesDict=invertedStatesDict,
#           statesDict=statesDict,
#           probTransitions=probTransitions,
#           categoriasDict=categoriasDict,
#           invertedCategoriasDict=invertedCategoriasDict,
#           arrTransitions=arrTransitions,
#           preInitialized=True,
#           mdpTransitions=mdpTransitions, 
#           values=values, 
#           policy=policy
#         )
# print("inicializacion completa")

class VideoRequest(BaseModel):
    # videoId: Union[str, int] = Field(..., description="Unique identifier for the video")
    videoId: str = Field(..., description="Id del video")
    behavior: str = Field(..., description="Comportamiento del usuario")
    categoryId: str = Field(..., description="Categoria")
    receiveLike: str = Field(..., description="Recibio like o no")
    
    class Config:
        json_schema_extra = {
            "example": {
                "videoId": "video_123",
                "behavior": "fastSkip",
                "categoryId": "music",
                "receiveLike": True
            }
        }

class RecommendedVideo(BaseModel):
    videoId: str
    title: str
    categoryId: str
    score: float = Field(..., ge=0.0, le=1.0, description="Recommendation score between 0 and 1")
    duration: int = Field(..., gt=0, description="Video duration in seconds")

class RecommendationResponse(BaseModel):
    status: str = "success"
    input: VideoRequest
    recommendations: List[RecommendedVideo]
    count: int

class ErrorResponse(BaseModel):
    error: str
    status: str = "error"

app = FastAPI(
    title="Video Recommendation API",
    description="API for getting video recommendations based on user interaction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan  # ¡Aquí está la magia!
)


@app.post(
    "/api/recommendations",
    # response_model=RecommendationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    },
    tags=["recommendations"],
    summary="Get video recommendations",
    description="Get personalized video recommendations based on video ID, category, and like status"
)

async def endPoint_GetRecomendations(request: VideoRequest):
    try:
        userState = (request.receiveLike, request.categoryId, request.behavior)
        
        model = ml_models["recommender"]
        videosIdMapping = ml_models["videos_mapping"]
        
        if userState not in list(model.invertedStatesDict.keys()):
            raise HTTPException(
                status_code=400,
                detail=f"Estado de usuario inválido: {userState}. Verifique los valores de receiveLike, categoryId y behavior."
            )
        # print("recomienda")
        recommendations = model.recommend(state_index=model.invertedStatesDict[userState], n_steps=5)
        # print("busca")
        videos = getVideos(model=model, videosIdMapping=videosIdMapping, recomendations=recommendations)
        # print("some")
        # print(videos)
        return {"response":videos}
        
    except HTTPException as he:
        # Re-lanza las excepciones HTTP para que FastAPI las maneje
        raise he
    except Exception as e:
        print(f"Failed recomendation engine: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}, Failed recomendation engine"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "healthy", "service": "video-recommendation-api"}



if __name__ == "__main__":
    uvicorn.run(
        "server:app",  # Asegúrate que el archivo se llame server.py
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
"""
import requests

response = requests.post(
    "http://localhost:8000/api/recommendations",
    json={
        "dioLike": "video_123",
        "categoria": "music",
        "comportamiento": "fastSkip"
    }
)
print(response.json())
# """