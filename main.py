from model.layer import Bi_LSTM_CRF
from flair.data import Sentence
from schema import WordSchema
from settings import app

from fastapi import  APIRouter
from starlette import status
from starlette.responses import JSONResponse
import uvicorn

api_router = APIRouter()



# Variables for Interactive selections
tagger = Bi_LSTM_CRF.load("checkpoints/best-model.pt")

@api_router.get("/")
def health():
    return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "code": 200,
                "version": "1.0.0"
            },
        )

                               

@api_router.get("/api/model", response_model=WordSchema)
def model(
    *, 
    word: str,
    file_name: str
    ):
    """
    An api for serving the model for the PHI classification.
    :param word: list of word tokens in a paragraph.
    :param file_name: name of the wile.

    :returns: json response that contains labeled 
        tags their respective classification.
    """
    txt = Sentence(word)
    tagger.predict(txt)
    labels, tags = [], []

    for entity in txt.get_spans('ner'):
        labels.append(entity.text)
        tags.append(entity.get_label("ner").value)
    

    return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "code": 200,
                "data": {
                    "labels": labels,
                    "tags": tags
                }
            },
        )


app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, debug=True)
