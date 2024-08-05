from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import http.client
import requests
from codecs import encode
from pydantic import HttpUrl
from huggingface_hub import InferenceClient
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware

client = InferenceClient("VietTung04/vit-l-16-food-classifier", token="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")


app = FastAPI()

# Add CORSMiddleware to the application instance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True, 
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

async def classify_image(image_data):
    try:
        result = client.image_classification(image_data)
        if result and len(result) > 0:
            class_prediction = result[0].label
            confidence = result[0].score
            return {"class": class_prediction, "confidence": confidence}
        else:
            raise HTTPException(status_code=400, detail="Classification failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during classification: {str(e)}")


@app.post("/classify/file/")
async def classify_image_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        return await classify_image(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/classify/url/")
async def classify_image_url(image_url: HttpUrl = Form(...)):
    try:
        headers = {
            'User-Agent': 'PostmanRuntime/7.28.4'
        }
        response = requests.get(image_url, headers=headers, allow_redirects=False)
        response.raise_for_status()
        return await classify_image(response.content)
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Check if the file type is supported
    supported_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff', 'application/pdf']
    if file.content_type not in supported_types:
        raise HTTPException(status_code=400, detail="Unsupported file type!")

    conn = http.client.HTTPSConnection("api.imgur.com")
    boundary = 'wL36Yn8afVp8Ag7AmP8qZ0SA4n1v9T'
    dataList = [
        encode('--' + boundary),
        encode(f'Content-Disposition: form-data; name="image"; filename="{file.filename}"'),
        encode(f'Content-Type: {file.content_type}'),
        encode(''),
        await file.read(),
        encode('--' + boundary + '--'),
        encode('')
    ]

    body = b'\r\n'.join(dataList)
    headers = {
        'Authorization': 'Client-ID 0aec906896029c6',
        'Content-type': f'multipart/form-data; boundary={boundary}'
    }

    conn.request("POST", "/3/image", body, headers)
    response = conn.getresponse()
    data = response.read()
    return data.decode("utf-8")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)