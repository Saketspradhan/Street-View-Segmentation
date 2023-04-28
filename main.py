from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import base64
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import requests
from segmentation_mask_overlay import overlay_masks
import base64
from loguru import logger
from pydantic import BaseModel
from io import BytesIO


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_credentials=True,
    allow_headers=["*"],
)

class UserPrompt(BaseModel):
    lat_value: str
    long_value: str
    model_selected : str
    mask_alpha : str 

def PILImage_to_base64str(pilimg):
    buffered = io.BytesIO()
    pilimg.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode('utf-8')

def stringToRGB(base64_string):
            imgdata = base64.b64decode(str(base64_string))
            logger.info("Image data decoded")
            image = Image.open(BytesIO(imgdata))
            return np.array(image)

@app.get("/")
async def health():
    return {"message": "Status 200"}

@app.post("/model")
async def getdata(userPrompt: UserPrompt):
    API_TOKEN = r"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"

    available_models = {
        "DACS" : "https://api-inference.huggingface.co/models/nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        "CorDA": "https://api-inference.huggingface.co/models/nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        "BAPA" : "https://api-inference.huggingface.co/models/nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
        "ProDA" : "https://api-inference.huggingface.co/models/nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
        "DAFormer" : "https://api-inference.huggingface.co/models/nvidia/segformer-b5-finetuned-cityscapes-1024-1024"          
    } 

    API_URL = available_models[userPrompt.model_selected]
    print('Model selected: ' + API_URL)
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    
    image_fetch_url = "https://maps.googleapis.com/maps/api/streetview?size=750x500&location=" + str(userPrompt.lat_value) + "," + str(userPrompt.long_value) +"&fov=80&heading=70&pitch=0&key=AIzaSyBYPxA6cLXn1TM0IC4zOfej44J9zjRBh7w"
    response = requests.get(image_fetch_url)
    img_data = response.content
    google_img = Image.open(BytesIO(img_data))
    google_img.save('input_image.png', 'PNG')
    
    image_path = "input_image.png"
    logger.info("Image uploaded successfully!")
    with open(image_path, "rb") as f:
            data = f.read()
            response = requests.post(API_URL, headers=headers, data=data)
            res = response.json()
            logger.info("Response received from model")
    
    # Extract the masks and their labels from the response
    masks = [stringToRGB(r["mask"]).astype('bool') for r in res]
    masks_labels = [r["label"] for r in res]

    # Define a colormap for the masks
    cmap = plt.cm.tab20(np.arange(len(masks_labels)))
    logger.info("Masks and labels extracted from response")

    # Open the original image and overlay the masks on it
    image = Image.open(image_path)
    overlay_masks(image, masks, labels=masks_labels,
                      colors=cmap, mask_alpha=float(userPrompt.mask_alpha)) 
    # mask_alpha value from 0.1 to 0.9 in incrememts of 0.1
    logger.info("Segmentation mask overlay complete!")

    # Render the figure to a bytes object in PNG format
    buf = BytesIO()

    plt.savefig('segmented.png')
    img_crop = Image.open('segmented.png')

    width, height = img_crop.size
    top_crop = 120
    bottom_crop = 120

    img_cropped = img_crop.crop((0, top_crop, width, height - bottom_crop))
    img_cropped.save(buf, format='png')
    buf.seek(0)

    output_image = base64.b64encode(buf.read()).decode('utf-8')
    return output_image
