# This is your BentoML inference service. It loads the trained model, receives an image through an API endpoint, preprocesses it, runs prediction, and returns the class label.

# import required lobrary or module
import io # it allows in memory file operations, which is useful for handling image data without saving it to disk.
import bentoml # Used for model serving and deployment
import numpy as np
import torch
from bentoml.io import Image, Text # Define API input and output types.
from PIL import Image as PILImage # Used for reading and manipulating images.
from x_ray.constant.training_pipeline import *

# Load BentoML Model
bento_model = bentoml.pytorch.get(BENTOML_MODEL_NAME)

# Create Runner: A Runner is the execution engine.
runner = bento_model.to_runner()
# Create Service: It is the main entry point for the BentoML service, which will handle incoming requests and route them to the appropriate runner.
svc = bentoml.Service(name=BENTOML_SERVICE_NAME, runners=[runner])

# API Endpoint
# svc decorator tells BentoMl that input image is of JPEG type and output is a string as response
@svc.api(input=Image(allowed_mime_types=["image/jpeg"]), output=Text())
async def predict(img):
    # create a BytesIO object to hold the image data in memory
    b = io.BytesIO()
    # Stores the uploaded image inside the memory buffer.
    img.save(b, "jpeg")
    # Get the byte data from the BytesIO object.
    im_bytes = b.getvalue()
    # Get the custom transforms from the BentoML model's custom objects. These transforms are used to preprocess the input image before feeding it into the model for prediction.
    my_transforms = bento_model.custom_objects.get(TRAIN_TRANSFORMS_KEY)
    # Convert the byte data to a PIL image and ensure it is in RGB format. This is necessary because the model expects input images to have three color channels (RGB).
    image = PILImage.open(io.BytesIO(im_bytes)).convert("RGB")
    # Apply the custom transforms to the image and convert it to a PyTorch tensor. The unsqueeze(0) adds a batch dimension, making it compatible with the model's expected input shape.
    image = torch.from_numpy(np.array(my_transforms(image).unsqueeze(0)))
    # input image reshape so that model can process it (1 batch_size, RGB = 3, 224 by 224)
    image = image.reshape(1, 3, 224, 224)
    # Runs the image through your CNN model asynchronously and gets the prediction results. The async_run method allows for non-blocking execution, which is useful for handling multiple requests concurrently.
    batch_ret = await runner.async_run(image)
    # Find Predicted Class
    pred = PREDICTION_LABEL[max(torch.argmax(batch_ret, dim=1).detach().cpu().tolist())]

    return pred
