# This is your BentoML inference service. It loads the trained model, receives an image through an API endpoint, preprocesses it, runs prediction, and returns the class label.

# import required lobrary or module
import io # convert the images into bytes
import bentoml # Used for model serving and deployment
import numpy as np
import torch
from bentoml.io import Image, Text # Define API input and output types.
from PIL import Image as PILImage
from x_ray.constant.training_pipeline import *

# Load BentoML Model
bento_model = bentoml.pytorch.get(BENTOML_MODEL_NAME)
# Create Runner: A Runner is the execution engine.
runner = bento_model.to_runner()
# Create Service
svc = bentoml.Service(name=BENTOML_SERVICE_NAME, runners=[runner])

# API Endpoint
@svc.api(input=Image(allowed_mime_types=["image/jpeg"]), output=Text())
async def predict(img):
    b = io.BytesIO()
    # convert image into bytes
    img.save(b, "jpeg")

    im_bytes = b.getvalue()

    my_transforms = bento_model.custom_objects.get(TRAIN_TRANSFORMS_KEY)

    image = PILImage.open(io.BytesIO(im_bytes)).convert("RGB")

    image = torch.from_numpy(np.array(my_transforms(image).unsqueeze(0)))
    # input image reshape so that model can process it (1 batch_size, RGB = 3, 224 by 224)
    image = image.reshape(1, 3, 224, 224)

    batch_ret = await runner.async_run(image)

    pred = PREDICTION_LABEL[max(torch.argmax(batch_ret, dim=1).detach().cpu().tolist())]

    return pred
