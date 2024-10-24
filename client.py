from io import BytesIO
from pprint import pprint

import numpy as np
import requests
import tritonclient.http as httpclient
import typer
from PIL import Image
from torchvision import transforms

DEFAULT_HOST = "triton.up.railway.app"
app = typer.Typer()

def resnet_preprocess(img_url="https://www.railway-technology.com/wp-content/uploads/sites/13/2023/05/shutterstock_2123019209.jpg"):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return preprocess(img).numpy()

@app.command()
def list_of_models(host: str = DEFAULT_HOST):
    triton_client = httpclient.InferenceServerClient(url=host, ssl=True)
    d = triton_client.get_model_repository_index()
    pprint(d)

@app.command()
def call_add_sub_model(host: str = DEFAULT_HOST):
    triton_client = httpclient.InferenceServerClient(url=host, ssl=True)

    shape = [4]
    a_batch = np.random.rand(*shape).astype(np.float32)
    b_batch = np.random.rand(*shape).astype(np.float32)

    inputs = [httpclient.InferInput('input_a', a_batch.shape, "FP32"), httpclient.InferInput('input_b', b_batch.shape, "FP32")]
    inputs[0].set_data_from_numpy(a_batch)
    inputs[1].set_data_from_numpy(b_batch)

    outputs = [httpclient.InferRequestedOutput('add'), httpclient.InferRequestedOutput('sub')]
    results = triton_client.infer(model_name="add_sub", model_version="1", inputs=inputs, outputs=outputs)
    print(f"add_result = {results.as_numpy('add')};\nsub_result = {results.as_numpy('sub')}")

@app.command()
def call_resnet(host: str = DEFAULT_HOST):
    transformed_img = resnet_preprocess()

    # Setting up client
    client = httpclient.InferenceServerClient(url=host, ssl=True)

    inputs = httpclient.InferInput("input__0", transformed_img.shape, datatype="FP32")
    inputs.set_data_from_numpy(transformed_img, binary_data=True)

    outputs = httpclient.InferRequestedOutput("output__0", binary_data=True, class_count=1000)
    results = client.infer(model_name="resnet", model_version="1", inputs=[inputs], outputs=[outputs])
    inference_output = results.as_numpy('output__0')
    print(inference_output[:5])

if __name__ == '__main__':
    app()
