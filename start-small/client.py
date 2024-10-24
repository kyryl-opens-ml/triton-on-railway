import numpy as np
import tritonclient.http as httpclient    

def call_add_sub_model(host: str):
    batch_size = 2
    a_batch = np.ones((batch_size, 1), dtype=np.float32)
    b_batch = np.ones((batch_size, 1), dtype=np.float32)

    triton_client = httpclient.InferenceServerClient(url=host, verbose=True, ssl=True)

    inputs = []
    inputs.append(httpclient.InferInput('input_a', a_batch.shape, "FP32"))
    inputs.append(httpclient.InferInput('input_b', b_batch.shape, "FP32"))

    inputs[0].set_data_from_numpy(a_batch)
    inputs[1].set_data_from_numpy(b_batch)

    outputs = []
    outputs.append(httpclient.InferRequestedOutput('add'))
    outputs.append(httpclient.InferRequestedOutput('sub'))
    results = triton_client.infer(
        model_name="AddSub",
        inputs=inputs,
        outputs=outputs
    )

    add_result = results.as_numpy('add')
    sub_result = results.as_numpy('sub')
    print(f"add_result = {add_result}; sub_result = {sub_result}")

if __name__ == '__main__':
    host = "py-triton-production.up.railway.app"
    call_add_sub_model(host=host)
