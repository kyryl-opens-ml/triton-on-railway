import numpy as np

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton


@batch
def _add_sub(**inputs):
    a_batch, b_batch = inputs.values()
    add_batch = a_batch + b_batch
    sub_batch = a_batch - b_batch
    return {"add": add_batch, "sub": sub_batch}


with Triton() as triton:
    print("Loading AddSub model")
    triton.bind(
        model_name="AddSub",
        infer_func=_add_sub,
        inputs=[
            Tensor(name="input_a", dtype=np.float32, shape=(-1,)),
            Tensor(name="input_b", dtype=np.float32, shape=(-1,)),
        ],
        outputs=[
            Tensor(name="add", dtype=np.float32, shape=(-1,)),
            Tensor(name="sub", dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128),
        strict=True,
    )
    print("Serving model")
    triton.serve()
