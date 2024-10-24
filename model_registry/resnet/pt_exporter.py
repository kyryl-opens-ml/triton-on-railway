import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

model = (
    torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
    .eval()
    .to(device)
)
traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224).to(device))
torch.jit.save(traced_model, "model.pt")