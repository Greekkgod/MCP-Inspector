from huggingface_hub import HfApi

api = HfApi()
models = list(api.list_models(search="llama", pipeline_tag="text-generation", limit=3, expand=["safetensors"]))
print("With expand=safetensors:")
for m in models:
    print(getattr(m, "safetensors", None))
