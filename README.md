<div align="center">
  <img alt="logo" height="200px" src=".readme/icon.png" >
  <h1>Osam</h1>
  <p>
    Get up and running vision foundational models locally.
  </p>
  <br>
  <br>
  <br>
</div>

*Osam* (/oʊˈsɑm/) is a tool to run open source vision foundational models locally,
built inspired by [Ollama](https://github.com/ollama/ollama).

*Osam* gives you:

- **Visual foundational models** - Segment-Anything Model, Efficient-SAM, etc;
- **Local APIs** - CLI & Python & HTTP interface;
- **Customization** - Host custom vision models.


## Installation

```bash
pip install osam
```


## Usage

### CLI

```bash
# Run a model with an image
osam run sam:308m --image examples/_images/dogs.jpg > mask.jpg

# Get a JSON output
osam run sam:308m --image examples/_images/dogs.jpg --json
# {"model": "sam:308m", "mask": "..."}

# Give a prompt
osam run sam:308m --image examples/_images/dogs.jpg --json \
  --prompt '{"points": [[1439, 504], [1439, 1289]], "point_labels": [1, 1]}'
```

### Python

```python
import numpy as np
import PIL.Image

import osam

request = osam.types.GenerateRequest(
    model=model_name,
    image=np.asarray(PIL.Image.open("examples/_images/dogs.jpg)),
    prompt=osam.types.Prompt(points=[[1439, 504], [1439, 1289]], point_labels=[1, 1]),
)
response = osam.apis.generate(request=request)
PIL.Image.fromarray(response.mask).save("mask.jpg")
print(response.mask.shape, request.mask.dtype)  # (1600, 2560), "uint8"
```

### HTTP

```bash
# Get up the server
osam serve

# POST request
curl 127.0.0.1:11368/api/generate -X POST \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"sam:308m\", \"image\": \"$(cat examples/_images/dogs.jpg | base64)\"}"
```

## License

MIT
