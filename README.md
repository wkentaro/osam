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
osam run efficient-sam:25m --image examples/_images/dogs.jpg > output.jpg

# Get a JSON output
osam run efficient-sam:25m --image examples/_images/dogs.jpg --json
# {"model": "efficient-sam:25m", "mask": "..."}

# Give a prompt
osam run efficient-sam:25m --image examples/_images/dogs.jpg \
  --prompt '{"points": [[1439, 504], [1439, 1289]], "point_labels": [1, 1]}' > output.jpg
```

<img src="examples/_images/dogs.jpg" width="35%"> <img src=".readme/dogs_output.jpg" width="35%">  
<i>Input and output images ('dogs.jpg', 'output.jpg').</i>

### Python

```python
import osam.apis
import osam.types

request = osam.types.GenerateRequest(
    model="efficient-sam:25m",
    image=np.asarray(PIL.Image.open("examples/_images/dogs.jpg")),
    prompt=osam.types.Prompt(points=[[1439, 504], [1439, 1289]], point_labels=[1, 1]),
)
response = osam.apis.generate(request=request)
PIL.Image.fromarray(response.mask).save("mask.jpg")
```
<img src="examples/_images/dogs.jpg" width="35%"> <img src=".readme/dogs_mask.jpg" width="35%">  
<i>Input and output images ('dogs.jpg', 'mask.jpg').</i>

### HTTP

```bash
# Get up the server
osam serve

# POST request
curl 127.0.0.1:11368/api/generate -X POST \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"efficient-sam:25m\", \"image\": \"$(cat examples/_images/dogs.jpg | base64)\"}" \
  | jq -r .mask | base64 --decode > mask.jpg
```

## License

MIT
