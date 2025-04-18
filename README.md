# Big Data course. PyTorch demo

## Install from requirements.txt

```
pip install -r requirements.txt
```

## Run model training

```
python model.py
```

## If monitoring is required run TensorBoard

```
tensorboard --logdir=runs
```

## Enable FastAPI server

```
uvicorn serve:app --reload
```

## Open interactive client to test out model

```
python drawer.py
```

## For torchserve archive model and neccessary files into the export directory with:

```
torch-model-archiver --model-name mnist --version 1.0 --serialized-file mnist_model.pth --handler mnist_handler.py --extra-files "model.py" --export-path model_store
```

## Create config.properties

```
touch config.properties
```

## And then run torchserve with:

```
torchserve --start --ts-config ./config.properties --model-store ./model_store --models mnist.mar --disable-token-auth
```