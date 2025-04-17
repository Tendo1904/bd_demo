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

## Enable server

```
uvicorn serve:app --reload
```

## Open interactive client to test out model

```
python drawer.py
```