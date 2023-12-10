# project

## setup

### for windows

- create a virtual environment

```bash
python -m venv venv
```

- set execution policy

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

- activate the virtual environment

```bash
venv\Scripts\activate
```

- install pip

```bash
python.exe -m pip install --upgrade pip
```

- run makefile

```bash
make install
```

### for linux and mac

- create a virtual environment

```bash
python3 -m venv venv
```

- activate the virtual environment

```bash
source venv/bin/activate
```

- run makefile

```bash
make install
```

## DVC setup

```bash
dvc remote add --default storage gdrive://11WDFgw2fv8O6B2GXB1X5ET-tvLfMGd9V
```

### pull

```bash
dvc remote modify --local myremote profile autobots
```

```bash
dvc remote modify myremote gdrive_acknowledge_abuse true
```

```bash
dvc pull
```

### push

```bash
dvc add <filePath>
```

```bash
dvc push
```

### run stages

```bash
dvc repro <stageName>
```

## MLFLOW setup

### For train.py

```bash
python train.py ../data/prepared/train/train.csv ../data/prepared/test/test.csv
```

### For train_LSTM.py

```bash
python train_LSTM.py ../data/prepared/train/train.csv ../data/prepared/test/test.csv
```

### Open MLFLOW UI

```bash
mlflow server --host 127.0.0.1 --port 8080
```