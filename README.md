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
