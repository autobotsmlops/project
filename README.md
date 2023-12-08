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

```bash
dvc init
```

```bash
dvc remote add --default myremote \
gdrive://11WDFgw2fv8O6B2GXB1X5ET-tvLfMGd9V
```

```bash 
dvc push datafilename
```

