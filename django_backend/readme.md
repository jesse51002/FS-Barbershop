# Setup
```sh
# clone and cd
git clone https://github.com/jesse51002/FS-DjangoBackend.git && cd FS-DjangoBackend

# create conda environment
conda create -n DjangoBackend python=3.12

# activate the environment
conda activate DjangoBackend

# install dependencies
pip install -r requirements.txt
```


# Running the server
```sh
python manage.py runserver
```

# Running Tests
```sh
python manage.py test --pattern="tests_*.py"    
```
