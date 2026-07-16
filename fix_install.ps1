$ErrorActionPreference = "Stop"

Write-Host "Downgrading setuptools and wheel..."
.\.venv\Scripts\python -m pip install setuptools==65.5.0 wheel==0.38.4

Write-Host "Installing gym with no-build-isolation..."
.\.venv\Scripts\python -m pip install gym==0.21.0 --no-build-isolation

Write-Host "Installing other dependencies..."
.\.venv\Scripts\python -m pip install sacred==0.8.4 numpy==1.26.4 PyYAML==6.0.1 tensorboard==2.15.1
.\.venv\Scripts\python -m pip install torch==2.1.2+cpu torchvision==0.16.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

Write-Host "All done!"
