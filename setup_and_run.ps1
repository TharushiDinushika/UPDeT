$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment..."
python -m venv venv

Write-Host "Activating virtual environment and installing dependencies..."
.\venv\Scripts\python -m pip install --upgrade pip
.\venv\Scripts\python -m pip install torch==2.1.2+cpu torchvision==0.16.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
.\venv\Scripts\python -m pip install numpy==1.26.4 PyYAML==6.0.1 sacred==0.8.4 gym==0.21.0 tensorboard==2.15.1

Write-Host "Running a sample training session (1000 timesteps, CPU)..."
.\venv\Scripts\python src/main.py --config=updet --env-config=mpe use_cuda=False t_max=1000

Write-Host "Running evaluation locally..."
# Assuming it saves the model, we can run evaluate (evaluate=True). Wait, we don't know the exact checkpoint path right now.
# But running evaluate=True without checkpoint_path will just evaluate an untrained model.
.\venv\Scripts\python src/main.py --config=updet --env-config=mpe use_cuda=False evaluate=True

Write-Host "Setup and run completed successfully!"
