# Disaster-Zone Rescue Simulation (CPS5002)

## Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Run single simulation (creates run1 logs + plots)
python main.py --mode single --steps 300 --seed 42 --out out/run1

## Run batch simulations (creates batch csv + summary plot)
python main.py --mode batch --runs 20 --steps 300 --seed 42 --out out/batch
