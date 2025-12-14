# Docker-based Machine Learning Pipeline

Ez a projekt egy teljes, Docker-alapú gépi tanulási pipeline-t valósít meg, amely a következő lépésekből áll:

1. **Adatelőkészítés** (`01_data_processing.py`)
2. **Modellek tanítása** (`02_train.py`)
   - baseline modellek (mean, current delay, linear regression)
   - GraphSAGE-alapú GNN
3. **Kiértékelés** (`03_evaluation.py`)
   - MAE, RMSE
   - threshold-alapú accuracy (60 / 120 / 180 másodperc)

A pipeline automatikusan lefut a Docker konténer indításakor a `run.sh` script segítségével.

---

## Futtatás

A teljes pipeline egyetlen paranccsal indítható:

```bash
docker build -t my-dl-project-work-app:1.0 .

docker run --rm --gpus all -v "%cd%\data:/data" -v "%cd%\output:/app/output" my-dl-project-work-app:1.0 > training_log.txt 2>&1
```

# Megjegyzés az adatmennyiségről

- A teljes eredeti adathalmaz több millió rekordot tartalmaz, amelynek teljes körű feldolgozása és GNN-alapú tanítása jelentős futásidőt igényel.
- A futásidő ésszerű keretek között tartása érdekében a tanítási fázisban az elérhető adatmennyiséget szándékosan jelentősen redukáltam, miközben a pipeline felépítése, a modellek struktúrája és az értékelési logika változatlan maradt.
- A modell így is a legtöbb baseline-tól jobban teljesít, illetve elmondható hogy a statisztikai és heurisztikus modellek nem javulnának akkora mértékben a több adattól, mint a GNN-n ezzel is mutatva, hogy jobban teljesít.

