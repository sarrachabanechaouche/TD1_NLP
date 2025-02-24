python src/main.py train --input_filename="data/raw/dataset.csv"  --model_dump_filename="models/model.pkl"

python src\main.py predict  --input_filename "C:\Users\sarra\OneDrive\Bureau\NLP\data\raw\names_train - names_train.csv"  --model_dump_filename "models\model.pkl"  --output_filename "data\processed\prediction.csv"

python src\main.py evaluate  --input_filename "C:\Users\sarra\OneDrive\Bureau\NLP\data\raw\names_train - names_train.csv"