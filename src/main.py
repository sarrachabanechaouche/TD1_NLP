

import argparse
from train import train
from predict import predict
from evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(description="NLP Classification - is_comic")
    subparsers = parser.add_subparsers(dest='command')
    
    # train
    train_parser = subparsers.add_parser('train', help="Entraîner le modèle (split 80/20 en interne)")
    train_parser.add_argument('--input_filename', type=str, required=True, help="CSV avec 'video_name' et 'is_comic'")
    train_parser.add_argument('--model_dump_filename', type=str, required=True, help="Chemin du modèle .pkl à sauvegarder")
    
    # predict
    predict_parser = subparsers.add_parser('predict', help="Prédire sur un CSV")
    predict_parser.add_argument('--input_filename', type=str, required=True, help="CSV d'entrée (doit contenir 'video_name')")
    predict_parser.add_argument('--model_dump_filename', type=str, required=True, help="Modèle entraîné (pipeline) à charger")
    predict_parser.add_argument('--output_filename', type=str, required=True, help="CSV de sortie")
    
    #  evaluate
    eval_parser = subparsers.add_parser('evaluate', help="Évaluation par cross-validation (optionnel)")
    eval_parser.add_argument('--input_filename', type=str, required=True, help="CSV complet pour la CV")
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args.input_filename, args.model_dump_filename)
    elif args.command == 'predict':
        predict(args.input_filename, args.model_dump_filename, args.output_filename)
    elif args.command == 'evaluate':
        evaluate(args.input_filename)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
