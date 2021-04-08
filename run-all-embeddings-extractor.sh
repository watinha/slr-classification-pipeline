rm result/*.csv

python3 main.py games svm 0 3 true 20 embeddings_glove
python3 main.py games svm 0 3 true 20 embeddings_se
python3 main.py games dt 0 3 true 20 embeddings_glove
python3 main.py games dt 0 3 true 20 embeddings_se

python3 main.py illiterate svm 0 3 true 20 embeddings_glove
python3 main.py illiterate svm 0 3 true 20 embeddings_se
python3 main.py illiterate dt 0 3 true 20 embeddings_glove
python3 main.py illiterate dt 0 3 true 20 embeddings_se

python3 main.py mdwe svm 0 3 true 20 embeddings_glove
python3 main.py mdwe svm 0 3 true 20 embeddings_se
python3 main.py mdwe dt 0 3 true 20 embeddings_glove
python3 main.py mdwe dt 0 3 true 20 embeddings_se

python3 main.py ontologies svm 0 3 true 20 embeddings_glove
python3 main.py ontologies svm 0 3 true 20 embeddings_se
python3 main.py ontologies dt 0 3 true 20 embeddings_glove
python3 main.py ontologies dt 0 3 true 20 embeddings_se

python3 main.py pair svm 0 3 true 20 embeddings_glove
python3 main.py pair svm 0 3 true 20 embeddings_se
python3 main.py pair dt 0 3 true 20 embeddings_glove
python3 main.py pair dt 0 3 true 20 embeddings_se

python3 main.py slr svm 0 3 true 20 embeddings_glove
python3 main.py slr svm 0 3 true 20 embeddings_se
python3 main.py slr dt 0 3 true 20 embeddings_glove
python3 main.py slr dt 0 3 true 20 embeddings_se

python3 main.py testing svm 0 3 true 20 embeddings_glove
python3 main.py testing svm 0 3 true 20 embeddings_se
python3 main.py testing dt 0 3 true 20 embeddings_glove
python3 main.py testing dt 0 3 true 20 embeddings_se

python3 main.py xbi svm 0 3 true 20 embeddings_glove
python3 main.py xbi svm 0 3 true 20 embeddings_se
python3 main.py xbi dt 0 3 true 20 embeddings_glove
python3 main.py xbi dt 0 3 true 20 embeddings_se
