rm result/*.csv

python3 main.py games svm 3 true 20 tfidf
python3 main.py games dt 3 true 20 tfidf
python3 main.py games svm 3 false 20 embeddings_glove
python3 main.py games svm 3 false 20 embeddings_se
python3 main.py games dt 3 false 20 embeddings_glove
python3 main.py games dt 3 false 20 embeddings_se

python3 main.py illiterate svm 3 true 20 tfidf
python3 main.py illiterate dt 3 true 20 tfidf
python3 main.py illiterate svm 3 false 20 embeddings_glove
python3 main.py illiterate svm 3 false 20 embeddings_se
python3 main.py illiterate dt 3 false 20 embeddings_glove
python3 main.py illiterate dt 3 false 20 embeddings_se

python3 main.py mdwe svm 3 true 20 tfidf
python3 main.py mdwe dt 3 true 20 tfidf
python3 main.py mdwe svm 3 false 20 embeddings_glove
python3 main.py mdwe svm 3 false 20 embeddings_se
python3 main.py mdwe dt 3 false 20 embeddings_glove
python3 main.py mdwe dt 3 false 20 embeddings_se

python3 main.py ontologies svm 3 true 20 tfidf
python3 main.py ontologies dt 3 true 20 tfidf
python3 main.py ontologies svm 3 false 20 embeddings_glove
python3 main.py ontologies svm 3 false 20 embeddings_se
python3 main.py ontologies dt 3 false 20 embeddings_glove
python3 main.py ontologies dt 3 false 20 embeddings_se

python3 main.py pair svm 3 true 20 tfidf
python3 main.py pair dt 3 true 20 tfidf
python3 main.py pair svm 3 false 20 embeddings_glove
python3 main.py pair svm 3 false 20 embeddings_se
python3 main.py pair dt 3 false 20 embeddings_glove
python3 main.py pair dt 3 false 20 embeddings_se

python3 main.py slr svm 3 true 20 tfidf
python3 main.py slr dt 3 true 20 tfidf
python3 main.py slr svm 3 false 20 embeddings_glove
python3 main.py slr svm 3 false 20 embeddings_se
python3 main.py slr dt 3 false 20 embeddings_glove
python3 main.py slr dt 3 false 20 embeddings_se

python3 main.py testing svm 3 true 20 tfidf
python3 main.py testing dt 3 true 20 tfidf
python3 main.py testing svm 3 false 20 embeddings_glove
python3 main.py testing svm 3 false 20 embeddings_se
python3 main.py testing dt 3 false 20 embeddings_glove
python3 main.py testing dt 3 false 20 embeddings_se

python3 main.py xbi svm 3 true 20 tfidf
python3 main.py xbi dt 3 true 20 tfidf
python3 main.py xbi svm 3 false 20 embeddings_glove
python3 main.py xbi svm 3 false 20 embeddings_se
python3 main.py xbi dt 3 false 20 embeddings_glove
python3 main.py xbi dt 3 false 20 embeddings_se
