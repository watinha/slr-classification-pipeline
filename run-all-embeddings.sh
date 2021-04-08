rm result/*.csv

python3 main.py games embeddings_glove 3 false 200 tfidf
python3 main.py games embeddings_se 3 false 200 tfidf

python3 main.py illiterate embeddings_glove 3 false 200 tfidf
python3 main.py illiterate embeddings_se 3 false 200 tfidf

python3 main.py mdwe embeddings_glove 3 false 200 tfidf
python3 main.py mdwe embeddings_se 3 false 200 tfidf

python3 main.py ontologies embeddings_glove 3 false 200 tfidf
python3 main.py ontologies embeddings_se 3 false 200 tfidf

python3 main.py pair embeddings_glove 3 false 200 tfidf
python3 main.py pair embeddings_se 3 false 200 tfidf

python3 main.py slr embeddings_glove 3 false 200 tfidf
python3 main.py slr embeddings_se 3 false 200 tfidf

python3 main.py testing embeddings_glove 3 false 200 tfidf
python3 main.py testing embeddings_se 3 false 200 tfidf

python3 main.py xbi embeddings_glove 3 false 200 tfidf
python3 main.py xbi embeddings_se 3 false 200 tfidf

