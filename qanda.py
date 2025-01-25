import nltk, sys, math, os
from nltk.corpus import stopwords
nltk.download('stopwords')

FILE_MATCHES = 1
SENTENCE_MATCHES = 1

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python qanda.py corpus")
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs =  compute_idfs(file_words)
    
    query = set(tokenize(input('Query: ')))
    
    filenames  = top_files(query, file_words, file_idfs, n=FILE_MATCHES)
    
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens
                    
    idfs = compute_idfs(sentences)
    
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)
        
    
    
def load_files(dirct):
    files = {}
    for filename in os.listdir(dirct):
        if filename.endswith(".txt"):
            with open(os.path.join(dirct, filename), "r", encoding='utf-8') as file:
                files[filename] = file.read()
    return files
    raise NotImplementedError

def tokenize(docs):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(docs.lower())
    return [word for word in words if word.isalnum() and word not in stop_words]
    raise NotImplementedError
    
def compute_idfs(docs):
    idfs = {}
    total_doc = len(docs)
    all_words = set(word for doc in docs.values() for word in doc)
    
    for word in all_words:
        cont_docs = sum(1 for doc in docs.values() if word in doc)
        idfs[word] = math.log(total_doc/cont_docs)
        
    return idfs
    raise NotImplementedError


def top_files(query, files, idfs, n):
    scores = {}
    for filename, words in files.items():
        tf_idf = 0
        for word in query:
            if word in words:
                tf = words.count(word)
                tf_idf += tf *idfs[word]
        scores[filename] = tf_idf
    return sorted(scores, key=scores.get, reverse=True)[:n]
    raise NotImplementedError
    

def top_sentences(query, sentences, idfs, n):
    results = []
    for sentence, words in sentences.items():
        idf_score = sum(idfs[word] for word in query if word in words)
        query_density = sum(word in query for word in words) / len(words)
        results.append((sentence, idf_score, query_density))
    
    results.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [results[0] for result in results[:n]]
    raise NotImplementedError

if __name__ == '__main__':
    main()


