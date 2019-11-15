import collections
import math


class Tfidf:

    def get_tfidf(self, document, corpus):
        tf = self.get_tf(document)
        corpus_frequencies = [self.get_tf(doc) for doc in corpus]
        idf = self.get_idf(tf, corpus_frequencies)
        return self.get_tfidf(tf, idf)

    def get_tf(self, document):
        """
        :param document: text as a string
        Count Term Frequency for types (a.k.a. unique words) in a document
        TF equation: (frequency of type)/(maximum frequency of any type)
        :return: dictionary where k=type and v=TF
        """
        list_with_words = document.split(" ")
        counted = collections.Counter(list_with_words)
        max_occurencies = max(counted.values())
        tf = {word: (count / max_occurencies) for (word, count) in counted.items()}
        return tf

    def get_idf(self, document, corpus, remove_duplicate=True):
        """
        :param document: dictionary with k: word and v: frequency
        :param corpus: list with dictionaries with k: word and v: frequency
        :param remove_duplicate: if True then first entry in corpus identical to the document is removed, if found
        Count Inverse Document Frequency for every type in document
        IDF equation: log2(number of documents in my corpus/number doc in corpus containing the word)
        :return: dictionary where k=type and v=IDF
        """
        corpus_for_analysis = corpus.copy()
        if remove_duplicate == True:
            if document in corpus:
                corpus_for_analysis.remove(document)
        idf = {}
        number_of_docs = len(corpus) + 1  # to avoid division by zero issues
        # I start my doc_occurences with 1 instead of 0. \
        # If I divide a doc_occurences with a bigger value the log of the quotient would be negative.
        for word, frequency in document.items():
            corpus_occurrences = 1  # to avoid division by zero issues
            for doc in corpus_for_analysis:
                if word in doc:
                    corpus_occurrences += 1
            idf[word] = math.log2(number_of_docs/corpus_occurrences)
        return idf

    def count_tfidf(self, tf, idf):
        """
        :param tf: dictionary where k=type and v=TF
        :param idf: dictionary where k=type and v=IDF
        Multiply every type's tf with it's idf and get tf*idf. Create tuples (type, tfidf-value).
        :return: append tuples to self.sorted_tfidf, sorted after size of tfidf-value in falling order.
        """
        tfidf = {}
        for word, value in tf.items():
            tfidf[word] = value * idf[word]
        # Get k-v-tuples from dictionary, sort by tuple[1] in falling order. Save in order to list with tuples:
        sorted_tfidf = sorted(tfidf.items(), key=lambda tfidf_value: tfidf_value[1], reverse=True)
        return sorted_tfidf
