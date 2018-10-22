from __future__ import print_function

import pdb
import sys
import time
import faiss
import pickle
import os.path
import argparse
import linecache
import numpy as np
import more_itertools as mit

from aux_retrieve_faiss import *
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------- Auxi ------------------------------------#

## Vectors ----------------------------------------------------------

def single_sent_vector(sentence, model, stopwords, tfidf_weights):
    """ Returns the model input vector for a given
        sentence """

    input_vec = np.zeros((model.vector_size,))
    nr_words = 0
    # Build the sentence vector of the input sentence
    for word in sentence:
        try:
            if word not in stopwords:
                if word in tfidf_weights:
                    w = tfidf_weights[word]
                else:
                    w = 1
                
                input_vec = np.add(input_vec, w*model.wv[word])
                nr_words += 1
        except:
            pass

    if nr_words <= 0:
        pass
    else:
        input_vec = np.divide(input_vec, nr_words)

    input_vec = input_vec.reshape(1,-1)
    return input_vec

def corpus_vectors(sentence_path, model, stopwords, tfidf_weights):
    """ Returns a matrix with the model vector for each
        sentence of a given corpus """

    nr_sents = sum(1 for line in open(sentence_path))
    dim = model.vector_size
    sent_vectors = np.zeros((nr_sents, dim))

    for ix, line in enumerate(open(sentence_path, 'r')):
        progress(ix, nr_sents)
        sent_vectors[ix] = single_sent_vector(line.split(), model, 
                                              stopwords, tfidf_weights)

    sent_vectors = sent_vectors.astype('float32')

    return sent_vectors

def get_tfidf_weights(path):
    """ Returns a dictionary with the tdidf weight for every word """

    def _tokenizer(string):
        return string.split()

    vectorizer = TfidfVectorizer(tokenizer=_tokenizer)
    vectorizer.fit_transform(open(path, 'r'))
    idf = vectorizer.idf_
    
    return dict(zip(vectorizer.get_feature_names(), idf))

## Distance and Similarities ----------------------------------------

def similarity_score(ins, rets):
    """
    ins - input sentence
    rets - retrieved sentence
    """
    return 1 - (float(levenshtein(ins, rets))/max(len(ins), len(rets)))


## Translation Pieces -----------------------------------------------

def get_translation_pieces(input_sentence, retrieved_sentence, target_sentence,
                           alignments, n_max):
    """ Returns list of all the translation pieces """

    # Auxiliary -------------------------------------------------------

    def _create_alignment_dict(alignments):
        
        alignment_dict = dict()

        for alignment in alignments:

            src_ix = int(alignment.split('-')[0])
            tgt_ix = int(alignment.split('-')[1])

            if src_ix in alignment_dict:
                alignment_dict[src_ix].append(tgt_ix)
            else:
                alignment_dict[src_ix] = list()
                alignment_dict[src_ix].append(tgt_ix)

        return alignment_dict

    def _get_unedited_aligned_words(in_sent, ret_sent, alignments):

        aligned_target_ixs = list()

        for j, word in enumerate(ret_sent):
            if word in in_sent:
                try:
                    for aligned_tgt_ix in set(alignments[j]):
                        if aligned_tgt_ix not in aligned_target_ixs:
                            aligned_target_ixs.append(aligned_tgt_ix)
                except:
                    pass

        # Sort the target indexes
        aligned_target_ixs.sort()
        return aligned_target_ixs

    def _get_ngram_list(aligned_target_ixs, grouped_list, n_max):
        n_gram_list = [[x] for x in aligned_target_ixs]

        if n_max > 1:
            for group in grouped_list:
                if len(group) > 1:
                    for n in range(2, n_max+1):
                        if n > len(group):
                            break
                        for j in range(0, len(group)-n+1):
                            n_gram_list.append(group[j:j+n])

        return n_gram_list

    def _get_translation_pieces(n_gram_list, target_sentence):
        translation_pieces = list()

        for sublist in n_gram_list:
            aux_list = list()

            for ix in sublist:
                aux_list.append(target_sentence[ix])

            translation_pieces.append(aux_list)

        return translation_pieces
    

    # Main ---------------------------------------------------------------

    # Alignments SRC(retrieved)-TGT
    alignment_dict = _create_alignment_dict(alignments)

    # Find unedited and respective aligned words
    aligned_target_ixs = _get_unedited_aligned_words(input_sentence,
                                                     retrieved_sentence,
                                                     alignment_dict)
                                                     
    
    # Create the grouped list
    grouped_list = [list(group) for group
                                in mit.consecutive_groups(aligned_target_ixs)]

    # Produce the n-gram list
    n_gram_list = _get_ngram_list(aligned_target_ixs, grouped_list, n_max)

    # Translate the n_gram_list to actual words
    translation_pieces = _get_translation_pieces(n_gram_list, target_sentence)

    return translation_pieces


def get_weighted_translation_pieces(G_X_m, G_X, G_X_scores, simi):

    for translation_piece in G_X_m:
        if translation_piece in G_X:
            for tuple_ in G_X_scores:
                if translation_piece == tuple_[0]:
                    if simi > tuple_[1]:
                        G_X_scores.remove(tuple_)
                        G_X_scores.append((translation_piece, simi))
                    break
        else:
            G_X.append(translation_piece)
            G_X_scores.append((translation_piece, simi))

    return G_X_m, G_X, G_X_scores

## Similarity Distribution ------------------------------------------

def update_simi_distribution(simi, simi_distribution):
    digit = int(str(simi).split('.')[0])
    decimal = int(str(simi).split('.')[1][0])

    if digit:
        simi_distribution[10] += 1
    else:
        simi_distribution[decimal] += 1

    return simi_distribution

def simi_distribution_percentage(simi_distribution):
    total = 0
    for v in simi_distribution:
        total += v
    simi_distribution_percentage = list()
    for v in simi_distribution:
        simi_distribution_percentage.append(100.0*v/total)

    return simi_distribution_percentage


## Utilities --------------------------------------------------------

def save_pickle(item, path):
    with open(path, 'wb') as handle:
        pickle.dump(item, handle)

def load_pickle(path):
    with open(path, 'rb') as handle:
        x = pickle.load(handle)
    return x

def print_time(st):
    print("\nProcess took {:.2f} seconds\n".format(time.time()-st))

def file_to_list(path):
    return [x.split()[0] for x in open(path, 'r')]

def file_to_gen(path):
    return (line.split()[0] for line in open(path, 'r'))

#https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 

# ---------------------------------- Main ------------------------------------#
def main(k, n_max, simi_threshold, dev, verbose=False):

    print("Creating translation pieces")
    print("k: ", k)
    print("n_max: ", n_max)
    print("simi_threshold: ", simi_threshold)
    print("dev: ", dev)

    # UPDATE PATHS ------------------------------------------------------------
    src = "de"
    tgt = "en"
    base_path = "/mnt/in-domain/es-en-md"
    guided_path = "/home/ubuntu/guided_nmt"
    
    # NOTE: Only src corpus for query and the extra sentece have merged paths.
    # Full words are used when creating the averaged sentence embedding.
    if dev:
        src_dev_bpe_path = base_path + "/dev.bpe." + src
        src_dev_mrg_path = base_path + "/dev." + src
    else:
        src_test_bpe_path = base_path + "/test.bpe." + src
        src_test_mrg_path = base_path + "/test." + src

    src_extra_bpe_path = base_path + "/extra.bpe." + src
    src_extra_mrg_path = base_path + "/extra." + src
    tgt_extra_bpe_path = base_path + "/extra.bpe." + tgt
    
    join_extra_mrg_path = base_path + "/all." + src
   
    stopwords_path = guided_path + "/stopwords/stopwords." + src
    
    ft_model = "/mnt/ft/wiki." + src + ".bin"
    
    extra_align_path = base_path + "/alignments/extra_data.align"
    tp_path = "/mnt/translation_pieces/" + base_path.split("/")[-1]
    # -------------------------------------------------------------------------

    # Load the FastText model
    print("Loading the model...")
    st = time.time()
    model = FastText.load_fasttext_format(ft_model)
    print_time(st)
    print("Loaded model:", model)

    # Create list with all the stopwords
    print("Creating list of stopwords...")
    st = time.time()
    stopwords = file_to_list(stopwords_path)
    print_time(st)

    # Create the tfidf weights for all the values
    tfidf_weights = get_tfidf_weights(join_extra_mrg_path)

    # Create the vectors for each sentence in the source extra corpus
    print("Creating the vector for each extra sentence...")
    st = time.time()
    if os.path.exists(tp_path + "/extra_src_vectors.pickle"):
        print("Already exists. Loading...")
        extra_src_vectors = load_pickle(tp_path + "/extra_src_vectors.pickle")
    else:
        print(src_extra_mrg_path)
        extra_src_vectors = corpus_vectors(src_extra_mrg_path, model, stopwords, tfidf_weights)
        save_pickle(extra_src_vectors, tp_path + "/extra_src_vectors.pickle")
    print("Number of extra sentence vectors: ", len(extra_src_vectors))
    print_time(st)

    # Create the vectors for each sentence in the source test/dev corpus
    st = time.time()

    if dev:
        print("Creating the vector for each dev sentence...")
        if os.path.exists(tp_path + "/dev_src_vectors.pickle"):
            print("Already exists. Loading...")
            dev_src_vectors = load_pickle(tp_path + "/dev_src_vectors.pickle")
        else:
            print(src_dev_mrg_path)
            dev_src_vectors = corpus_vectors(src_dev_mrg_path, model, stopwords, tfidf_weights)
            save_pickle(dev_src_vectors, tp_path + "/dev_src_vectors.pickle")
        print("\nNumber of dev sentence vectors: ", len(dev_src_vectors))
    else:
        print("Creating the vector for each test sentence...")
        if os.path.exists(tp_path + "/test_src_vectors.pickle"):
            print("Already exists. Loading...")
            test_src_vectors = load_pickle(tp_path + "/test_src_vectors.pickle")
        else:
            print(src_test_mrg_path)
            test_src_vectors = corpus_vectors(src_test_mrg_path, model, stopwords, tfidf_weights)
            save_pickle(test_src_vectors, tp_path + "/test_src_vectors.pickle")
        print("\nNumber of test sentence vectors: ", len(test_src_vectors))

    print_time(st)

    # Initialize the index, in this case using L2 distance
    index = faiss.IndexFlatL2(model.vector_size)

    # Add the databse to the index
    index.add(extra_src_vectors)

    # Search for the test corpus similar sentences
    print("Retrieving similar sentences...")
    st = time.time()
    if dev:
        retrieved_sentence_distance, retrieved_sentence_indexes = index.search(dev_src_vectors, k)
    else:
        retrieved_sentence_distance, retrieved_sentence_indexes = index.search(test_src_vectors, k)
    print_time(st)

    # Main cycle
    print("Creating translation pieces...")
    test_translation_pieces = list()
    similarity_distribution_all = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    similarity_distribution_max = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    st = time.time()

    # Initialize the length to use in the progress
    if dev:
        progress_tot_len = len(dev_src_vectors)
    else:
        progress_tot_len = len(test_src_vectors)

    # Main cycle to create tps
    for test_sentence_ix, similar_sentence_indexes in enumerate(retrieved_sentence_indexes):

        # Just change to avoid accessing something that doesn't exist. The ix is the same.
        progress(test_sentence_ix, progress_tot_len)

        G_X = list()
        G_X_scores = list()

        # Get the list of the test/dev sentence words
        if dev:
            X = linecache.getline(src_dev_bpe_path, test_sentence_ix + 1)
        else:
            X = linecache.getline(src_test_bpe_path, test_sentence_ix + 1)

        # To obtain the max similarity value for each sentence
        max_simi = -1

        if verbose:
            print("\nSentence ", test_sentence_ix + 1)
            print("In: ", X)

        # The input sentence is splitted before hand to save time
        X_splitted = X.split()

        # Obtain the final translation pieces
        for retr_sentence_ix in similar_sentence_indexes:
            # Get the list of the retrieved sentence words
            X_m = linecache.getline(src_extra_bpe_path, retr_sentence_ix + 1)

            # Sentence similarity
            simi = similarity_score(X, X_m)

            if simi > max_simi:
                max_simi = simi

            # Update the similarity distribution
            similarity_distribution_all = update_simi_distribution(simi, similarity_distribution_all)

            # TEST - If the similarity is low, dont't bother fetching anything
            if simi < simi_threshold:
                continue

            if verbose:
                print("Ret: ", X_m)
                print("Simi: ", simi)

            # Get the translation pieces
            G_X_m = get_translation_pieces(X_splitted,
                                X_m.split(),
                                linecache.getline(tgt_extra_bpe_path, retr_sentence_ix + 1).split(),
                                linecache.getline(extra_align_path, retr_sentence_ix + 1).split(),
                                n_max)

            # Add the obtained translation pieces to the final list for this specific input sentence
            G_X_m, G_X, G_X_scores = get_weighted_translation_pieces(G_X_m, G_X, G_X_scores, simi)

        if verbose:
            print("Translation pieces: ", G_X_scores)

        similarity_distribution_max = update_simi_distribution(max_simi, similarity_distribution_max)

        test_translation_pieces.append(G_X_scores)

    print_time(st)
    print("Produced {} sets of translation pieces".format(len(test_translation_pieces)))
    
    # Print similarity informations
    print("Similarity Score Distribution: ", similarity_distribution_all)
    print("Similarity Score Distribution Percentage: ", simi_distribution_percentage(similarity_distribution_all))
    
    print("Similarity Score Distribution: ", similarity_distribution_max)
    print("Similarity Score Distribution Percentage: ", simi_distribution_percentage(similarity_distribution_max))

    # Save the lists we need for guiding the nmt
    a = str(simi_threshold).split('.')[0]
    b = str(simi_threshold).split('.')[1]

    if dev:
        name = tp_path + "/dev_translation_pieces_" + str(k) + "-th" + a + "pt" + b + ".pickle"
    else:
        name = tp_path + "/test_translation_pieces_" + str(k) + "-th" + a + "pt" + b + ".pickle"
    
    save_pickle(test_translation_pieces, name)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, required=True,
                        help="Number of retrieved sentences")
    parser.add_argument("-n_max", type=int, required=True,
                        help="Maximum number of n-grams")
    parser.add_argument("-simi_th", type=float, required=True,
                        help="Cutoff similarity")
    parser.add_argument("-dev", action='store_true',
                        help="Use if using the dev set")
    parser.add_argument("-verbose", action="store_true",
                        help="Print extra information")

    args = parser.parse_args() 

    main(args.k, args.n_max, args.simi_th, args.dev, args.verbose)
