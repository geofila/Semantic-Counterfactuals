from nltk.corpus import wordnet

def find_maximum_path(paths):
    max_path, max_length = paths[0], len(paths[0])
    for path in paths[1:]:
        if len(path) > max_length:
            max_length = len(path)
            max_path = path
    return [p.name() for p in max_path]



def _connect_term_to_wordnet(term):
    synset = wordnet.synsets(term)
    if len(synset) == 0:
        raise (Exception("Term not found in WordNet: " + term))
    synset = synset[0]
    return set(find_maximum_path(synset.hypernym_paths()))

def connect_term_to_wordnet(term):

    if "^" in term:
        term1, term2 = term.split("^")
        return _connect_term_to_wordnet(term1).union(_connect_term_to_wordnet(term2))
    else:
        return _connect_term_to_wordnet(term)


def connect_list_to_wordnet(list_of_terms):
    return [connect_term_to_wordnet(term) for term in list_of_terms]



# def enrich_query(query):
#     enriched_query = []
#     for concept in concepts:
#         synset = wordnet.synsets(list(concept)[0])[0]
#         enriched_query.append(find_maximum_path(synset.hypernym_paths()))
#     return enriched_query
