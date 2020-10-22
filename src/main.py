import loader as ldr

path_queries = r"B:\document_parser\document_parses\topics-rnd5.xml"
path_texts = "B:\document_parser\document_parses\pdf_json"

queries = ldr.load_queries(path_queries)
docs = ldr.load_corpus_parallel(path_texts)