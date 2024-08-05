import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import faiss
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import re
from config import DEVICE, MODEL_SAVE_PATH, RUSSIAN_STOPWORDS, MORPH
import os


class SemanticSearchEngine:
    def __init__(self, blocks, model_name='DeepPavlov/rubert-base-cased-sentence', model_save_path='./saved_model', precomputed_embeddings=None):
        self.blocks = blocks
        self.model_name = model_name
        self.model_save_path = model_save_path
        self.model = self._load_or_create_model()
        if precomputed_embeddings is not None:
            self.block_embeddings = torch.tensor(precomputed_embeddings, device=DEVICE)
        else:
            self.block_embeddings = self._compute_block_embeddings()

    def _load_or_create_model(self):
        #return SentenceTransformer(model_name).to(DEVICE)
        if os.path.exists(self.model_save_path):
            print(f"Loading existing model from {self.model_save_path}")
            return SentenceTransformer(self.model_save_path)
        else:
            print(f"Creating new model {self.model_name}")
            model = SentenceTransformer(self.model_name)
            model.save(self.model_save_path)
            return model

    def _compute_block_embeddings(self):
        texts = [item[0] if isinstance(item, tuple) and len(item) >= 1 else item for item in self.blocks]
        return self.model.encode(texts, convert_to_tensor=True, device=DEVICE)

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        return ' '.join([MORPH.parse(token)[0].normal_form for token in tokens if token not in RUSSIAN_STOPWORDS])

    def extract_key_terms(self, text):
        preprocessed = self.preprocess_text(text)
        tokens = word_tokenize(preprocessed)
        return [token for token in tokens if token not in RUSSIAN_STOPWORDS]

    def semantic_search(self, query, top_k=5):
        query = self.preprocess_text(query)
        query_embedding = self.model.encode([query], convert_to_tensor=True, device=DEVICE)
        similarities = torch.nn.functional.cosine_similarity(query_embedding, self.block_embeddings)
        top_indices = similarities.argsort(descending=True)[:top_k].cpu().numpy()
        return [(self.blocks[i], similarities[i].item()) for i in top_indices]


class AdvancedSemanticSearchEngine:
    def __init__(self, documents, model_name='sberbank-ai/sbert_large_nlu_ru'):
        self.documents = documents
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.document_embeddings = self._compute_document_embeddings()
        self.index = self._build_faiss_index()
        self.bm25 = self._build_bm25_index()

    def _compute_document_embeddings(self):
        embeddings = []
        batch_size = 32
        for i in range(0, len(self.documents), batch_size):
            batch = self.documents[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(
                DEVICE)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu())
        return torch.cat(embeddings)

    def _build_faiss_index(self):
        d = self.document_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(self.document_embeddings.numpy())
        return index

    def _build_bm25_index(self):
        tokenized_docs = [self.preprocess_text(doc).split() for doc in self.documents]
        return BM25Okapi(tokenized_docs)

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        return ' '.join([MORPH.parse(token)[0].normal_form for token in tokens if token not in RUSSIAN_STOPWORDS])

    def expand_query(self, query):
        expanded_terms = set(self.preprocess_text(query).split())
        for term in list(expanded_terms):
            synonyms = self.get_synonyms(term)
            expanded_terms.update(synonyms)
        return ' '.join(expanded_terms)

    def search(self, query, k=10):
        expanded_query = self.expand_query(query)
        query_embedding = self._compute_embedding(expanded_query)
        _, candidate_ids = self.index.search(query_embedding.unsqueeze(0).numpy(), k * 2)
        candidates = [self.documents[id] for id in candidate_ids[0]]
        reranked_candidates = self._rerank_candidates(expanded_query, candidates)
        return reranked_candidates[:k]

    def _compute_embedding(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu()

    def _rerank_candidates(self, query, candidates):
        semantic_scores = self._compute_semantic_similarity(query, candidates)
        bm25_scores = self.bm25.get_scores(query.split())
        combined_scores = semantic_scores * 0.7 + bm25_scores * 0.3
        reranked = sorted(zip(candidates, combined_scores), key=lambda x: x[1], reverse=True)
        return reranked

    def _compute_semantic_similarity(self, query, candidates):
        query_embedding = self._compute_embedding(query)
        candidate_embeddings = self._compute_document_embeddings(candidates)
        return torch.nn.functional.cosine_similarity(query_embedding, candidate_embeddings)

    def get_synonyms(self, term):
        # Placeholder for synonym retrieval
        return []
