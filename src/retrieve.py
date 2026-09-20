def retrieve_chunks(query, vector_store, k=4):
    if not query or not vector_store:
        return []

    q = query.strip()
    if not q:
        return []

    return vector_store.similarity_search(q, k=k)
