from vechord import DataLoader, TextFile, VectorChordClient

if __name__ == "__main__":
    namespace = "local_pdf"
    client = VectorChordClient("postgresql://postgres:postgres@172.17.0.1:5432/")
    client.create_namespace(namespace)
    for file in DataLoader().local_files("data"):
        text_file = TextFile.from_filepath(file)
        client.insert_text(namespace, text_file)

    res = client.query(namespace=namespace, query="vector search", topk=5)
    print(res)
