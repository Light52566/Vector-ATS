from pdfreader import readPDF
from embedding import generateEmbeddings, rankEmbeddings


def main():
    model = 'all-MiniLM-L6-v2'
    pdf_file = "samples/cv-template-2.pdf"
    text = readPDF("pdfminer.six", pdf_file)
    embedding = generateEmbeddings(text, model=model)
    with open("out/pdfreader-output.txt", "w") as output_file:
        output_file.write(text)
    with open("out/embeddings.txt", "w") as embedding_file:
        embedding_file.write(str(embedding.tolist()))  # Convert to list for better readability
    
    ref_pdf_file = "samples/cv-template-1.pdf"
    ref_text = readPDF("pdfminer.six", ref_pdf_file)
    reference_embedding = generateEmbeddings(ref_text, model=model)
    with open("out/reference_embedding.txt", "w") as ref_embedding_file:
        ref_embedding_file.write(str(reference_embedding.tolist()))

    ranked_similarities = rankEmbeddings({"pdf_embedding": embedding}, reference_embedding, metric='cosine', model=model)
    print("Ranked similarities:")
    for key, similarity in ranked_similarities.items():
        print(f"{key}: {similarity}")

if __name__ == "__main__":
    main()