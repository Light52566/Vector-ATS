from pdfreader import readPDF
from embedding import generateEmbeddings


def main():
    pdf_file = "samples/cv-template-2.pdf"
    text = readPDF("pdfminer.six", pdf_file)
    with open("out/pdfreader-output.txt", "w") as output_file:
        output_file.write(text)
    with open("out/embeddings.txt", "w") as embedding_file:
        embedding = generateEmbeddings(text, model='intfloat/e5-small-v2')
        embedding_file.write(str(embedding.tolist()))  # Convert to list for better readability
        


if __name__ == "__main__":
    main()
