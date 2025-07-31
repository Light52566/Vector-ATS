from pdfreader import readPDF


def main():
    pdf_file = "samples/cv-template-2.pdf"
    text = readPDF("pdfminer.six", pdf_file)
    with open("out/pdfreader-output.txt", "w") as output_file:
        output_file.write(text)


if __name__ == "__main__":
    main()
