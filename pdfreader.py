from markitdown import MarkItDown
from pdfminer.high_level import extract_text
import pdfplumber
import pymupdf

pdf_libraries = (
    "markitdown",
    "pdfminer.six",
    "pdfplumber",
    "pymupdf",
)


def readPDF(library_name, file_path) -> str:
    """
    Reads a PDF file using the specified library and returns the extracted text.
    :param library_name: Name of the library to use for reading the PDF.
    :param file_path: Path to the PDF file.
    :return: Extracted text from the PDF file.
    """
    if library_name == "markitdown":
        md = MarkItDown(enable_plugins=False)
        result = md.convert(file_path)
        return result.markdown
    elif library_name == "pdfminer.six":
        return extract_text(file_path)
    elif library_name == "pdfplumber":
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            return text
    elif library_name == "pymupdf":
        with pymupdf.open(file_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()  # type: ignore
            return text
    return ""


def main():
    # MarkItDown conversion
    md = MarkItDown(enable_plugins=False)
    result = md.convert("samples/cv-template-2.pdf")
    with open("out/markitdown-output.md", "w") as output_file:
        output_file.write(result.markdown)

    # pdfminer.six conversion
    text = extract_text("samples/cv-template-2.pdf")
    with open("out/pdfminer-output.txt", "w") as output_file:
        output_file.write(text)

    # pdfplumber conversion
    with pdfplumber.open("samples/cv-template-2.pdf") as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    with open("out/pdfplumber-output.txt", "w") as output_file:
        output_file.write(text)

    # pymupdf conversion
    with pymupdf.open("samples/cv-template-2.pdf") as doc:  # open a document
        with open("out/pymupdf-output.txt", "wb") as out:  # create a text output
            for page in doc:  # iterate the document pages
                text = page.get_text().encode("utf8")  # type: ignore # get plain text (is in UTF-8)
                out.write(text)  # write text of page
                out.write(bytes((12,)))  # write page delimiter (form feed 0x0C)


if __name__ == "__main__":
    main()
