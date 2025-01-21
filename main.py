from md import MarkdownToPDFConverter
from mistune_converter import MistuneMarkdownConverter

Converter = MistuneMarkdownConverter

converter = Converter(
    markdown_dir="./documents",
    output_pdf="ebook-dominando_rag_e_langchain.pdf",
    cover_pdf="cover/coverpage.pdf",  # Agora é um PDF em vez de uma imagem
    title="Aprenda com funciona LangChain e como criar seu próprio RAG",
    author="Sérgio Berlotto"
)

converter.convert()
