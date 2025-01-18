from md import MarkdownToPDFConverter
from mistune_converter import MistuneMarkdownConverter

Converter = MistuneMarkdownConverter

converter = Converter(
    markdown_dir="./documents",
    output_pdf="documento_final.pdf",
    cover_pdf="cover/coverpage.pdf",  # Agora é um PDF em vez de uma imagem
    title="Material foda de Langchain",
    author="Sérgio Berlotto"
)

converter.convert()
