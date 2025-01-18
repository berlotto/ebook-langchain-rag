import os
from pathlib import Path
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
from bs4 import BeautifulSoup
import re
from PyPDF2 import PdfWriter, PdfReader
import pygments
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import base64
import requests
import json

class MarkdownToPDFConverter:
    def __init__(self, markdown_dir, output_pdf, cover_pdf, title, author):
        self.markdown_dir = Path(markdown_dir)
        self.output_pdf = output_pdf
        self.cover_pdf = cover_pdf
        self.title = title
        self.author = author
        self.current_page = 4  # Começa na página 4 (após capa, título e sumário)
        self.toc_entries = []

    def render_mermaid(self, diagram):
        """Renderiza um diagrama Mermaid como PNG com largura responsiva."""
        try:
            url = "https://mermaid.ink/img/"
            
            # Encode the diagram using UTF-8 first, then encode to base64
            graphbytes = diagram.encode("utf-8")
            base64_bytes = base64.b64encode(graphbytes)
            base64_string = base64_bytes.decode("ascii")
            
            # Get the SVG URL
            img_url = f"{url}{base64_string}?type=png"
            
            # Return the image HTML with max-width set to 100%
            return f'<img src="{img_url}" alt="Mermaid Diagram" style="display: block; margin: 20px auto; max-width: 100%; height: auto;">'
        except Exception as e:
            print(f"Erro ao renderizar diagrama Mermaid: {e}")
            return f'<pre>{diagram}</pre>'
            
    def process_mermaid(self, content):
        """Processa blocos Mermaid no markdown."""
        pattern = r'```mermaid\n(.*?)\n```'
        
        def replace_mermaid(match):
            diagram = match.group(1)
            return self.render_mermaid(diagram)
        
        return re.sub(pattern, replace_mermaid, content, flags=re.DOTALL)

    def highlight_code(self, match):
        """Aplica syntax highlighting ao código encontrado."""
        code = match.group(2)
        language = match.group(1) or 'text'
        
        if language.lower() == 'mermaid':
            return self.render_mermaid(code)
        
        try:
            lexer = get_lexer_by_name(language)
            formatter = HtmlFormatter(style='monokai', cssclass='highlight')
            highlighted = pygments.highlight(code, lexer, formatter)
            return highlighted
        except:
            return f'<pre><code>{code}</code></pre>'

    def process_markdown_content(self, content):
        # Primeiro normaliza as listas aninhadas
        lines = content.split('\n')
        processed_lines = []
        
        for line in lines:
            if line.strip().startswith('- '):
                # Conta os espaços antes do hífen
                indent = len(line) - len(line.lstrip())
                # Converte espaços em múltiplos de 2 para garantir hierarquia
                processed_lines.append('  ' * (indent // 2) + line.lstrip())
            else:
                processed_lines.append(line)
        
        content = '\n'.join(processed_lines)
        
        # Processa Mermaid e código como antes
        content = self.process_mermaid(content)
        pattern = r'```(\w+)?\n(.*?)\n```'
        processed = re.sub(pattern, self.highlight_code, content, flags=re.DOTALL)
        
        # Converte markdown com configurações específicas
        html = markdown.markdown(
            processed,
            extensions=[
                'tables', 
                'md_in_html', 
                'extra'
            ]
        )
        
        return html

    def generate_toc(self):
        """Gera o sumário com números de página."""
        toc = [
            "<html><body>",
            "<div style='height: 100vh; padding: 00px;'>",
            "<h1 style='margin-bottom: 30px;'>Sumário</h1>",
            "<nav style='width: 100%;'>",
            "<table style='width: 100%; border: none;'>"
        ]
        
        for entry in self.toc_entries:
            indent = (entry['level'] - 1) * 20
            dots = '.' * 50  # Aumenta o número de pontos
            
            toc.append(
                f"""
                <tr>
                    <td style="border: none; padding: 5px 0;">
                        <div style="margin-left: {indent}px;">
                            <a href="#page_{entry['page']}" style="text-decoration: none; color: #000;">
                                {entry['text']}
                            </a>
                        </div>
                    </td>
                    <td style="border: none; text-align: right; padding: 5px 0; white-space: nowrap;">
                        <span style="color: #000; padding-left: 10px;">{dots}</span>
                        <span style="color: #000;">{entry['page']}</span>
                    </td>
                </tr>
                """
            )
        
        toc.extend(["</table>", "</nav>", "</div>", "</body></html>"])
        return "\n".join(toc)

    def merge_pdfs(self, content_pdf):
        merger = PdfWriter()
        
        cover = PdfReader(self.cover_pdf)
        for page in cover.pages:
            merger.add_page(page)
            
        content = PdfReader(content_pdf)
        for page in content.pages:
            merger.add_page(page)
            
        with open(self.output_pdf, 'wb') as output:
            merger.write(output)


    def get_ordered_markdown_files(self):
        files = [f for f in os.listdir(self.markdown_dir) if f.endswith('.md')]
        files.sort(key=lambda x: int(re.search(r'^\d+', x).group()))
        return files

    def create_title_page(self):
        html_title = f"""
        <html>
            <body style="display: flex; justify-content: center; align-items: center; height: 100vh;">
                <div style="text-align: center;">
                    <h1 style="font-size: 32px; margin-bottom: 20px;">{self.title}</h1>
                    <p style="font-size: 18px;">{self.author}</p>
                </div>
            </body>
        </html>
        """
        return HTML(string=html_title)

    def convert(self):
        # Configuração de fonte
        font_config = FontConfiguration()
        
        # CSS com estilos do Pygments e ajustes de tamanho
        formatter = HtmlFormatter(style='monokai')
        pygments_css = formatter.get_style_defs('.highlight')
        
        css = CSS(string=f'''
            @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600&display=swap');
            
            body {{ 
                font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, 
                        'Segoe UI', Roboto, Oxygen-Sans, Ubuntu, Cantarell, 
                        'Helvetica Neue', sans-serif;
                font-size: 11pt;
                line-height: 1.6;
                padding: 40px;
                color: #2C3E50;
            }}
            
            .highlight {{
                background: #272822 !important;
                border-radius: 5px;
                padding: 15px !important;
                margin: 15px 0 !important;
                overflow-x: auto;
                width: 100%;
                box-sizing: border-box;
            }}
            
            .highlight pre {{
                margin: 0 !important;
                padding: 0 !important;
                font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
                font-size: 11px !important;
                line-height: 1.4 !important;
                color: #f8f8f2 !important;
                white-space: pre-wrap !important;
                word-wrap: break-word !important;
            }}
            
            /* Estilos específicos para o sumário */
            nav table {{
                width: 100%;
                border-collapse: collapse;
            }}
            
            nav table td {{
                vertical-align: bottom;
                padding: 4px 0;
            }}

            /* Estilos base para todas as listas */
            ul, ol {{
                margin: 0.5em 0;
                padding-left: 1.5em !important;
                list-style-position: outside !important;
            }}
            
            /* Estilo específico para items de lista */
            li {{
                display: list-item !important;
                margin: 0.3em 0;
            }}
            
            /* Estilos para listas aninhadas */
            li > ul,
            li > ol {{
                margin-top: 0.2em !important;
                margin-bottom: 0.2em !important;
                margin-left: 1em !important;
            }}
            
            /* Diferentes níveis de marcadores */
            ul {{
                list-style-type: disc !important;
            }}
            
            ul ul {{
                list-style-type: circle !important;
            }}
            
            ul ul ul {{
                list-style-type: square !important;
            }}
            
            /* Força a exibição de marcadores */
            ul li::marker {{
                display: inline-block !important;
                content: "•" !important;
            }}
            
            ul ul li::marker {{
                content: "○" !important;
            }}
            
            ul ul ul li::marker {{
                content: "▪" !important;
            }}

            {pygments_css}
        ''', font_config=font_config)

        # Coleta e processa conteúdo dos arquivos markdown
        markdown_contents = []
        processed_contents = []
        
        # Reset das variáveis de controle do sumário
        self.current_page = 4
        self.toc_entries = []
        
        for md_file in self.get_ordered_markdown_files():
            with open(self.markdown_dir / md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                markdown_contents.append(content)
                processed_contents.append(self.process_markdown_content(content))

        # Gera os documentos PDF separadamente
        pages = []
        pages.append(self.create_title_page().render(stylesheets=[css]))
        pages.append(HTML(string=self.generate_toc()).render(stylesheets=[css]))

        # Conteúdo principal
        content_html = f"""
        <html>
            <body>
                {''.join(processed_contents)}
            </body>
        </html>
        """
        pages.append(HTML(string=content_html).render(stylesheets=[css]))

        # Combina todas as páginas em um PDF temporário
        all_pages = []
        for doc in pages:
            all_pages.extend(doc.pages)

        # Salva o PDF do conteúdo temporariamente
        temp_pdf = "temp_content.pdf"
        pages[0].copy(all_pages).write_pdf(temp_pdf)

        # Mescla o PDF da capa com o conteúdo
        self.merge_pdfs(temp_pdf)

        # Remove o arquivo temporário
        os.remove(temp_pdf)
