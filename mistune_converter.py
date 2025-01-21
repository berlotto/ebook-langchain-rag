import os
from pathlib import Path
import mistune
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
from bs4 import BeautifulSoup
import re
from PyPDF2 import PdfWriter, PdfReader, PdfMerger
import pygments
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import base64
import requests

class MistuneMarkdownConverter:
    def __init__(self, markdown_dir, output_pdf, cover_pdf, title, author):
        self.markdown_dir = Path(markdown_dir)
        self.output_pdf = output_pdf
        self.cover_pdf = cover_pdf
        self.title = title
        self.author = author
        self.current_page = 4
        self.toc_entries = []
        self.plugins_list = [
            'footnotes',
            'strikethrough',
            'table',
            'def_list',
            'task_lists',
            'url',
            'def_list',
            'abbr',
            'mark',
            'insert',
            'superscript',
            'subscript',
            'math',
        ]
        
        # Custom renderer for code highlighting and Mermaid diagrams
        class HighlightRenderer(mistune.HTMLRenderer):
            def block_code(self, code, info=None):
                if not info:
                    return f'\n<pre><code>{mistune.escape(code)}</code></pre>\n'
                
                if info.lower() == 'mermaid':
                    try:
                        url = "https://mermaid.ink/img/"
                        graphbytes = code.encode("utf-8")
                        base64_string = base64.b64encode(graphbytes).decode("ascii")
                        img_url = f"{url}{base64_string}?type=png"
                        return f'<img src="{img_url}" alt="Mermaid Diagram" style="display: block; margin: 20px auto; max-width: 100%; height: auto;">'
                    except Exception as e:
                        print(f"Error rendering Mermaid diagram: {e}")
                        return f'<pre><code>{mistune.escape(code)}</code></pre>'
                
                try:
                    lexer = get_lexer_by_name(info)
                    formatter = HtmlFormatter(style='monokai', cssclass='highlight')
                    return pygments.highlight(code, lexer, formatter)
                except:
                    return f'<pre><code>{mistune.escape(code)}</code></pre>'
        
        # Create Markdown parser with custom renderer
        self.markdown = mistune.create_markdown(
            renderer=HighlightRenderer(),
            plugins=self.plugins_list
        )

    def create_chapter_cover(self, title):
        """Cria uma página de capa para o capítulo com o título perfeitamente centralizado."""
        html_chapter = f"""
        <html>
            <body style="margin: 0; padding: 0; height: 100vh;">
                <div class="chapter-cover">
                    <div class="chapter-title">
                        <h1>{title}</h1>
                    </div>
                </div>
            </body>
        </html>
        """
        return HTML(string=html_chapter)

    def extract_chapter_title(self, content):
        """Extrai o título do capítulo do conteúdo markdown."""
        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith('# '):
                return line.replace('# ', '').strip()
        return 'Capítulo'

    def process_markdown_content(self, content):
        # Process the content with Mistune
        html = self.markdown(content)
        
        # Envolve o conteúdo em tags html e body se não existirem
        if not html.startswith('<html>'):
            html = f"""
            <html>
            <body>
                {html}
            </body>
            </html>
            """
        
        # Parse with BeautifulSoup to handle nested lists
        soup = BeautifulSoup(html, 'html.parser')
        
        # Garante que existe um body
        if not soup.body:
            body = soup.new_tag('body')
            for child in list(soup.children):
                body.append(child)
            soup.append(body)
        
        # Find all headers to build TOC
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            level = int(heading.name[1])
            heading_id = f"heading_{self.current_page}_{len(self.toc_entries)}"
            
            # Adiciona um ID ao cabeçalho
            heading['id'] = heading_id
            
            self.toc_entries.append({
                'level': level,
                'text': heading.get_text(),
                'page': self.current_page,
                'id': heading_id
            })
        
        # Envolve o conteúdo em uma div com ID da página
        page_div = soup.new_tag('div', id=f'page_{self.current_page}')
        for child in list(soup.body.children):
            page_div.append(child)
        soup.body.clear()
        soup.body.append(page_div)
        
        return str(soup)

    def generate_toc(self):
        """Generate table of contents with page numbers."""
        toc = [
            "<html><body>",
            "<style>",
            "@page { size: A4; margin: 2cm }",
            "a { color: #000; text-decoration: none; }",
            "</style>",
            "<div style='height: 100vh; padding: 00px;'>",
            "<h1 style='margin-bottom: 30px;'>Sumário</h1>",
            "<nav style='width: 100%;'>",
            "<table style='width: 100%; border: none;'>"
        ]
        
        for entry in self.toc_entries:
            if entry['level'] > 2:
                continue
            indent = (entry['level'] - 1) * 20
            dots = '.' * 50
            
            toc.append(
                f"""
                <tr>
                    <td style="border: none; padding: 5px 0;">
                        <div style="margin-left: {indent}px;">
                            <a href="#{entry['id']}" style="text-decoration: none; color: #000;">
                                {entry['text']}
                            </a>
                        </div>
                    </td>
                    <td style="border: none; text-align: right; padding: 5px 0; white-space: nowrap;">
                        <span style="color: #000; padding-left: 10px;">{dots}</span>
                        <a href="#{entry['id']}" style="text-decoration: none; color: #000;">
                            {entry['page']}
                        </a>
                    </td>
                </tr>
                """
            )
        
        toc.extend(["</table>", "</nav>", "</div>", "</body></html>"])
        return "\n".join(toc)


    def merge_pdfs(self, content_pdf):
        merger = PdfMerger()
        
        cover = PdfReader(self.cover_pdf)
        merger.append(cover, import_outline=True)
            
        content = PdfReader(content_pdf)
        merger.append(content, import_outline=True)

        merger.write(self.output_pdf)
        merger.close()

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
                margin: 0;
            }}
            
            /* Estilos para página de capa do capítulo */
            .chapter-cover {{
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                page-break-after: always;
                page-break-before: always;
                padding: 0;
                margin: 0;
            }}
            
            /* Garante que o conteúdo do capítulo comece em nova página */
            .chapter-content {{
                page-break-before: always;
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
            
            nav table {{
                width: 100%;
                border-collapse: collapse;
            }}
            
            nav table td {{
                vertical-align: bottom;
                padding: 4px 0;
            }}

            ul, ol {{
                margin: 0.5em 0;
                padding-left: 1.5em !important;
                list-style-position: outside !important;
            }}
            
            li {{
                display: list-item !important;
                margin: 1em 0;
            }}
            
            li > ul,
            li > ol {{
                margin-top: 0.2em !important;
                margin-bottom: 0.2em !important;
                margin-left: 1em !important;
            }}
            
            ul {{
                list-style-type: disc !important;
            }}
            
            ul ul {{
                list-style-type: circle !important;
            }}
            
            ul ul ul {{
                list-style-type: square !important;
            }}
            
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

        # Gera os documentos PDF separadamente
        pages = []
        pages.append(self.create_title_page().render(stylesheets=[css]))
        
        # Reset das variáveis de controle do sumário
        self.current_page = 4
        self.toc_entries = []
        
        # Processa cada arquivo markdown separadamente
        for md_file in self.get_ordered_markdown_files():
            with open(self.markdown_dir / md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Extrai o título do capítulo e cria a página de capa
                chapter_title = self.extract_chapter_title(content)
                pages.append(self.create_chapter_cover(chapter_title).render(stylesheets=[css]))
                
                # Incrementa a página para o conteúdo
                self.current_page += 1
                
                # Processa o conteúdo do capítulo
                processed_content = self.process_markdown_content(content)
                chapter_html = f"""
                <html>
                    <body class="chapter-content">
                        {processed_content}
                    </body>
                </html>
                """
                pages.append(HTML(string=chapter_html).render(stylesheets=[css]))
                
                # Incrementa a página para o próximo capítulo
                self.current_page += 1

        # Insere o sumário após processar todo o conteúdo
        toc_page = HTML(string=self.generate_toc()).render(stylesheets=[css])
        pages.insert(1, toc_page)

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
        