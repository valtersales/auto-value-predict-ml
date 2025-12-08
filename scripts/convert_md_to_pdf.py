#!/usr/bin/env python3
"""
Script para converter arquivo Markdown para PDF.
Requer: markdown2, weasyprint ou pdfkit
"""

import sys
import os
import argparse
from pathlib import Path

def convert_with_weasyprint(md_file, pdf_file):
    """Converte usando weasyprint (requer instalação: pip install weasyprint markdown)"""
    try:
        import markdown
        from weasyprint import HTML
        
        # Ler arquivo markdown
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Converter markdown para HTML
        html_content = markdown.markdown(md_content, extensions=['extra', 'codehilite'])
        
        # Adicionar estilos CSS básicos
        html_with_style = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: 'Helvetica', 'Arial', sans-serif;
                    line-height: 1.6;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3, h4, h5, h6 {{
                    color: #333;
                    margin-top: 1.5em;
                    margin-bottom: 0.5em;
                }}
                h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
                h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 1em 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                code {{
                    background-color: #f4f4f4;
                    padding: 2px 4px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                }}
                pre {{
                    background-color: #f4f4f4;
                    padding: 10px;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                blockquote {{
                    border-left: 4px solid #ddd;
                    margin: 0;
                    padding-left: 20px;
                    color: #666;
                }}
            </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """
        
        # Converter HTML para PDF
        HTML(string=html_with_style).write_pdf(pdf_file)
        print(f"✅ PDF criado com sucesso: {pdf_file}")
        return True
        
    except ImportError:
        print("❌ Bibliotecas necessárias não encontradas.")
        print("   Instale com: pip install weasyprint markdown")
        return False
    except Exception as e:
        print(f"❌ Erro ao converter: {e}")
        return False

def convert_with_pdfkit(md_file, pdf_file):
    """Converte usando pdfkit (requer wkhtmltopdf instalado)"""
    try:
        import markdown
        import pdfkit
        
        # Ler arquivo markdown
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Converter markdown para HTML
        html_content = markdown.markdown(md_content, extensions=['extra', 'codehilite'])
        
        # Converter HTML para PDF
        pdfkit.from_string(html_content, pdf_file)
        print(f"✅ PDF criado com sucesso: {pdf_file}")
        return True
        
    except ImportError:
        return False
    except Exception as e:
        print(f"❌ Erro ao converter: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Converte arquivo Markdown para PDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  %(prog)s docs/relatorio.md
  %(prog)s docs/relatorio.md -o docs/relatorio.pdf
  %(prog)s docs/relatorio.md --output relatorio_final.pdf
        """
    )
    parser.add_argument(
        'md_file',
        type=str,
        help='Caminho do arquivo Markdown a ser convertido'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Caminho do arquivo PDF de saída (padrão: mesmo nome do MD com extensão .pdf)'
    )
    
    args = parser.parse_args()
    
    # Resolver caminho do arquivo markdown
    md_file = Path(args.md_file)
    if not md_file.is_absolute():
        # Se for caminho relativo, resolver a partir do diretório atual
        md_file = Path.cwd() / md_file
    
    if not md_file.exists():
        print(f"❌ Arquivo não encontrado: {md_file}")
        sys.exit(1)
    
    if not md_file.suffix.lower() in ['.md', '.markdown']:
        print(f"⚠️  Aviso: O arquivo não tem extensão .md ou .markdown")
    
    # Determinar caminho do PDF
    if args.output:
        pdf_file = Path(args.output)
        if not pdf_file.is_absolute():
            pdf_file = Path.cwd() / pdf_file
    else:
        # Gerar nome do PDF baseado no MD
        pdf_file = md_file.with_suffix('.pdf')
    
    # Criar diretório de saída se não existir
    pdf_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📄 Convertendo: {md_file}")
    print(f"📄 Para: {pdf_file}")
    print()
    
    # Tentar diferentes métodos
    if convert_with_weasyprint(md_file, pdf_file):
        sys.exit(0)
    
    if convert_with_pdfkit(md_file, pdf_file):
        sys.exit(0)
    
    # Se nenhum método funcionou
    print("\n" + "="*60)
    print("❌ Nenhum método de conversão disponível.")
    print("\nOpções de instalação:")
    print("\n1. Instalar weasyprint (recomendado):")
    print("   pip install weasyprint markdown")
    print("\n2. Instalar pandoc via Homebrew:")
    print("   brew install pandoc")
    print("   pandoc docs/EDA_PHASE_1_1_1_2_REPORT.md -o docs/EDA_PHASE_1_1_1_2_REPORT.pdf")
    print("\n3. Usar ferramenta online:")
    print("   - Dillinger.io")
    print("   - Markdown to PDF")
    print("="*60)
    sys.exit(1)

if __name__ == "__main__":
    main()

