import json
from pathlib import Path
root = Path(__file__).resolve().parent
code = (root / 'main.py').read_text(encoding='utf-8')
notebook = {
    'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['# Heart Health Information Assistant\n', 'UCS420 · Even roll number 1024160112']},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': ['%pip -q install pandas scikit-learn nltk ipywidgets']},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': ['from google.colab import files\n', 'uploaded = files.upload()\n', 'import pathlib\n', 'for name, data in uploaded.items(): pathlib.Path(name).write_bytes(data)']},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': ['%run main.py']},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': ['interactive_colab()']}
    ],
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
    'nbformat': 4,
    'nbformat_minor': 5
}
(root / 'Heart_Health_Colab.ipynb').write_text(json.dumps(notebook, indent=2), encoding='utf-8')
