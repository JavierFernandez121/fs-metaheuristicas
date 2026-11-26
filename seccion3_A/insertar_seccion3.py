"""Inserta la parte A de la seccion 3 y sus referencias en report/memoria.md.
Uso (desde la raiz del repo):  python seccion3_A/insertar_seccion3.py
- Texto de la seccion 3: sustituye el encabezado hueco '3.x ...' que dejo Javier; si no existe, lo anade
  justo debajo del encabezado '## 3.'
- Referencias: las anade debajo del encabezado 'Referencias de Persona A'; si no existe, debajo de '## 12.'
- No borra nada mas. Es seguro repetirlo (no duplica). Copia de seguridad: report/memoria.md.bak"""
import re
import shutil
import sys
from pathlib import Path

aqui = Path(__file__).resolve().parent
memoria = Path("report/memoria.md")
if not memoria.exists():
    sys.exit("No encuentro report/memoria.md: ejecuta este script desde la raiz del repo (repo-javier).")

texto = memoria.read_text(encoding="utf-8")
sec3 = (aqui / "seccion3_A.md").read_text(encoding="utf-8").rstrip("\n").split("\n")
refs_total = (aqui / "referencias_A.md").read_text(encoding="utf-8").rstrip("\n").split("\n")
refs_cuerpo = [l for l in refs_total if not l.startswith("### Referencias (parte A)")]
lineas = texto.split("\n")


def buscar(patron):
    for i, l in enumerate(lineas):
        if re.match(patron, l):
            return i
    return None


cambios = False
if "### 3.A Estado del arte" in texto:
    print("La seccion 3.A ya estaba: no la duplico")
else:
    hueco = buscar(r"^#{2,4}\s*3\.x")
    if hueco is not None:
        lineas[hueco:hueco + 1] = sec3          # sustituye el encabezado hueco
        print(f"OK: seccion 3.A insertada en el hueco de la linea {hueco + 1}")
    else:
        i3 = buscar(r"^## 3\.")
        if i3 is None:
            sys.exit("No encuentro ni '3.x' ni '## 3.' en memoria.md")
        lineas[i3 + 1:i3 + 1] = sec3
        print("OK: seccion 3.A insertada debajo del encabezado '## 3.'")
    cambios = True

texto2 = "\n".join(lineas)
if "Zhang, H. y Sun, G. (2002)" in texto2:
    print("Las referencias de A ya estaban: no las duplico")
else:
    hueco = buscar(r"^#{2,4}\s*Referencias de Persona A")
    if hueco is not None:
        lineas[hueco + 1:hueco + 1] = refs_cuerpo
        print("OK: referencias insertadas debajo de 'Referencias de Persona A'")
    else:
        i12 = buscar(r"^## 12\.")
        if i12 is None:
            sys.exit("No encuentro ni 'Referencias de Persona A' ni '## 12.' en memoria.md")
        lineas[i12 + 1:i12 + 1] = refs_total
        print("OK: referencias insertadas debajo de '## 12.'")
    cambios = True

if cambios:
    shutil.copy(memoria, "report/memoria.md.bak")
    memoria.write_text("\n".join(lineas), encoding="utf-8")
    print("Guardado. Copia de seguridad: report/memoria.md.bak (borrala antes del commit)")
    print("Revisa las referencias duplicadas con las de Javier (p. ej. Kohavi y John, Xue et al.).")
