# MITACS lab notebook — unified Dash browser over the lab's existing
# markdown / YAML / LaTeX content surfaces (notes/lab session entries,
# notes/preregistrations registry artifacts, notes/seeds + notes/literature,
# ~/.claude/.../memory files, writeup/tex PDFs). The Dash entry point is
# post_processing.lab_notebook.app; importing this package must not pull
# in Dash or start a server, so nothing is imported here. The pure
# data-access layer (loader) is importable directly for headless tests.
