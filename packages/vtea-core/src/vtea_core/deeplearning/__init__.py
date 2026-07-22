"""Deep learning integrations: in-process Cellpose, native PyTorch models, bioimageio.core.

Replaces vtea.deeplearning's two parallel Java integrations (the Py4J subprocess
bridge to python/cellpose_server.py, and the JavaCPP-PyTorch 3D VAE/CNN stack)
with direct, in-process Python library usage.
"""
