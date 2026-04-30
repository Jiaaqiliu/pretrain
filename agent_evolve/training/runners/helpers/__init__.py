"""Stage-support utilities — not stages themselves.

``dataset``    — render training rows into Datums (smoke) or HF Dataset (real SFT).
``pack_adapter`` — package a checkpoint dir into an adapter ``CheckpointRef``
                   (used only by the smoke SFT path).
"""
