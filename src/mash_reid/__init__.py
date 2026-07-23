"""mash_reid — verify whether a vehicle seen at point A is the same one seen at
point B, using visual re-identification (appearance embeddings) only.

Public building blocks:
    frame_loader.load_folder    -> list[Frame]
    detector.VehicleDetector    -> detect vehicles in a frame
    embedder.ResNet50Embedder   -> crop -> L2-normalized feature vector
    matcher.match               -> rank A x B pairs with temporal gating
    pipeline.Pipeline           -> orchestrate folder -> detections + embeddings
"""

__version__ = "0.1.0"
