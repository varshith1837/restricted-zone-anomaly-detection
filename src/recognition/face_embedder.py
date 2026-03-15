"""
Sentinel Vision — FaceNet Embedding Extractor
===============================================
Extracts 512-dimensional face embeddings using InceptionResnetV1 (FaceNet).

Key ML concepts demonstrated:
- Transfer learning with pre-trained deep embeddings
- Metric learning / embedding space
- Feature extraction from CNNs
- Batch inference for efficiency
"""

import cv2
import numpy as np
import yaml
import torch
from typing import List, Optional
from PIL import Image


class FaceEmbedder:
    """
    Extracts face embeddings using FaceNet (InceptionResnetV1).
    
    The embeddings live in a 512-dimensional space where:
    - Same person's faces are close together (small L2 distance)
    - Different people's faces are far apart
    
    These embeddings are then fed to a trained SVM/KNN classifier
    instead of raw distance thresholding (which is what the old code did).
    
    Args:
        config_path: Path to config.yaml
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        rec_config = config['recognition']
        
        self.embedding_dim = rec_config['embedding_dim']
        pretrained_model = rec_config['embedding_model']  # 'vggface2' or 'casia-webface'
        
        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError:
            raise ImportError(
                "facenet-pytorch not installed. Run: pip install facenet-pytorch"
            )
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained FaceNet model
        self.model = InceptionResnetV1(
            pretrained=pretrained_model,
            classify=False  # We want embeddings, not classification logits
        ).eval().to(self.device)
        
        print(f"[FaceEmbedder] InceptionResnetV1 ({pretrained_model}) loaded on {self.device}")
        print(f"[FaceEmbedder] Embedding dimension: {self.embedding_dim}")
    
    def preprocess_face(self, face_img: np.ndarray, target_size: int = 160) -> torch.Tensor:
        """
        Preprocess a face image for FaceNet.
        
        Steps:
        1. Resize to 160x160 (FaceNet input size)
        2. Convert BGR → RGB
        3. Normalize to [-1, 1]
        4. Convert to tensor with batch dimension
        
        Args:
            face_img: Cropped face image (BGR, numpy array)
            target_size: Target size (default 160 for FaceNet)
            
        Returns:
            Preprocessed tensor (1, 3, 160, 160)
        """
        if face_img is None or face_img.size == 0:
            return None
        
        # Resize
        face_resized = cv2.resize(face_img, (target_size, target_size))
        
        # BGR → RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1] (standard for FaceNet)
        face_normalized = (face_rgb.astype(np.float32) - 127.5) / 128.0
        
        # HWC → CHW and add batch dimension
        face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return face_tensor.to(self.device)
    
    def extract_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a single face embedding.
        
        Args:
            face_img: Cropped face image (BGR, numpy)
            
        Returns:
            512-d embedding as numpy array, or None if failed
        """
        tensor = self.preprocess_face(face_img)
        if tensor is None:
            return None
        
        with torch.no_grad():
            embedding = self.model(tensor)
        
        # L2 normalize the embedding
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().numpy().flatten()
    
    def extract_batch_embeddings(self, face_images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Extract embeddings for a batch of face images.
        
        More efficient than calling extract_embedding() in a loop
        because it batches the GPU inference.
        
        Args:
            face_images: List of cropped face images (BGR, numpy)
            
        Returns:
            List of 512-d embeddings (or None for invalid faces)
        """
        if not face_images:
            return []
        
        # Preprocess all valid faces
        tensors = []
        valid_indices = []
        
        for i, face_img in enumerate(face_images):
            tensor = self.preprocess_face(face_img)
            if tensor is not None:
                tensors.append(tensor)
                valid_indices.append(i)
        
        if not tensors:
            return [None] * len(face_images)
        
        # Batch inference
        batch = torch.cat(tensors, dim=0)
        
        with torch.no_grad():
            embeddings = self.model(batch)
        
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        embeddings_np = embeddings.cpu().numpy()
        
        # Map back to original indices
        results = [None] * len(face_images)
        for idx, valid_idx in enumerate(valid_indices):
            results[valid_idx] = embeddings_np[idx]
        
        return results
    
    def compute_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute L2 (Euclidean) distance between two embeddings.
        
        Smaller distance = more similar faces.
        Typical thresholds:
        - < 0.6: Same person (high confidence)
        - 0.6 - 1.0: Possibly same person
        - > 1.0: Different people
        
        Args:
            emb1: First embedding (512-d)
            emb2: Second embedding (512-d)
            
        Returns:
            L2 distance (float)
        """
        return float(np.linalg.norm(emb1 - emb2))
    
    def compute_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Range: [-1, 1], where 1 = identical.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Cosine similarity
        """
        dot = np.dot(emb1, emb2)
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        return float(dot / max(norm, 1e-8))


# ==========================================
# Standalone test
# ==========================================
if __name__ == "__main__":
    import os
    
    print("=" * 50)
    print("FaceEmbedder — Standalone Test")
    print("=" * 50)
    
    embedder = FaceEmbedder()
    
    # Test with webcam - capture a face and show embedding stats
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        exit()
    
    print("Press 's' to capture embedding, 'q' to quit.")
    
    captured_embeddings = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.putText(frame, f"Embeddings captured: {len(captured_embeddings)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 's' to capture, 'q' to quit", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        cv2.imshow("FaceEmbedder Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Use entire frame as a test (in real use, face would be cropped)
            emb = embedder.extract_embedding(frame)
            if emb is not None:
                captured_embeddings.append(emb)
                print(f"  Embedding {len(captured_embeddings)}: shape={emb.shape}, "
                      f"norm={np.linalg.norm(emb):.4f}")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Show pairwise distances
    if len(captured_embeddings) >= 2:
        print(f"\n--- Pairwise Distances ({len(captured_embeddings)} embeddings) ---")
        for i in range(len(captured_embeddings)):
            for j in range(i + 1, len(captured_embeddings)):
                dist = embedder.compute_distance(captured_embeddings[i], captured_embeddings[j])
                sim = embedder.compute_cosine_similarity(captured_embeddings[i], captured_embeddings[j])
                print(f"  [{i}] vs [{j}]: L2={dist:.4f}, Cosine={sim:.4f}")
