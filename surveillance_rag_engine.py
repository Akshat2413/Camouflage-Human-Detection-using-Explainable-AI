"""
Surveillance RAG Engine
Adapted from your existing RAG engine for detection explanations
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests
import json
from typing import Tuple, List, Dict
from datetime import datetime


class SurveillanceRAGEngine:
    """
    RAG Engine adapted for Surveillance Detection System
    Uses ChromaDB + Ollama for intelligent Q&A about detections
    """
    
    def __init__(self, 
                 ollama_url="http://localhost:11434",
                 model_name="llama3.2",
                 embedding_model="all-MiniLM-L6-v2"):
        """Initialize Surveillance RAG Engine"""
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.knowledge_loaded = False
        
        # Initialize embedding model
        print(f"📚 Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Create collection for surveillance knowledge
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name="surveillance_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
        except:
            self.chroma_client.reset()
            self.collection = self.chroma_client.create_collection(
                name="surveillance_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
        
        # Load domain knowledge
        self.load_surveillance_knowledge()
    
    def load_surveillance_knowledge(self):
        """Load surveillance domain knowledge into vector database"""
        
        print("📖 Loading surveillance domain knowledge...")
        
        # Define knowledge chunks
        knowledge_chunks = [
            {
                "id": "kb_001",
                "text": """Detection Confidence Levels Explanation:
Very High Confidence (90%+): The system is highly certain this is a person. Both YOLO and ResNet agree strongly. 
Grad-CAM shows clear activation on human features like head, shoulders, and body outline. Edge patterns are distinct.
Recommendation: Trust this detection.""",
                "category": "confidence_scoring"
            },
            {
                "id": "kb_002",
                "text": """Detection Confidence Levels Explanation:
Medium Confidence (70-89%): The system detects a person but with some uncertainty. This can occur due to partial occlusion,
camouflage effectiveness, or background interference. Grad-CAM may show scattered activation. 
Recommendation: Manual review suggested.""",
                "category": "confidence_scoring"
            },
            {
                "id": "kb_003",
                "text": """Detection Confidence Levels Explanation:
Low Confidence (<70%): Weak detection signals. Significant background noise or heavy camouflage. Grad-CAM shows weak 
or inconsistent activation patterns. High chance of false positive.
Recommendation: Verification required.""",
                "category": "confidence_scoring"
            },
            {
                "id": "kb_004",
                "text": """Detection Pipeline Architecture:
Stage 1 - YOLO Detection: Fast object detection identifies potential persons in the image. Uses YOLOv8 trained on 
camouflaged person dataset. Confidence threshold typically 0.5. May have false positives.
Stage 2 - ResNet Verification: Deep learning verification using ResNet50. Analyzes each YOLO detection to confirm 
if it's actually a person. Filters out false positives. Confidence threshold typically 0.7.""",
                "category": "pipeline"
            },
            {
                "id": "kb_005",
                "text": """Explainable AI (XAI) Methods:
Grad-CAM (Gradient-weighted Class Activation Mapping): Shows which regions of the image the model focused on to make 
its decision. Red/yellow areas indicate high importance. Helps understand WHY the model detected something.
Grad-CAM++: Improved version with better localization of important features. More accurate than standard Grad-CAM.""",
                "category": "xai_methods"
            },
            {
                "id": "kb_006",
                "text": """Why Detection Confidence Varies:
High confidence occurs when: Clear human silhouette, distinct edge patterns, no occlusion, good lighting, 
texture patterns match human clothing, strong Grad-CAM activation on key features.
Low confidence occurs when: Partial occlusion (trees, bushes), effective camouflage blending with background, 
poor lighting conditions, small or distant target, weak Grad-CAM activation, conflicting signals between YOLO and ResNet.""",
                "category": "confidence_factors"
            },
            {
                "id": "kb_007",
                "text": """False Positive Filtering:
YOLO alone has ~15-20% false positive rate. ResNet verification reduces this to ~5%. Common false positives include:
tree stumps, rock formations, shadows, dense vegetation, mannequins. ResNet is trained to distinguish these from 
real humans by analyzing texture, shape, and contextual features.""",
                "category": "filtering"
            },
            {
                "id": "kb_008",
                "text": """Threat Level Assessment:
Critical (Red): Confidence >90%, clear hostile posture or weapon indicators, rapid movement patterns.
High (Orange): Confidence 80-90%, suspicious behavior, proximity to restricted zones.
Medium (Yellow): Confidence 70-80%, normal behavior but verified human presence.
Low (Green): Confidence 50-70%, likely false alarm but flagged for review.""",
                "category": "threat_assessment"
            },
            {
                "id": "kb_009",
                "text": """Grad-CAM Interpretation Guide:
Red/Yellow regions: High importance for detection decision. Model strongly focused here.
Green regions: Moderate importance.
Blue/Purple regions: Low importance or background.
If Grad-CAM highlights head, shoulders, torso: Strong human features detected.
If Grad-CAM scattered or unfocused: Weak or uncertain detection.""",
                "category": "xai_interpretation"
            },
            {
                "id": "kb_010",
                "text": """Common Detection Scenarios:
Scenario 1 - Clear Detection: Person in open area, good lighting, minimal camouflage. YOLO: 85%+, ResNet: 90%+.
Scenario 2 - Partial Occlusion: Person behind tree/bush. YOLO: 65-75%, ResNet: 70-85%. Requires verification.
Scenario 3 - Heavy Camouflage: Military-grade camo in matching environment. YOLO: 50-65%, ResNet: 60-75%. High skill required.
Scenario 4 - False Positive: Tree stump, shadow. YOLO: 50-70%, ResNet: <60%. Filtered out.""",
                "category": "scenarios"
            },
            {
                "id": "kb_011",
                "text": """Model Performance Metrics:
YOLO Detection Rate: ~85% of persons detected. Fast (50ms per image). Some false positives.
ResNet Verification: ~94% accuracy. Slower (30ms per detection). Excellent false positive filtering.
Combined System: ~88% final accuracy. False positive rate: ~5%. Processing time: <1 second per image.
Improvement over YOLO-only: 12% reduction in false positives.""",
                "category": "performance"
            },
            {
                "id": "kb_012",
                "text": """When to Trust Detections:
Trust HIGH when: Both YOLO and ResNet scores >85%, Clear Grad-CAM activation on human features, 
Consistent detection across multiple frames (if video), No conflicting indicators.
Review NEEDED when: Scores 70-85%, Grad-CAM shows scattered activation, Partial occlusion visible, 
Environmental conditions poor (rain, fog, darkness).
Likely FALSE when: ResNet <70%, Grad-CAM unfocused or on background, Object stationary and matches environment,
No clear human features visible.""",
                "category": "decision_making"
            }
        ]
        
        # Add each chunk to vector database
        for chunk in knowledge_chunks:
            embedding = self.embedding_model.encode(chunk['text']).tolist()
            
            self.collection.add(
                embeddings=[embedding],
                documents=[chunk['text']],
                ids=[chunk['id']],
                metadatas=[{"category": chunk['category']}]
            )
        
        self.knowledge_loaded = True
        print(f"✅ Loaded {len(knowledge_chunks)} knowledge chunks into vector database")
    
    def add_detection_to_knowledge(self, detection_data: Dict):
        """Add a detection result to knowledge base for future reference"""
        
        # Create knowledge text from detection
        text = f"""Detection Record:
Person detected with {detection_data.get('resnet_score', 0)*100:.1f}% confidence.
YOLO Score: {detection_data.get('yolo_score', 0)*100:.1f}%
ResNet Score: {detection_data.get('resnet_score', 0)*100:.1f}%
Location: {detection_data.get('bbox', 'unknown')}
Timestamp: {datetime.now().isoformat()}
"""
        
        # Generate embedding and add
        embedding = self.embedding_model.encode(text).tolist()
        detection_id = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.collection.add(
            embeddings=[embedding],
            documents=[text],
            ids=[detection_id],
            metadatas=[{"category": "detection_history", "confidence": detection_data.get('resnet_score', 0)}]
        )
    
    def retrieve_context(self, query: str, top_k=5) -> List[str]:
        """Retrieve relevant knowledge chunks for a query"""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'distances', 'metadatas']
        )
        
        # Debug output
        print(f"\n🔍 Retrieved {len(results['documents'][0])} relevant knowledge chunks")
        for i, (doc, dist, meta) in enumerate(zip(
            results['documents'][0], 
            results['distances'][0],
            results['metadatas'][0]
        )):
            preview = doc[:80].replace('\n', ' ')
            category = meta.get('category', 'unknown')
            print(f"  [{i+1}] Distance: {dist:.3f} | Category: {category} | {preview}...")
        
        return results['documents'][0]
    
    def query_ollama(self, prompt: str) -> str:
        """Send query to Ollama and get response"""
        url = f"{self.ollama_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.ConnectionError:
            return "⚠️ Cannot connect to Ollama. Please ensure: 1) Ollama is installed, 2) Run 'ollama serve' in terminal"
        except requests.exceptions.Timeout:
            return "⚠️ Ollama timeout. Try a smaller model or simpler question."
        except Exception as e:
            return f"⚠️ Ollama error: {str(e)}"
    
    def query(self, question: str, detection_context: Dict = None) -> Tuple[str, List[str]]:
        """
        Answer a question using RAG + current detection context
        
        Args:
            question: User's question
            detection_context: Current detection data
        
        Returns:
            (answer, source_chunks)
        """
        
        # Retrieve relevant knowledge
        knowledge_chunks = self.retrieve_context(question, top_k=5)
        
        # Build context
        context_parts = []
        
        # Add retrieved knowledge
        context_parts.append("KNOWLEDGE BASE:")
        for i, chunk in enumerate(knowledge_chunks, 1):
            context_parts.append(f"\n[Knowledge {i}]\n{chunk}")
        
        # Add current detection context if provided
        if detection_context:
            context_parts.append("\n\nCURRENT DETECTION DATA:")
            if detection_context.get('detections'):
                for idx, det in enumerate(detection_context['detections'], 1):
                    context_parts.append(f"\nPerson {idx}:")
                    context_parts.append(f"  - YOLO Score: {det.get('yolo_score', 0)*100:.1f}%")
                    context_parts.append(f"  - ResNet Score: {det.get('resnet_score', 0)*100:.1f}%")
                    context_parts.append(f"  - Combined Score: {det.get('combined_score', 0)*100:.1f}%")
            
            if detection_context.get('statistics'):
                stats = detection_context['statistics']
                context_parts.append("\nStatistics:")
                context_parts.append(f"  - Total YOLO Detections: {stats.get('yolo_detections', 0)}")
                context_parts.append(f"  - Verified Detections: {stats.get('verified_detections', 0)}")
                context_parts.append(f"  - Filtered Out: {stats.get('filtered_out', 0)}")
        
        context = "\n".join(context_parts)
        
        # Build prompt
        prompt = f"""You are an AI Surveillance Assistant. Answer the user's question directly and professionally.

{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer directly and concisely
- Use the knowledge base and detection data provided
- Be professional and security-focused
- Cite specific confidence scores when relevant
- If asking about a specific person, use their exact scores
- Keep responses clear and actionable

YOUR ANSWER:"""
        
        # Get answer from Ollama
        answer = self.query_ollama(prompt)
        
        return answer, knowledge_chunks
    
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def test_connection(self) -> str:
        """Test if system is working"""
        if not self.is_available():
            return """❌ Ollama is NOT running!

TO FIX:
1. Download Ollama: https://ollama.com/download
2. Install it
3. Open terminal and run: ollama serve
4. In another terminal: ollama pull llama3.2
5. Keep ollama serve running"""
        
        return "✅ Surveillance RAG Engine ready!"


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("Testing Surveillance RAG Engine")
    print("=" * 70)
    
    # Initialize
    rag = SurveillanceRAGEngine(model_name="llama3.2")
    
    # Test connection
    print("\n" + rag.test_connection())
    
    if rag.is_available():
        # Test with sample detection data
        sample_detection = {
            "detections": [
                {
                    "yolo_score": 0.82,
                    "resnet_score": 0.94,
                    "combined_score": 0.88
                }
            ],
            "statistics": {
                "yolo_detections": 1,
                "verified_detections": 1,
                "filtered_out": 0
            }
        }
        
        # Test questions
        questions = [
            "Why is this detection high confidence?",
            "What does Grad-CAM show?",
            "Should I trust this detection?"
        ]
        
        for q in questions:
            print(f"\n{'='*70}")
            print(f"Q: {q}")
            answer, sources = rag.query(q, sample_detection)
            print(f"\nA: {answer}")
            print(f"\nSources used: {len(sources)} knowledge chunks")