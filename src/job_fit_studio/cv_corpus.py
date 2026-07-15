"""
Your CV as a set of discrete, citable CAPABILITY chunks -- deliberately
different from the arbitrary word-count chunking used in the papers RAG
project. There, a paper's content is continuous prose with no natural
break points, so fixed-size chunks with overlap make sense. Here, your CV
is already naturally composed of discrete units (one project, one skill
group, one job) -- splitting a project's description in half would just
break something that was already a complete, meaningful unit. So each
chunk here IS one project/skill/experience entry, tagged with a type and
(critically) an `evidence` field, so that when a chunk matches a job
requirement, we can say exactly WHY: not just "this seems related" but
"here's the specific project and the specific test/metric that backs it up."
"""
from dataclasses import dataclass, field


@dataclass
class CapabilityChunk:
    chunk_id: str
    kind: str          # "project" | "skill" | "experience" | "education"
    title: str          # short name, e.g. "Multi-Camera Traffic Congestion Detection"
    text: str            # the full description used for embedding/matching
    evidence: str = ""    # concrete proof: test count, metric, repo link
    tags: list = field(default_factory=list)  # keywords for cheap pre-filtering


# This is YOUR real CV, structured -- built from the resume content we
# already wrote together. Keep this file updated as you finish more
# projects (RAG Phase 2/3, job-fit-studio itself once it exists, etc.) --
# the matcher is only as honest as this corpus is current.
CV_CORPUS = [
    CapabilityChunk(
        chunk_id="proj-diabetes",
        kind="project",
        title="Diabetes Risk Prediction — End-to-End MLOps Pipeline",
        text=("Built and deployed a diabetes risk prediction system end-to-end: trained a "
              "scikit-learn Random Forest classifier, served it via a FastAPI REST API, "
              "containerized with Docker, and validated with an automated test suite integrated "
              "into GitHub Actions CI/CD. Designed a browser-based prediction interface with "
              "real-time risk visualization."),
        evidence="10 passing automated tests; live FastAPI REST endpoint; Dockerized; GitHub Actions CI",
        tags=["fastapi", "docker", "ci/cd", "scikit-learn", "rest api", "mlops", "deployment", "web frontend"],
    ),
    CapabilityChunk(
        chunk_id="proj-traffic",
        kind="project",
        title="Multi-Camera Traffic Congestion Detection System",
        text=("Engineered a real-time, multi-camera vehicle congestion detection pipeline "
              "combining OpenCV optical flow, classical background subtraction, and YOLOv8 "
              "object detection, using a custom density-aware priority score to allocate "
              "compute-intensive detection to only the most congested camera feed. Diagnosed "
              "and fixed a background-subtraction defect and calibrated per-camera detection "
              "thresholds against real traffic footage."),
        evidence="9 passing automated tests; real footage calibration; documented bug fix with root-cause analysis",
        tags=["opencv", "yolo", "yolov8", "computer vision", "object detection", "optical flow",
              "real-time systems", "video processing", "debugging"],
    ),
    CapabilityChunk(
        chunk_id="proj-rnn-mlflow",
        kind="project",
        title="Comparative Language Modeling: RNN vs LSTM vs GRU (PyTorch + MLflow)",
        text=("Implemented and benchmarked RNN, LSTM, and GRU architectures in PyTorch for "
              "character-level text generation; quantitatively compared performance via "
              "validation perplexity. Tracked all experiments with MLflow, registered the "
              "best-performing model, and deployed it as a live REST API for text generation."),
        evidence="31 passing automated tests; MLflow experiment tracking + model registry; perplexity RNN 5.60 vs LSTM 4.97; live served model",
        tags=["pytorch", "mlflow", "rnn", "lstm", "gru", "nlp", "deep learning", "experiment tracking",
              "model deployment", "text generation"],
    ),
    CapabilityChunk(
        chunk_id="proj-rag",
        kind="project",
        title="Retrieval-Augmented Generation (RAG) System for ML Research Papers",
        text=("Designed a RAG pipeline (document chunking, embedding, ChromaDB vector storage, "
              "similarity search) with a swappable embedding-model interface, enabling direct "
              "comparison between keyword-based (TF-IDF) and semantic retrieval. Built a "
              "quantitative evaluation harness (recall@k) across natural and paraphrased query sets."),
        evidence="23 passing automated tests; measured recall@k across two eval tiers; documented TF-IDF vs semantic embedding gap",
        tags=["rag", "vector database", "chromadb", "embeddings", "llm", "retrieval",
              "evaluation", "nlp", "semantic search"],
    ),
    CapabilityChunk(
        chunk_id="proj-carclassifier",
        kind="project",
        title="16-Class Car Brand Classifier",
        text=("Trained a 16-class car brand image classifier on a custom dataset of "
              "approximately 12,000 images using PyTorch, the timm library, and the "
              "RexNet-150 architecture."),
        evidence="Custom 12,000-image dataset; PyTorch + timm + RexNet-150",
        tags=["pytorch", "computer vision", "image classification", "timm", "cnn"],
    ),
    CapabilityChunk(
        chunk_id="skill-core-ml",
        kind="skill",
        title="Core ML/DL Stack",
        text=("Machine learning and deep learning fundamentals: Python, PyTorch, TensorFlow, "
              "Keras, scikit-learn, Pandas, NumPy. Applied across classical ML (Random Forest), "
              "computer vision (CNNs, YOLO), and NLP (RNN/LSTM/GRU) projects."),
        evidence="Demonstrated across 5 shipped, tested projects",
        tags=["python", "pytorch", "tensorflow", "keras", "scikit-learn", "pandas", "numpy",
              "machine learning", "deep learning"],
    ),
    CapabilityChunk(
        chunk_id="skill-mlops-deployment",
        kind="skill",
        title="MLOps & Deployment",
        text=("Building and deploying real, served ML systems rather than notebooks only: "
              "FastAPI REST APIs, Docker containerization, GitHub Actions CI/CD, MLflow "
              "experiment tracking and model registry."),
        evidence="diabetes-risk-predictor (Docker + FastAPI + CI/CD); rnn-mlflow (MLflow registry + serving)",
        tags=["fastapi", "docker", "mlops", "ci/cd", "mlflow", "deployment", "rest api"],
    ),
    CapabilityChunk(
        chunk_id="skill-cv",
        kind="skill",
        title="Computer Vision",
        text=("Applied computer vision: OpenCV (optical flow, background subtraction), YOLOv8 "
              "object detection, CNN-based image classification, real-time video pipeline design."),
        evidence="traffic-switch (OpenCV + YOLOv8); car brand classifier (CNN)",
        tags=["computer vision", "opencv", "yolo", "cnn", "object detection", "image classification"],
    ),
    CapabilityChunk(
        chunk_id="skill-rag-vectordb",
        kind="skill",
        title="RAG & Vector Databases",
        text=("Retrieval-augmented generation system design: document chunking strategy, "
              "embedding model selection (TF-IDF vs semantic), ChromaDB vector storage, "
              "similarity search, and quantitative retrieval evaluation (recall@k)."),
        evidence="rag-papers project, 23 tests, measured recall@k gap between TF-IDF and semantic embeddings",
        tags=["rag", "vector database", "chromadb", "embeddings", "llm", "retrieval augmented generation"],
    ),
    CapabilityChunk(
        chunk_id="exp-graduate-researcher",
        kind="experience",
        title="Graduate Researcher, SMN Lab, SeoulTech",
        text=("Conducting research on 3D Segmentation and point cloud processing using deep "
              "learning techniques. Developing and optimizing CNN architectures using PyTorch "
              "and TensorFlow. Analyzing large-scale datasets to improve model accuracy and "
              "inference speed."),
        evidence="Sep 2024 - Present, SeoulTech Smart Media & Network Lab",
        tags=["research", "3d segmentation", "point cloud", "cnn", "pytorch", "tensorflow"],
    ),
    CapabilityChunk(
        chunk_id="edu-mlflow-mlops-tooling",
        kind="skill",
        title="Experiment Tracking & Reproducibility Discipline",
        text=("Consistent practice of automated testing (pytest), version control (Git/GitHub), "
              "and honest documentation of limitations and known issues across every project, "
              "rather than only reporting favorable results."),
        evidence="104 total automated tests across 4 shipped projects; every README documents known limitations",
        tags=["testing", "pytest", "git", "github", "reproducibility", "documentation"],
    ),
]
