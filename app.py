import os
import json
import faiss
import numpy as np
import random
import spacy
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from phi.agent import Agent
from phi.model.groq import Groq
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# TTS imports
from gtts import gTTS
from playsound import playsound
import tempfile

# ========== Enhanced Configuration ==========
@dataclass
class InterviewConfig:
    max_questions: int = 15
    max_tokens_per_chunk: int = 800
    similarity_threshold: float = 0.7
    min_response_words: int = 8
    max_follow_ups_per_topic: int = 3
    embedding_model: str = 'all-MiniLM-L6-v2'
    llm_model: str = "llama-3.3-70b-versatile"
    difficulty_adaptation: bool = True
    real_time_feedback: bool = True
    save_visualizations: bool = True
    adaptive_questioning: bool = True
    topic_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.topic_weights is None:
            self.topic_weights = {
                "intro": 0.1,
                "projects": 0.3,
                "skills": 0.25,
                "behavioral": 0.2,
                "hr": 0.15
            }

@dataclass
class InterviewState:
    question_count: int = 0
    follow_ups: Dict[str, int] = None
    strength_scores: Dict[str, int] = None
    topic_coverage: Dict[str, bool] = None
    candidate_responses: List[str] = None
    response_quality_scores: List[float] = None
    difficulty_level: str = "medium"
    engagement_score: float = 5.0
    technical_depth: Dict[str, int] = None
    question_types: List[str] = None
    response_times: List[float] = None

    def __post_init__(self):
        if self.follow_ups is None:
            self.follow_ups = {"intro": 0, "projects": 0, "skills": 0, "internships": 0, "hr": 0, "behavioral": 0}
        if self.strength_scores is None:
            self.strength_scores = {"ml": 0, "dsa": 0, "fullstack": 0, "backend": 0, "frontend": 0, "devops": 0, "database": 0, "cloud": 0}
        if self.topic_coverage is None:
            self.topic_coverage = {"projects": False, "skills": False, "internships": False, "behavioral": False}
        if self.candidate_responses is None:
            self.candidate_responses = []
        if self.response_quality_scores is None:
            self.response_quality_scores = []
        if self.technical_depth is None:
            self.technical_depth = {"beginner": 0, "intermediate": 0, "advanced": 0}
        if self.question_types is None:
            self.question_types = []
        if self.response_times is None:
            self.response_times = []

# ========== Enhanced Logging for Colab ==========
def setup_logging():
    """Logging setup for local (VS Code) environment with Windows-safe unicode handling."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter (strip emojis for console on Windows)
    class SafeFormatter(logging.Formatter):
        def format(self, record):
            try:
                # Remove emojis for Windows console
                record.msg = str(record.msg).encode('ascii', 'ignore').decode('ascii')
            except Exception:
                pass
            return super().format(record)
    formatter = SafeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler for local logs directory (keep full unicode)
    log_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'interview.log'), encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logging()

# ========== Enhanced API Key Management for Colab ==========
class APIKeyManager:
    def __init__(self):
        # Only use environment variables or .env file for local use
        self.llama_api_key = self._get_api_key("LLAMA_API_KEY3")
        self.groq_api_key = self._get_api_key("GROQ_API_KEY2")

    def _get_api_key(self, env_var: str) -> str:
        """Get API key from environment or user input"""
        load_dotenv()
        key = os.getenv(env_var)
        if not key:
            print(f"🔑 Please enter your {env_var}:")
            key = input().strip()
            os.environ[env_var] = key
        return key

# ========== Advanced Resume Processing ==========
class EnhancedResumeProcessor:
    def __init__(self, api_key: str, config: InterviewConfig):
        self.api_key = api_key
        self.config = config
        self.nlp = self._load_nlp_model()
        self.embedder = SentenceTransformer(config.embedding_model)
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        self.chunk_metadata = []
        self.resume_analysis = {}

    def _load_nlp_model(self):
        """Load spaCy model with fallback for Colab"""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("📦 Installing spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            return spacy.load("en_core_web_sm")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Enhanced PDF extraction with error handling"""
        try:
            parser = LlamaParse(api_key=self.api_key, result_type="text", verbose=True)
            documents = SimpleDirectoryReader(
                input_files=[pdf_path],
                file_extractor={".pdf": parser}
            ).load_data()

            text = "\n".join([doc.text for doc in documents])
            logger.info(f"✅ Extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            logger.error(f"❌ PDF extraction failed: {e}")
            # Fallback to basic PDF processing
            return self._fallback_pdf_extraction(pdf_path)

    def _fallback_pdf_extraction(self, pdf_path: str) -> str:
        """Fallback PDF extraction using PyPDF2"""
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                logger.info("✅ Used fallback PDF extraction")
                return text
        except Exception as e:
            logger.error(f"❌ Fallback extraction failed: {e}")
            raise Exception("Unable to extract text from PDF")

    def advanced_resume_analysis(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive resume analysis"""
        doc = self.nlp(text)

        # Extract entities
        entities = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],  # Geopolitical entities
            "DATE": [],
            "SKILL": []
        }

        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)

        # Extract skills using pattern matching
        skills = self._extract_technical_skills(text)

        # Calculate experience level
        experience_level = self._calculate_experience_level(text)

        # Extract education information
        education = self._extract_education(text)

        # Calculate resume completeness score
        completeness_score = self._calculate_completeness_score(text)

        analysis = {
            "entities": entities,
            "technical_skills": skills,
            "experience_level": experience_level,
            "education": education,
            "completeness_score": completeness_score,
            "word_count": len(text.split()),
            "sentence_count": len(list(doc.sents))
        }

        self.resume_analysis = analysis
        return analysis

    def _extract_technical_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract technical skills with categorization"""
        skill_patterns = {
            "programming_languages": [
                "python", "java", "javascript", "c++", "c#", "go", "rust", "kotlin",
                "swift", "typescript", "php", "ruby", "scala", "r", "matlab"
            ],
            "web_technologies": [
                "html", "css", "react", "angular", "vue", "node.js", "express",
                "django", "flask", "spring", "laravel", "bootstrap", "sass"
            ],
            "databases": [
                "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
                "cassandra", "dynamodb", "sqlite", "oracle", "sql server"
            ],
            "cloud_platforms": [
                "aws", "azure", "gcp", "google cloud", "heroku", "digitalocean",
                "kubernetes", "docker", "terraform", "ansible"
            ],
            "ml_ai": [
                "machine learning", "deep learning", "tensorflow", "pytorch",
                "scikit-learn", "keras", "pandas", "numpy", "opencv", "nltk"
            ],
            "tools": [
                "git", "jenkins", "jira", "confluence", "slack", "figma",
                "photoshop", "illustrator", "postman", "swagger"
            ]
        }

        detected_skills = {}
        text_lower = text.lower()

        for category, skills in skill_patterns.items():
            detected_skills[category] = []
            for skill in skills:
                if skill in text_lower:
                    detected_skills[category].append(skill)

        return detected_skills

    def _calculate_experience_level(self, text: str) -> str:
        """Calculate experience level based on resume content"""
        text_lower = text.lower()

        # Count experience indicators
        senior_indicators = ["senior", "lead", "principal", "architect", "manager"]
        mid_indicators = ["years", "year", "experience", "developed", "implemented"]
        junior_indicators = ["intern", "trainee", "graduate", "student", "fresher"]

        senior_count = sum(1 for indicator in senior_indicators if indicator in text_lower)
        mid_count = sum(1 for indicator in mid_indicators if indicator in text_lower)
        junior_count = sum(1 for indicator in junior_indicators if indicator in text_lower)

        if senior_count >= 2:
            return "senior"
        elif mid_count >= 3 and junior_count <= 1:
            return "mid-level"
        else:
            return "junior"

    def _extract_education(self, text: str) -> Dict[str, Any]:
        """Extract education information"""
        education_keywords = [
            "bachelor", "master", "phd", "doctorate", "degree", "university",
            "college", "institute", "school", "b.tech", "m.tech", "mba"
        ]

        text_lower = text.lower()
        education_mentions = []

        for keyword in education_keywords:
            if keyword in text_lower:
                education_mentions.append(keyword)

        return {
            "mentions": education_mentions,
            "level": "graduate" if any(kw in education_mentions for kw in ["master", "phd", "m.tech", "mba"]) else "undergraduate"
        }

    def _calculate_completeness_score(self, text: str) -> float:
        """Calculate resume completeness score"""
        required_sections = [
            "education", "experience", "skills", "projects", "contact"
        ]

        text_lower = text.lower()
        present_sections = 0

        for section in required_sections:
            if section in text_lower:
                present_sections += 1

        return (present_sections / len(required_sections)) * 100

# ========== Enhanced Question Generation System ==========
class AdaptiveQuestionGenerator:
    def __init__(self, model_id: str, config: InterviewConfig, api_key: str):
        self.config = config
        self.question_bank = self._initialize_question_bank()
        self.difficulty_levels = ["beginner", "intermediate", "advanced"]
        self.used_questions = set()

        self.agent = Agent(
            name="Adaptive Question Generator",
            model=Groq(id=model_id, api_key=api_key),  # Add api_key parameter
            instructions=self._get_adaptive_instructions(),
            markdown=True
        )

    def _initialize_question_bank(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize categorized question bank"""
        return {
            "intro": {
                "beginner": [
                    "Tell me about yourself and what interests you in this role.",
                    "What motivated you to pursue a career in technology?",
                    "Walk me through your educational background."
                ],
                "intermediate": [
                    "How do you stay updated with the latest technology trends?",
                    "What's the most challenging project you've worked on recently?",
                    "How do you approach learning new technologies?"
                ],
                "advanced": [
                    "How do you evaluate and choose technologies for large-scale projects?",
                    "Describe your approach to technical leadership and mentoring.",
                    "How do you balance technical debt with feature development?"
                ]
            },
            "projects": {
                "beginner": [
                    "Tell me about a recent project you've worked on.",
                    "What technologies did you use in your last project?",
                    "How did you handle challenges in your project?"
                ],
                "intermediate": [
                    "Explain the architecture of your most complex project.",
                    "How did you ensure code quality and testing in your projects?",
                    "Describe a time when you had to optimize performance."
                ],
                "advanced": [
                    "How do you design systems for scalability and maintainability?",
                    "Describe your approach to microservices architecture.",
                    "How do you handle distributed system challenges?"
                ]
            },
            "skills": {
                "beginner": [
                    "What programming languages are you most comfortable with?",
                    "How would you explain [specific technology] to a non-technical person?",
                    "What's your experience with version control systems?"
                ],
                "intermediate": [
                    "Compare and contrast different database technologies you've used.",
                    "How do you approach debugging complex issues?",
                    "Explain the trade-offs between different architectural patterns."
                ],
                "advanced": [
                    "How do you design APIs for large-scale distributed systems?",
                    "Discuss your experience with performance optimization at scale.",
                    "How do you approach system design for high availability?"
                ]
            },
            "behavioral": {
                "beginner": [
                    "Tell me about a time you worked in a team.",
                    "How do you handle feedback and criticism?",
                    "Describe a challenging situation you overcame."
                ],
                "intermediate": [
                    "Tell me about a time you had to learn something quickly.",
                    "How do you prioritize tasks when everything seems urgent?",
                    "Describe a time you had to convince others of your technical approach."
                ],
                "advanced": [
                    "Tell me about a time you had to make a difficult technical decision.",
                    "How do you handle conflicts between technical and business requirements?",
                    "Describe your approach to building and leading technical teams."
                ]
            }
        }

    def _get_adaptive_instructions(self) -> str:
        return """
You are an expert technical interviewer specializing in adaptive questioning.

Your role is to:
1. Generate contextual follow-up questions based on candidate responses
2. Adapt question difficulty based on candidate's demonstrated skill level
3. Ensure questions are relevant to the candidate's background
4. Maintain conversational flow while gathering technical insights
5. Ask ONE clear, focused question at a time
6. while generating questions,do not mentions *,# unwanted symbols

Guidelines:
- Keep questions concise (max 25 words)
- Build on previous responses
- Adjust technical depth appropriately
- Avoid repetitive or generic questions
- Focus on practical experience and problem-solving
"""

    def generate_adaptive_question(self, topic: str, context: str, conversation_history: List[Dict],
                                 state: InterviewState, difficulty_level: str = None) -> str:
        """Generate adaptive question based on context and performance"""

        # Determine difficulty level if not provided
        if not difficulty_level:
            difficulty_level = self._determine_difficulty(state, topic)

        # Try to get a contextual question from the bank first
        bank_question = self._get_bank_question(topic, difficulty_level)

        if bank_question and self.config.adaptive_questioning:
            # Use AI to adapt the bank question to the specific context
            return self._adapt_question_to_context(bank_question, context, conversation_history)
        elif bank_question:
            return bank_question
        else:
            # Generate completely new question using AI
            return self._generate_contextual_question(topic, context, conversation_history, difficulty_level)

    def _determine_difficulty(self, state: InterviewState, topic: str) -> str:
        """Determine appropriate difficulty level based on candidate performance"""
        if state.question_count < 3:
            return "beginner"

        avg_quality = np.mean(state.response_quality_scores) if state.response_quality_scores else 5.0

        if avg_quality >= 8.0:
            return "advanced"
        elif avg_quality >= 6.0:
            return "intermediate"
        else:
            return "beginner"

    def _get_bank_question(self, topic: str, difficulty: str) -> Optional[str]:
        """Get question from predefined bank"""
        if topic in self.question_bank and difficulty in self.question_bank[topic]:
            available_questions = [q for q in self.question_bank[topic][difficulty]
                                 if q not in self.used_questions]

            if available_questions:
                question = random.choice(available_questions)
                self.used_questions.add(question)
                return question
        return None

    def _adapt_question_to_context(self, base_question: str, context: str, history: List[Dict]) -> str:
        """Adapt a base question to the specific context"""
        recent_history = history[-2:] if len(history) > 2 else history
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])

        prompt = f"""
Base question: {base_question}
Resume context: {context[:500]}...
Recent conversation: {history_text}

Adapt this question to be more specific and contextual to this candidate's background.
Keep it concise and focused. Return only the adapted question.
"""

        try:
            response = self.agent.run(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error adapting question: {e}")
            return base_question

    def _generate_contextual_question(self, topic: str, context: str, conversation_history: List[Dict], difficulty_level: str) -> str:
        """Generate a completely new contextual question using AI"""
        recent_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])

        prompt = f"""
Topic: {topic}
Difficulty Level: {difficulty_level}
Resume Context: {context[:500]}...
Recent Conversation: {history_text}

Generate a specific, contextual interview question for this topic and difficulty level.
The question should:
1. Be relevant to the candidate's background
2. Match the {difficulty_level} difficulty level
3. Build on the conversation so far
4. Focus on {topic}
5. Be concise (max 25 words)

Return only the question, nothing else.
"""

        try:
            response = self.agent.run(prompt)
            question = response.content.strip()

            # Clean up the response in case it contains extra text
            if question.startswith('"') and question.endswith('"'):
                question = question[1:-1]

            # Fallback to bank question if AI generation fails
            if not question or len(question.split()) > 30:
                fallback_question = self._get_fallback_question(topic, difficulty_level)
                return fallback_question

            return question

        except Exception as e:
            logger.error(f"Error generating contextual question: {e}")
            # Return fallback question
            return self._get_fallback_question(topic, difficulty_level)

    def _get_fallback_question(self, topic: str, difficulty_level: str) -> str:
        """Get a fallback question when AI generation fails"""
        fallback_questions = {
            "intro": "Tell me more about your background and interests.",
            "projects": "Can you describe a project you're particularly proud of?",
            "skills": "What technical skills do you consider your strongest?",
            "behavioral": "How do you handle challenging situations at work?",
            "hr": "What are your long-term career goals?"
        }

        return fallback_questions.get(topic, "Tell me more about your experience in this area.")

# ========== Enhanced Response Analysis ==========
class AdvancedResponseAnalyzer:
    def __init__(self, config: InterviewConfig):
        self.config = config
        self.skill_keywords = {
            "ml": ["machine learning", "ml", "ai", "tensorflow", "pytorch", "scikit", "pandas", "numpy", "neural", "algorithm"],
            "dsa": ["algorithm", "data structure", "sorting", "searching", "complexity", "big o", "tree", "graph", "hash"],
            "fullstack": ["full stack", "frontend", "backend", "react", "angular", "vue", "node", "express", "django"],
            "backend": ["api", "database", "server", "microservices", "rest", "graphql", "sql", "nosql", "cache"],
            "frontend": ["html", "css", "javascript", "react", "angular", "vue", "ui", "ux", "responsive", "dom"],
            "devops": ["docker", "kubernetes", "aws", "cloud", "ci/cd", "jenkins", "terraform", "ansible", "deployment"],
            "database": ["mysql", "postgresql", "mongodb", "redis", "sql", "nosql", "database", "query", "schema"],
            "cloud": ["aws", "azure", "gcp", "cloud", "lambda", "s3", "ec2", "kubernetes", "serverless"]
        }

        # Enhanced sentiment analysis patterns
        self.confidence_indicators = {
            "high": ["definitely", "absolutely", "certainly", "confident", "sure", "expert", "mastered"],
            "medium": ["think", "believe", "probably", "likely", "familiar", "experience"],
            "low": ["maybe", "perhaps", "not sure", "limited", "basic", "beginner", "learning"]
        }

    def analyze_response(self, response: str, question_type: str = None) -> Dict[str, Any]:
        """Comprehensive response analysis"""
        words = response.split()
        word_count = len(words)

        # Basic quality metrics
        is_weak = word_count < self.config.min_response_words
        is_detailed = word_count > 50
        is_very_detailed = word_count > 100

        # Technical content analysis
        detected_skills = self._analyze_technical_content(response)

        # Confidence level analysis
        confidence_level = self._analyze_confidence(response)

        # Communication quality
        communication_score = self._analyze_communication_quality(response)

        # Problem-solving indicators
        problem_solving_score = self._analyze_problem_solving(response)

        # Experience level indicators
        experience_indicators = self._analyze_experience_level(response)

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            word_count, detected_skills, confidence_level,
            communication_score, problem_solving_score
        )

        return {
            "word_count": word_count,
            "is_weak": is_weak,
            "is_detailed": is_detailed,
            "is_very_detailed": is_very_detailed,
            "detected_skills": detected_skills,
            "confidence_level": confidence_level,
            "communication_score": communication_score,
            "problem_solving_score": problem_solving_score,
            "experience_indicators": experience_indicators,
            "quality_score": quality_score,
            "technical_depth": self._assess_technical_depth(response),
            "specificity_score": self._calculate_specificity(response)
        }

    def _analyze_technical_content(self, response: str) -> Dict[str, int]:
        """Analyze technical content with weighted scoring"""
        detected_skills = {}
        response_lower = response.lower()

        for skill, keywords in self.skill_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in response_lower:
                    # Weight based on keyword specificity
                    weight = 2 if len(keyword.split()) > 1 else 1
                    score += weight

            if score > 0:
                detected_skills[skill] = min(score, 10)  # Cap at 10

        return detected_skills

    def _analyze_confidence(self, response: str) -> str:
        """Analyze confidence level in response"""
        response_lower = response.lower()

        confidence_scores = {"high": 0, "medium": 0, "low": 0}

        for level, indicators in self.confidence_indicators.items():
            for indicator in indicators:
                if indicator in response_lower:
                    confidence_scores[level] += 1

        # Return the level with highest score
        return max(confidence_scores, key=confidence_scores.get)

    def _analyze_communication_quality(self, response: str) -> float:
        """Analyze communication quality"""
        sentences = response.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])

        # Factors for good communication
        factors = []

        # Sentence length (ideal: 10-20 words)
        if 10 <= avg_sentence_length <= 20:
            factors.append(2)
        elif 8 <= avg_sentence_length <= 25:
            factors.append(1)
        else:
            factors.append(0)

        # Use of examples
        if any(word in response.lower() for word in ["example", "instance", "like", "such as"]):
            factors.append(2)

        # Clear structure
        if any(word in response.lower() for word in ["first", "second", "then", "next", "finally"]):
            factors.append(1)

        return min(10, sum(factors))

    def _analyze_problem_solving(self, response: str) -> float:
        """Analyze problem-solving approach in response"""
        problem_solving_indicators = [
            "approach", "strategy", "solution", "solve", "analyze", "break down",
            "step by step", "process", "method", "technique", "consider", "evaluate"
        ]

        response_lower = response.lower()
        score = 0

        for indicator in problem_solving_indicators:
            if indicator in response_lower:
                score += 1

        return min(10, score)

    def _analyze_experience_level(self, response: str) -> Dict[str, int]:
        """Analyze experience level indicators"""
        experience_patterns = {
            "junior": ["learning", "studying", "new to", "beginner", "started", "basic"],
            "mid": ["experience", "worked with", "familiar", "used", "developed", "implemented"],
            "senior": ["led", "architected", "designed", "mentored", "optimized", "scaled", "expert"]
        }

        response_lower = response.lower()
        indicators = {}

        for level, patterns in experience_patterns.items():
            count = sum(1 for pattern in patterns if pattern in response_lower)
            indicators[level] = count

        return indicators

    def _calculate_quality_score(self, word_count: int, skills: Dict, confidence: str,
                               communication: float, problem_solving: float) -> float:
        """Calculate overall response quality score"""

        # Word count component (0-3 points)
        if word_count < 10:
            word_score = 0
        elif word_count < 30:
            word_score = 1
        elif word_count < 80:
            word_score = 2
        else:
            word_score = 3

        # Technical content (0-3 points)
        skill_score = min(3, len(skills))

        # Confidence (0-2 points)
        confidence_score = {"high": 2, "medium": 1, "low": 0}.get(confidence, 0)

        # Communication and problem-solving (0-2 points each, scaled)
        comm_score = min(2, communication / 5)
        ps_score = min(2, problem_solving / 5)

        total_score = word_score + skill_score + confidence_score + comm_score + ps_score
        return min(10, total_score)

    def _assess_technical_depth(self, response: str) -> str:
        """Assess technical depth of response"""
        depth_indicators = {
            "surface": ["use", "know", "heard", "basic"],
            "intermediate": ["implement", "develop", "configure", "integrate"],
            "deep": ["optimize", "architect", "design", "scale", "performance", "architecture"]
        }

        response_lower = response.lower()
        depth_scores = {}

        for level, indicators in depth_indicators.items():
            score = sum(1 for indicator in indicators if indicator in response_lower)
            depth_scores[level] = score

        return max(depth_scores, key=depth_scores.get)

    def _calculate_specificity(self, response: str) -> float:
        """Calculate how specific/concrete the response is"""
        specific_indicators = [
            "project", "company", "version", "years", "months", "team",
            "users", "performance", "time", "size", "scale"
        ]

        response_lower = response.lower()
        specificity = sum(1 for indicator in specific_indicators if indicator in response_lower)

        return min(10, specificity)

# ========== Visualization & Analytics ==========
# ========== Updated Interview Analytics Class ==========
class InterviewAnalytics:
    def __init__(self, config: InterviewConfig):
        self.config = config

    def create_performance_dashboard(self, state: InterviewState, conversation: List[Dict]) -> None:
        """Create comprehensive performance dashboard"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Interview Performance Dashboard', fontsize=16, fontweight='bold')

            # 1. Quality Score Progression
            self._plot_quality_progression(axes[0, 0], state)

            # 2. Skills Radar Chart
            self._plot_skills_radar(axes[0, 1], state)

            # 3. Topic Coverage
            self._plot_topic_coverage(axes[0, 2], state)

            # 4. Response Length Distribution
            self._plot_response_lengths(axes[1, 0], conversation)

            # 5. Question Type Distribution
            self._plot_question_types(axes[1, 1], state)

            # 6. Engagement Over Time
            self._plot_engagement_timeline(axes[1, 2], state)

            plt.tight_layout()

            # Save to current directory instead of /tmp/
            plt.savefig('interview_dashboard.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("📊 Dashboard saved as 'interview_dashboard.png'")

        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")

    # ... (keep all the existing plotting methods unchanged) ...

    def generate_word_cloud(self, conversation: List[Dict]) -> None:
        """Generate word cloud from candidate responses"""
        try:
            candidate_text = " ".join([msg['content'] for msg in conversation if msg['role'] == 'user'])

            if not candidate_text.strip():
                print("No candidate responses to generate word cloud")
                return

            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(candidate_text)

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Candidate Response Word Cloud', fontsize=16, fontweight='bold')
            plt.tight_layout()

            # Save to current directory instead of /tmp/
            plt.savefig('interview_wordcloud.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("☁️ Word cloud saved as 'interview_wordcloud.png'")

        except Exception as e:
            logger.error(f"Error generating word cloud: {e}")

    def export_detailed_report(self, state: InterviewState, conversation: List[Dict],
                             resume_analysis: Dict) -> str:
        """Export detailed interview report"""

        # Calculate additional metrics
        avg_quality = np.mean(state.response_quality_scores) if state.response_quality_scores else 0
        total_words = sum(len(msg['content'].split()) for msg in conversation if msg['role'] == 'user')
        response_count = len([msg for msg in conversation if msg['role'] == 'user'])
        avg_response_length = total_words / response_count if response_count > 0 else 0

        report = {
            "interview_summary": {
                "total_questions": state.question_count,
                "total_responses": response_count,
                "average_quality_score": round(avg_quality, 2),
                "average_response_length_words": round(avg_response_length, 1),
                "total_words_spoken": total_words,
                "topic_coverage": state.topic_coverage,
                "topics_covered_count": sum(state.topic_coverage.values()),
                "topics_total": len(state.topic_coverage),
                "coverage_percentage": round((sum(state.topic_coverage.values()) / len(state.topic_coverage)) * 100, 1),
                "overall_engagement": round(state.engagement_score, 2),
                "difficulty_level": state.difficulty_level,
                "interview_duration_questions": state.question_count
            },
            "detailed_scores": {
                "quality_scores_per_question": state.response_quality_scores,
                "quality_trend": "improving" if len(state.response_quality_scores) > 1 and
                               state.response_quality_scores[-1] > state.response_quality_scores[0] else "stable",
                "consistency_score": round(1 / (1 + np.std(state.response_quality_scores)), 2) if len(state.response_quality_scores) > 1 else 1.0
            },
            "skills_assessment": {
                "detected_skills": state.strength_scores,
                "strongest_skills": [skill for skill, score in sorted(state.strength_scores.items(), key=lambda x: x[1], reverse=True)[:3] if score > 0],
                "technical_depth_distribution": state.technical_depth,
                "primary_technical_level": max(state.technical_depth.items(), key=lambda x: x[1])[0] if any(state.technical_depth.values()) else "beginner"
            },
            "question_analysis": {
                "question_types_asked": dict(Counter(state.question_types)),
                "follow_ups_per_topic": state.follow_ups,
                "response_times": state.response_times if state.response_times else []
            },
            "conversation_log": conversation,
            "resume_analysis": resume_analysis,
            "recommendations": self._generate_recommendations(state),
            "performance_rating": self._calculate_overall_rating(state),
            "timestamp": datetime.now().isoformat(),
            "report_version": "1.0"
        }

        # Save to current directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f'interview_report_{timestamp}.json'

        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"Report saved to {report_filename}")
            print(f"📄 Detailed report saved as '{report_filename}'")

            # Also create a human-readable summary
            self._create_human_readable_summary(report, timestamp)

            return report_filename

        except Exception as e:
            logger.error(f"Error saving report: {e}")
            print(f"❌ Error saving report: {e}")
            return None

    def _create_human_readable_summary(self, report: Dict, timestamp: str) -> None:
        """Create a human-readable summary file"""
        try:
            summary_filename = f'interview_summary_{timestamp}.txt'

            with open(summary_filename, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("           AI TECHNICAL INTERVIEW REPORT\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"Interview Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Report Generated: {report['timestamp']}\n\n")

                # Interview Summary
                summary = report['interview_summary']
                f.write("INTERVIEW OVERVIEW\n")
                f.write("-" * 20 + "\n")
                f.write(f"Questions Asked: {summary['total_questions']}\n")
                f.write(f"Responses Given: {summary['total_responses']}\n")
                f.write(f"Average Response Quality: {summary['average_quality_score']}/10\n")
                f.write(f"Topic Coverage: {summary['topics_covered_count']}/{summary['topics_total']} ({summary['coverage_percentage']}%)\n")
                f.write(f"Overall Engagement: {summary['overall_engagement']}/10\n")
                f.write(f"Difficulty Level: {summary['difficulty_level'].title()}\n")
                f.write(f"Average Response Length: {summary['average_response_length_words']} words\n")
                f.write(f"Total Words Spoken: {summary['total_words_spoken']}\n\n")

                # Performance Rating
                f.write("PERFORMANCE RATING\n")
                f.write("-" * 18 + "\n")
                f.write(f"Overall Rating: {report['performance_rating']['rating']}\n")
                f.write(f"Rating Description: {report['performance_rating']['description']}\n\n")

                # Skills Assessment
                skills = report['skills_assessment']
                f.write("SKILLS ASSESSMENT\n")
                f.write("-" * 17 + "\n")
                f.write(f"Primary Technical Level: {skills['primary_technical_level'].title()}\n")
                f.write("Strongest Skills:\n")
                for skill in skills['strongest_skills']:
                    f.write(f"  • {skill.upper()}\n")

                f.write("\nDetected Skills (by mentions):\n")
                for skill, score in sorted(skills['detected_skills'].items(), key=lambda x: x[1], reverse=True):
                    if score > 0:
                        f.write(f"  • {skill.upper()}: {score} mentions\n")
                f.write("\n")

                # Topic Coverage
                f.write("TOPIC COVERAGE\n")
                f.write("-" * 14 + "\n")
                for topic, covered in summary['topic_coverage'].items():
                    status = "✓ Covered" if covered else "✗ Not Covered"
                    f.write(f"  • {topic.title()}: {status}\n")
                f.write("\n")

                # Recommendations
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 15 + "\n")
                for i, rec in enumerate(report['recommendations'], 1):
                    f.write(f"{i}. {rec}\n")
                f.write("\n")

                # Performance Trend
                detailed = report['detailed_scores']
                f.write("PERFORMANCE ANALYSIS\n")
                f.write("-" * 19 + "\n")
                f.write(f"Quality Trend: {detailed['quality_trend'].title()}\n")
                f.write(f"Consistency Score: {detailed['consistency_score']}/1.0\n")
                f.write(f"Quality Scores per Question: {detailed['quality_scores_per_question']}\n\n")

                f.write("=" * 60 + "\n")
                f.write("End of Report\n")
                f.write("=" * 60 + "\n")

            print(f"📋 Human-readable summary saved as '{summary_filename}'")

        except Exception as e:
            logger.error(f"Error creating summary: {e}")
            print(f"❌ Error creating summary: {e}")

    def _calculate_overall_rating(self, state: InterviewState) -> Dict[str, str]:
        """Calculate overall performance rating"""
        if not state.response_quality_scores:
            return {"rating": "Incomplete", "description": "Insufficient data for rating"}

        avg_quality = np.mean(state.response_quality_scores)
        topic_coverage = sum(state.topic_coverage.values()) / len(state.topic_coverage)
        engagement = state.engagement_score / 10

        # Weighted score
        overall_score = (avg_quality * 0.4 + topic_coverage * 10 * 0.3 + engagement * 10 * 0.3) / 10

        if overall_score >= 8.5:
            return {"rating": "Excellent", "description": "Outstanding performance with strong technical knowledge and communication"}
        elif overall_score >= 7.0:
            return {"rating": "Good", "description": "Solid performance with good technical understanding"}
        elif overall_score >= 5.5:
            return {"rating": "Average", "description": "Adequate performance with room for improvement"}
        elif overall_score >= 4.0:
            return {"rating": "Below Average", "description": "Needs improvement in technical knowledge or communication"}
        else:
            return {"rating": "Poor", "description": "Significant improvement needed in multiple areas"}

    def _generate_recommendations(self, state: InterviewState) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []

        # Quality-based recommendations
        if not state.response_quality_scores:
            recommendations.append("Complete more questions to get better assessment")
            return recommendations

        avg_quality = np.mean(state.response_quality_scores)

        if avg_quality < 4:
            recommendations.append("Focus on providing more detailed responses with specific examples from your experience")
            recommendations.append("Practice explaining technical concepts clearly and concisely")
        elif avg_quality < 6:
            recommendations.append("Good foundation - try to include more technical depth and specific examples")
            recommendations.append("Consider elaborating on your problem-solving approach")
        elif avg_quality < 8:
            recommendations.append("Strong responses overall - continue to demonstrate technical leadership and decision-making")
        else:
            recommendations.append("Excellent technical communication demonstrated throughout the interview")

        # Skills-based recommendations
        skill_scores = state.strength_scores
        if skill_scores:
            strong_skills = [skill for skill, score in skill_scores.items() if score >= 5]
            weak_skills = [skill for skill, score in skill_scores.items() if score < 2 and score > 0]

            if strong_skills:
                recommendations.append(f"Continue leveraging your strong skills: {', '.join(strong_skills)}")

            if weak_skills:
                recommendations.append(f"Consider strengthening these areas: {', '.join(weak_skills)}")

        # Topic coverage recommendations
        uncovered_topics = [topic for topic, covered in state.topic_coverage.items() if not covered]
        if uncovered_topics:
            recommendations.append(f"Prepare examples and stories for these topics: {', '.join(uncovered_topics)}")

        # Consistency recommendations
        if len(state.response_quality_scores) > 2:
            consistency = np.std(state.response_quality_scores)
            if consistency > 2:
                recommendations.append("Work on maintaining consistent response quality throughout interviews")

        # Engagement recommendations
        if state.engagement_score < 6:
            recommendations.append("Consider being more enthusiastic and engaged in your responses")

        return recommendations

# ========== Updated Main Interview System Methods ==========

def _generate_interview_conclusion(self) -> str:
    """Generate interview conclusion and summary"""

    if not self.state.response_quality_scores:
        return "Interview completed, but no responses were recorded. Please try again."

    # Calculate final scores
    avg_quality = np.mean(self.state.response_quality_scores)
    topic_coverage_pct = (sum(self.state.topic_coverage.values()) / len(self.state.topic_coverage)) * 100

    # Get strongest skills
    strong_skills = [skill for skill, score in sorted(self.state.strength_scores.items(),
                                                     key=lambda x: x[1], reverse=True)[:3] if score > 0]

    # Generate summary
    summary = f"""
🎉 **Interview Complete!**

**Performance Summary:**
• Questions Answered: {self.state.question_count}
• Average Response Quality: {avg_quality:.1f}/10
• Topics Covered: {sum(self.state.topic_coverage.values())}/{len(self.state.topic_coverage)} ({topic_coverage_pct:.1f}%)
• Overall Engagement: {self.state.engagement_score:.1f}/10
• Technical Level: {self.state.difficulty_level.title()}

**Strong Areas Demonstrated:**
"""

    if strong_skills:
        for skill in strong_skills:
            summary += f"• {skill.upper()}\n"
    else:
        summary += "• Communication and engagement\n"

    # Add performance rating
    rating = self.analytics._calculate_overall_rating(self.state)
    summary += f"\n**Overall Rating: {rating['rating']}**\n"
    summary += f"*{rating['description']}*\n"

    summary += "\n**📊 Generating detailed analytics and reports...**"

    # Generate final analytics
    try:
        if self.config.save_visualizations:
            print("\n📈 Creating performance dashboard...")
            self.analytics.create_performance_dashboard(self.state, self.conversation_history)

            print("☁️ Generating word cloud...")
            self.analytics.generate_word_cloud(self.conversation_history)

        print("📄 Exporting detailed report...")
        report_path = self.analytics.export_detailed_report(
            self.state,
            self.conversation_history,
            self.resume_processor.resume_analysis
        )

        if report_path:
            summary += f"\n\n✅ **Files Generated:**\n"
            summary += f"• Detailed JSON Report: {report_path}\n"
            summary += f"• Human-readable Summary: interview_summary_*.txt\n"
            if self.config.save_visualizations:
                summary += f"• Performance Dashboard: interview_dashboard.png\n"
                summary += f"• Response Word Cloud: interview_wordcloud.png\n"

    except Exception as e:
        logger.error(f"Error generating final analytics: {e}")
        summary += f"\n⚠️ Error generating some reports: {e}"

    summary += "\n\n**Thank you for your time! All interview data has been saved to your current directory.**"

    return summary

# Add this method to the AIInterviewSystem class
def export_interview_data(self) -> str:
    """Export complete interview data"""
    try:
        return self.analytics.export_detailed_report(
            self.state,
            self.conversation_history,
            self.resume_processor.resume_analysis
        )
    except Exception as e:
        logger.error(f"Error exporting interview data: {e}")
        return None

# ========== Main Interview System ==========
class AIInterviewSystem:
    def __init__(self, config: InterviewConfig = None):
        self.config = config or InterviewConfig()
        self.api_manager = APIKeyManager()
        self.state = InterviewState()
        self.conversation_history = []

        # Initialize components
        self.resume_processor = EnhancedResumeProcessor(
            self.api_manager.llama_api_key,
            self.config
        )
        self.question_generator = AdaptiveQuestionGenerator(
            self.config.llm_model,
            self.config,
            self.api_manager.groq_api_key  # Add the API key
        )
        self.response_analyzer = AdvancedResponseAnalyzer(self.config)
        self.analytics = InterviewAnalytics(self.config)

        # Initialize interviewer agent
        self.interviewer = Agent(
            name="AI Technical Interviewer",
            model=Groq(id=self.config.llm_model, api_key=self.api_manager.groq_api_key),  # Add api_key
            instructions=self._get_interviewer_instructions(),
            markdown=True
        )

        logger.info("✅ AI Interview System initialized successfully")

    def _get_interviewer_instructions(self) -> str:
        return """
You are an expert technical and HR interviewer conducting a friendly, human-like interview.

Your responsibilities:
1. Start with a brief self-introduction, mentioning you are an AI interviewer and setting a supportive, conversational tone.
2. Ask thoughtful, relevant questions based on the candidate's background.
3. Use small talk and light humor to put the candidate at ease (e.g., "No pressure, just a friendly chat!").
4. Provide encouraging, empathetic feedback (e.g., "That's a great example!", "I appreciate your honesty.").
5. Vary your responses and feedback to sound natural and avoid repetition.
6. Adapt your questioning style based on candidate responses.
7. Focus on practical experience, problem-solving, and communication skills.
8. Keep questions concise, clear, and conversational.
9. If the candidate seems nervous, reassure them and offer a moment to relax.

Interview flow:
- Begin with a self-intro and some friendly small talk.
- Move to introductory questions about the candidate and their background.
- Progress to technical skills, projects, and scenario-based questions.
- Include behavioral and HR questions (e.g., teamwork, conflict, aspirations).
- Maintain a conversational, supportive, and human-like tone throughout.
- Ask ONE question at a time.
- Provide brief, varied, and encouraging responses before next questions.
- End with a warm thank you and a summary of the interview.

Remember: Your goal is to make the candidate feel comfortable, supported, and engaged while evaluating both technical and interpersonal qualities.
"""

    def load_resume(self, resume_path: str) -> Dict[str, Any]:
        """Load and process resume"""
        try:
            logger.info(f"📄 Processing resume: {resume_path}")

            # Extract text from resume
            resume_text = self.resume_processor.extract_text_from_pdf(resume_path)

            # Perform comprehensive analysis
            analysis = self.resume_processor.advanced_resume_analysis(resume_text)

            # Create embeddings for resume content
            self._create_resume_embeddings(resume_text)

            logger.info("✅ Resume processed successfully")
            return analysis

        except Exception as e:
            logger.error(f"❌ Error processing resume: {e}")
            raise

    def _create_resume_embeddings(self, text: str):
        """Create embeddings for resume content"""
        # Split text into chunks
        chunks = self._chunk_text(text, self.config.max_tokens_per_chunk)

        # Generate embeddings
        embeddings = self.resume_processor.embedder.encode(chunks)

        # Add to FAISS index
        self.resume_processor.faiss_index.add(embeddings.astype('float32'))

        # Store metadata
        self.resume_processor.chunk_metadata = [
            {"text": chunk, "index": i} for i, chunk in enumerate(chunks)
        ]

        logger.info(f"Created {len(chunks)} embeddings from resume")

    def _chunk_text(self, text: str, max_tokens: int) -> List[str]:
        """Split text into chunks"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) <= max_tokens:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def start_interview(self, resume_analysis: Dict[str, Any]) -> str:
        """Start the interview process"""
        logger.info("🎯 Starting interview session")

        # Initialize conversation
        opening_message = self._generate_opening_question(resume_analysis)

        # Personalize with candidate name if available
        name = getattr(self, 'candidate_name', None) or 'there'
        opening_message = opening_message.replace('{name}', name)

        self.conversation_history.append({
            "role": "assistant",
            "content": opening_message,
            "timestamp": time.time()
        })

        self.state.question_count += 1
        self.state.question_types.append("intro")

        return opening_message

    def _generate_opening_question(self, resume_analysis: Dict[str, Any]) -> str:
        """Generate personalized, human-like opening question"""
        context = f"""
Resume Analysis:
- Experience Level: {resume_analysis.get('experience_level', 'unknown')}
- Technical Skills: {resume_analysis.get('technical_skills', {})}
- Education: {resume_analysis.get('education', {})}
- Completeness Score: {resume_analysis.get('completeness_score', 0)}%
"""

        opening_question = self.question_generator.generate_adaptive_question(
            "intro", context, self.conversation_history, self.state
        )

        greetings = [
            "Let's start with something easy: ",
            "To break the ice, ",
            "Before we dive into the technical stuff, ",
            "I'd love to hear more about you: ",
            "Let's get to know you a bit: "
        ]
        greeting = random.choice(greetings)
        small_talk = random.choice([
            "No need to be nervous, just share your thoughts! ",
            "This is just a conversation, so feel free to be yourself. ",
            "Take your time—I'm here to listen. ",
            "Ready when you are! ",
            "Let's make this fun and insightful! "
        ])
        return f"{greeting}{opening_question}\n{small_talk}"

    def process_response(self, user_response: str) -> str:
        """Process candidate response and generate next question"""
        start_time = time.time()

        try:
            # Record user response
            self.conversation_history.append({
                "role": "user",
                "content": user_response,
                "timestamp": start_time
            })

            # Analyze response
            analysis = self.response_analyzer.analyze_response(
                user_response,
                self.state.question_types[-1] if self.state.question_types else None
            )

            # Update state based on analysis
            self._update_interview_state(analysis, user_response)

            # Check if interview should continue
            if self.state.question_count >= self.config.max_questions:
                return self._generate_interview_conclusion()

            # Generate next question
            next_question = self._generate_next_question(analysis)

            # Personalize with candidate name if available
            name = getattr(self, 'candidate_name', None) or 'there'
            next_question = next_question.replace('{name}', name)

            # Record response time
            response_time = time.time() - start_time
            self.state.response_times.append(response_time)

            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": next_question,
                "timestamp": time.time()
            })

            # Speak only the actual question/reply, not markdown headers
            spoken_text = get_spoken_text(next_question)
            if spoken_text:
                text_to_speech(spoken_text)

            return next_question

        except Exception as e:
            logger.error(f"Error processing response: {e}")
            return "I apologize, but I encountered an error. Could you please repeat your response?"

    def _update_interview_state(self, analysis: Dict[str, Any], response: str):
        """Update interview state based on response analysis"""

        # Update quality scores
        self.state.response_quality_scores.append(analysis['quality_score'])
        self.state.candidate_responses.append(response)

        # Update skill strengths
        for skill, score in analysis['detected_skills'].items():
            if skill in self.state.strength_scores:
                self.state.strength_scores[skill] += score

        # Update technical depth
        depth = analysis['technical_depth']
        if depth in self.state.technical_depth:
            self.state.technical_depth[depth] += 1

        # Update engagement score (moving average)
        current_engagement = min(10, analysis['quality_score'] + analysis['communication_score'])
        self.state.engagement_score = (self.state.engagement_score * 0.8 + current_engagement * 0.2)

        # Determine difficulty adjustment
        if self.config.difficulty_adaptation:
            self._adjust_difficulty_level(analysis)

    def _adjust_difficulty_level(self, analysis: Dict[str, Any]):
        """Adjust difficulty level based on performance"""
        quality_score = analysis['quality_score']

        if quality_score >= 8 and self.state.difficulty_level != "advanced":
            self.state.difficulty_level = "advanced"
            logger.info("📈 Difficulty increased to advanced")
        elif quality_score <= 4 and self.state.difficulty_level != "beginner":
            self.state.difficulty_level = "beginner"
            logger.info("📉 Difficulty decreased to beginner")
        elif 5 <= quality_score <= 7 and self.state.difficulty_level != "intermediate":
            self.state.difficulty_level = "intermediate"
            logger.info("📊 Difficulty adjusted to intermediate")

    def _generate_next_question(self, analysis: Dict[str, Any]) -> str:
        """Generate the next appropriate question"""

        # Determine next topic
        next_topic = self._select_next_topic()

        # Get relevant context from resume
        context = self._get_relevant_context(analysis['detected_skills'])

        # Generate question
        question = self.question_generator.generate_adaptive_question(
            next_topic,
            context,
            self.conversation_history,
            self.state,
            self.state.difficulty_level
        )

        # Update state
        self.state.question_count += 1
        self.state.question_types.append(next_topic)

        # Mark topic as covered
        if next_topic in self.state.topic_coverage:
            self.state.topic_coverage[next_topic] = True

        # Add brief encouraging feedback
        feedback = self._generate_brief_feedback(analysis)

        return f"{feedback}\n\n{question}"

    def _select_next_topic(self) -> str:
        """Select next topic based on interview progress and weights"""

        # Get uncovered topics
        uncovered_topics = [topic for topic, covered in self.state.topic_coverage.items() if not covered]

        # If all topics covered, select based on performance
        if not uncovered_topics:
            # Focus on weaker areas
            weak_skills = [skill for skill, score in self.state.strength_scores.items() if score < 3]
            if weak_skills:
                return "skills"
            else:
                return "behavioral"

        # Select topic based on weights and progress
        topic_priorities = {
            "projects": 0.3,
            "skills": 0.25,
            "behavioral": 0.2,
            "internships": 0.15,
            "hr": 0.1
        }

        # Weight uncovered topics higher
        weighted_topics = []
        for topic in uncovered_topics:
            weight = topic_priorities.get(topic, 0.1)
            # Boost weight if we haven't asked many questions in this area
            follow_up_count = self.state.follow_ups.get(topic, 0)
            if follow_up_count < self.config.max_follow_ups_per_topic:
                weight *= 1.5
            weighted_topics.append((topic, weight))

        # Select topic with highest weight
        if weighted_topics:
            return max(weighted_topics, key=lambda x: x[1])[0]
        else:
            return random.choice(list(topic_priorities.keys()))

    def _get_relevant_context(self, detected_skills: Dict[str, int]) -> str:
        """Get relevant context from resume based on detected skills"""

        if not detected_skills or not self.resume_processor.chunk_metadata:
            return ""

        # Create query from detected skills
        query = " ".join(detected_skills.keys())

        try:
            # Get query embedding
            query_embedding = self.resume_processor.embedder.encode([query])

            # Search for relevant chunks
            D, I = self.resume_processor.faiss_index.search(query_embedding.astype('float32'), k=2)

            # Get relevant text chunks
            relevant_chunks = []
            for idx in I[0]:
                if idx < len(self.resume_processor.chunk_metadata):
                    relevant_chunks.append(self.resume_processor.chunk_metadata[idx]['text'])

            return " ".join(relevant_chunks)

        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            return ""

    def _generate_brief_feedback(self, analysis: Dict[str, Any]) -> str:
        """Generate brief encouraging feedback"""
        quality_score = analysis['quality_score']

        if quality_score >= 8:
            feedback_options = [
                "Great detailed response, {name}!",
                "Excellent technical insight, {name}!",
                "Very comprehensive answer, {name}!",
                "Impressive depth of knowledge, {name}!",
                "That's a fantastic example, {name}!",
                "You explained that really well, {name}!",
                "I can see you're very comfortable with this topic, {name}!",
                "Wonderful, thank you for sharing, {name}!"
            ]
        elif quality_score >= 6:
            feedback_options = [
                "Good answer, {name}!",
                "Nice explanation, {name}!",
                "Solid response, {name}!",
                "Thanks for that insight, {name}!",
                "That's a clear explanation, {name}!",
                "Appreciate your thoughtful answer, {name}!",
                "You're doing great, keep it up, {name}!"
            ]
        else:
            feedback_options = [
                "Thank you for sharing, {name}!",
                "I appreciate your honesty, {name}!",
                "Thanks for the response, {name}!",
                "Good to know, {name}!",
                "No worries, let's move on to the next one, {name}.",
                "Feel free to elaborate more if you'd like, {name}!",
                "Everyone starts somewhere—thanks for your effort, {name}!"
            ]

        # Add some human-like variation
        feedback = random.choice(feedback_options)
        if random.random() < 0.2:
            feedback += " 😊"
        elif random.random() < 0.1:
            feedback += " (Take a breath, you're doing well!)"
        name = getattr(self, 'candidate_name', None) or 'there'
        return feedback.replace('{name}', name)

    def _generate_interview_conclusion(self) -> str:
        """Generate interview conclusion and summary"""

        # Calculate final scores
        avg_quality = np.mean(self.state.response_quality_scores) if self.state.response_quality_scores else 0

        # Generate summary
        summary = f"""
🎉 **Interview Complete!**

**Performance Summary:**
- Questions Answered: {self.state.question_count}
- Average Response Quality: {avg_quality:.1f}/10
- Topics Covered: {sum(self.state.topic_coverage.values())}/{len(self.state.topic_coverage)}
- Overall Engagement: {self.state.engagement_score:.1f}/10

**Strong Areas:**
"""

        # Add strong skills
        strong_skills = [skill for skill, score in self.state.strength_scores.items() if score >= 5]
        if strong_skills:
            summary += "- " + "\n- ".join(strong_skills)
        else:
            summary += "- Communication and engagement"

        summary += "\n\n**Thank you for your time! The interview data has been analyzed and a detailed report will be generated.**"

        # Generate final analytics
        if self.config.save_visualizations:
            self.analytics.create_performance_dashboard(self.state, self.conversation_history)
            self.analytics.generate_word_cloud(self.conversation_history)

        return summary

    def get_interview_analytics(self) -> Dict[str, Any]:
        """Get comprehensive interview analytics"""
        return {
            "state": asdict(self.state),
            "conversation_length": len(self.conversation_history),
            "total_duration": self.conversation_history[-1]['timestamp'] - self.conversation_history[0]['timestamp'] if len(self.conversation_history) > 1 else 0,
            "performance_metrics": {
                "avg_quality": np.mean(self.state.response_quality_scores) if self.state.response_quality_scores else 0,
                "consistency": np.std(self.state.response_quality_scores) if len(self.state.response_quality_scores) > 1 else 0,
                "topic_coverage_rate": sum(self.state.topic_coverage.values()) / len(self.state.topic_coverage) * 100,
                "engagement_level": self.state.engagement_score
            }
        }

    def export_interview_data(self) -> str:
        """Export complete interview data"""
        return self.analytics.export_detailed_report(
            self.state,
            self.conversation_history,
            self.resume_processor.resume_analysis
        )

# ========== Main Execution Function ==========
def main():
    """Main function to run the AI Interview System"""

    print("🤖 AI Technical Interview System")
    print("=" * 50)

    try:
        # Initialize system
        config = InterviewConfig()
        interview_system = AIInterviewSystem(config)

        # Get resume path
        resume_path = input("📄 Enter path to resume PDF: ").strip()

        if not os.path.exists(resume_path):
            print("❌ Resume file not found!")
            return

        # Process resume
        print("\n📊 Processing resume...")
        resume_analysis = interview_system.load_resume(resume_path)

        print(f"✅ Resume analysis complete!")
        print(f"   - Experience Level: {resume_analysis.get('experience_level', 'unknown')}")
        print(f"   - Skills Found: {len(resume_analysis.get('technical_skills', {}))}")
        print(f"   - Completeness: {resume_analysis.get('completeness_score', 0):.1f}%")

        # Start interview
        print("\n🎯 Starting interview...")
        print("=" * 50)

        # Human-like self introduction
        self_intro = ("Hello! My name is MJ, and I'll be your interviewer today. "
            "We'll be having a friendly conversation covering both technical and HR topics. "
            "Feel free to relax, take your time, and let me know if you need any clarification at any point.\n"
            "Let's get to know each other and see how your experience fits with the role!")
        text_to_speech(self_intro)
        print(f"\n🤖 Interviewer: {self_intro}")

        # Ask for candidate's name directly (replacement for name_utils extraction)
        print("\nTo make this more personal, may I know your name?")
        candidate_name = input("\n👤 Your name: ").strip()
        if not candidate_name:
            candidate_name = 'friend'
        interview_system.candidate_name = candidate_name  # Store for later use

        # Candidate self-introduction step (voice or text)
        candidate_intro_prompt = "Could you please tell me a little about your background?"
        text_to_speech(candidate_intro_prompt)
        print(f"\n🤖 Interviewer: {candidate_intro_prompt}")
        print("Would you like to introduce yourself via voice? (y/n): ", end='')
        use_voice_intro = input().strip().lower() == 'y'
        candidate_intro = ""
        if use_voice_intro:
            try:
                from voice_utils import record_and_transcribe
                print("(Press Enter to start recording. Speak clearly into your mic.)")
                input()
                candidate_intro = record_and_transcribe(duration=8)
                print(f"You said: {candidate_intro}")
            except Exception as e:
                print(f"❌ Audio input failed: {e}")
                print("Please type your self-introduction instead.")
                candidate_intro = input("\n👤 Your self-introduction: ").strip()
        else:
            candidate_intro = input("\n👤 Your self-introduction: ").strip()

        # Ask candidate if they want to use voice input for answers
        print("\nWould you like to answer interview questions via voice? (y/n): ", end='')
        use_voice = input().strip().lower() == 'y'
        if use_voice:
            try:
                from voice_utils import record_and_transcribe
            except ImportError:
                print("voice_utils or dependencies missing. Falling back to text input.")
                use_voice = False

        # Personalized opening question
        opening_question = interview_system.start_interview(resume_analysis)
        text_to_speech(opening_question.replace('{name}', candidate_name))
        print(f"\n🤖 Interviewer: {opening_question.replace('{name}', candidate_name)}")

        # Interview loop
        while interview_system.state.question_count < config.max_questions:
            print(f"\n👤 Your response (Question {interview_system.state.question_count}/{config.max_questions}):")
            if use_voice:
                print("(Press Enter to start recording. Speak clearly into your mic.)")
                input()
                user_response = record_and_transcribe(duration=10)
                print(f"You said: {user_response}")
            else:
                user_response = input().strip()

            if not user_response:
                print("Please provide a response.")
                continue

            if user_response.lower() in ['quit', 'exit', 'stop']:
                print("Interview ended by user.")
                break

            # Process response and get next question
            next_question = interview_system.process_response(user_response)
            print(f"\n🤖 Interviewer: {next_question}")

            # Show progress
            if interview_system.config.real_time_feedback:
                current_score = interview_system.state.response_quality_scores[-1]
                print(f"   💡 Response Quality: {current_score:.1f}/10")

        # Interview complete
        print("\n" + "=" * 50)
        print("🎉 Interview Complete!")

        # Generate final analytics
        analytics = interview_system.get_interview_analytics()
        print(f"\n📊 Final Performance:")
        print(f"   - Average Quality Score: {analytics['performance_metrics']['avg_quality']:.1f}/10")
        print(f"   - Topic Coverage: {analytics['performance_metrics']['topic_coverage_rate']:.1f}%")
        print(f"   - Engagement Level: {analytics['performance_metrics']['engagement_level']:.1f}/10")

        # Export data
        report_path = interview_system.export_interview_data()
        print(f"   - Detailed report saved: {report_path}")

        print("\n✅ Thank you for using the AI Interview System!")

    except KeyboardInterrupt:
        print("\n\n⏹️  Interview interrupted by user.")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        print(f"❌ An error occurred: {e}")

# ========== Utility Functions ==========

import os

def get_spoken_text(text: str) -> str:
    """Filter out markdown headers (lines starting with #, ##, ###) and empty lines."""
    lines = text.splitlines()
    filtered = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
    return ' '.join(filtered)

def text_to_speech(text: str, lang: str = 'en'):
    """Convert text to speech using Google TTS and play it (Windows-safe)."""
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_path = fp.name
            tts.save(temp_path)
        try:
            playsound(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    except Exception as e:
        logger.error(f"TTS playback failed: {e}")


  
# ========== Entry Point ==========
if __name__ == "__main__":
    # Entry point for local execution
    
    # Run main function
    main()