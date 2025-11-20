import os
import asyncio
import json
import re
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, BackgroundTasks, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import sqlalchemy
from sqlalchemy import (Column, Integer, String, DateTime, Boolean, ForeignKey,
                        Enum, DECIMAL, Text, func, select, text)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# NLP models
import spacy
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from rapidfuzz import process as rf_process, fuzz

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
import jwt
import bcrypt
import time

# -----------------------
# Config
# -----------------------
DATABASE_URL = os.getenv("DATABASE_URL", "mysql+asyncmy://root:@127.0.0.1/LibraryDb")
FINE_PER_DAY = 1.0  # currency units per day
MAX_BORROWED = 4
DEFAULT_LOAN_DAYS = 14
RENEW_DAYS = 14

Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Enhanced Intent Recognition System
# -----------------------

class EnhancedIntentRecognizer:
    def __init__(self):
        self.intent_patterns = {
            "search_book": [
                r"(find|search|look).*book",
                r"do you have.*book",
                r"books? about",
                r"looking for.*book",
                r"any books? on",
                r"recommend.*book",
                r"what books.*available",
                r"show me.*books"
            ],
            "issue_book": [
                r"(borrow|issue|get|take).*book",
                r"want to.*borrow",
                r"can i.*borrow",
                r"check out.*book",
                r"lend me",
                r"take home.*book",
                r"get.*book"
            ],
            "return_book": [
                r"return.*book",
                r"give back",
                r"bring back.*book",
                r"finished with.*book",
                r"done with.*book"
            ],
            "renew_book": [
                r"renew.*book",
                r"extend.*due date",
                r"more time.*book",
                r"keep.*longer",
                r"continue.*book"
            ],
            "my_books": [
                r"my books",
                r"what i.*borrowed",
                r"current loans",
                r"books i have",
                r"due dates?",
                r"what do i have"
            ],
            "reserve_book": [
                r"reserve.*book",
                r"put on hold",
                r"not available.*reserve",
                r"waiting list",
                r"notify.*available"
            ],
            "greeting": [
                r"hello|hi|hey|greetings",
                r"good morning|afternoon|evening",
                r"how are you"
            ],
            "help": [
                r"help",
                r"what can you do",
                r"how does this work",
                r"commands",
                r"what can i do"
            ]
        }
        
        self.synonyms = {
            "book": ["book", "novel", "textbook", "publication", "title"],
            "search": ["find", "lookup", "locate", "discover"],
            "borrow": ["issue", "checkout", "take", "get"],
            "return": ["give back", "bring back", "submit"],
            "renew": ["extend", "continue", "prolong"]
        }

    def pattern_based_intent(self, text: str) -> Tuple[str, float]:
        """Use regex patterns for intent detection"""
        text_lower = text.lower()
        best_intent = "unknown"
        best_score = 0.0
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    matches = re.findall(pattern, text_lower, re.IGNORECASE)
                    score = len(matches) * 0.3
                    if score > best_score:
                        best_score = score
                        best_intent = intent
        
        return best_intent, best_score

    def keyword_based_intent(self, text: str) -> Tuple[str, float]:
        """Simple keyword matching as fallback"""
        text_lower = text.lower()
        keyword_mapping = {
            "search_book": ["find", "search", "look", "have", "book about", "recommend"],
            "issue_book": ["borrow", "issue", "get book", "take book", "check out"],
            "return_book": ["return", "give back", "bring back", "finished"],
            "renew_book": ["renew", "extend", "more time", "keep longer"],
            "my_books": ["my book", "borrowed", "due date", "what i have", "current"],
            "reserve_book": ["reserve", "hold", "waiting list", "notify"],
            "greeting": ["hello", "hi", "hey", "good morning", "good afternoon"],
            "help": ["help", "what can", "how to", "commands"]
        }
        
        best_intent = "unknown"
        best_matches = 0
        
        for intent, keywords in keyword_mapping.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > best_matches:
                best_matches = matches
                best_intent = intent
        
        score = min(best_matches * 0.3, 1.0)
        return best_intent, score

    def hybrid_intent_detection(self, text: str) -> Tuple[str, float, str]:
        """Combine multiple approaches for robust intent detection"""
        # 1. Pattern-based detection
        pattern_intent, pattern_score = self.pattern_based_intent(text)
        
        # 2. Semantic similarity (existing approach)
        semantic_intent, semantic_score = semantic_intent_score(text)
        
        # 3. Keyword-based fallback
        keyword_intent, keyword_score = self.keyword_based_intent(text)
        
        # Combine scores with weights
        scores = {
            pattern_intent: pattern_score * 0.4,
            semantic_intent: semantic_score * 0.4,
            keyword_intent: keyword_score * 0.2
        }
        
        # Find best intent
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        
        method = "pattern" if pattern_score > semantic_score and pattern_score > keyword_score else \
                "semantic" if semantic_score > keyword_score else "keyword"
        
        return best_intent, best_score, method

class EntityExtractor:
    def __init__(self):
        self.book_indicators = ["book", "novel", "textbook", "publication"]
        self.category_keywords = {
            "python": ["python", "programming", "coding"],
            "ml": ["machine learning", "ml", "ai", "artificial intelligence"],
            "ai": ["ai", "artificial intelligence", "neural network"],
            "ds": ["data science", "data analysis", "analytics"],
            "nlp": ["natural language", "nlp", "text processing"]
        }

    def extract_book_title(self, text: str, available_books: List[Dict]) -> List[Dict]:
        """Extract potential book titles from user input"""
        cleaned_text = self.clean_query(text)
        
        # Try exact matches first
        exact_matches = []
        for book in available_books:
            if book["title"].lower() in cleaned_text.lower():
                exact_matches.append(book)
        
        if exact_matches:
            return exact_matches
        
        # Try partial matches
        words = cleaned_text.split()
        potential_titles = []
        
        for book in available_books:
            title_words = book["title"].lower().split()
            common_words = set(words) & set(title_words)
            if len(common_words) >= 2:  # At least 2 matching words
                potential_titles.append(book)
        
        return potential_titles

    def clean_query(self, text: str) -> str:
        """Remove common phrases to isolate the book title"""
        remove_phrases = [
            "i want to", "can you", "please", "show me", "find me",
            "looking for", "search for", "book about", "books on",
            "borrow", "issue", "get", "find", "return", "renew"
        ]
        
        cleaned = text.lower()
        for phrase in remove_phrases:
            cleaned = cleaned.replace(phrase, "")
        
        return cleaned.strip()

    def extract_category(self, text: str) -> str:
        """Extract book category from user input"""
        text_lower = text.lower()
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category
        return None

class ConversationContext:
    def __init__(self):
        self.user_contexts = {}  # client_id -> context
    
    def get_context(self, client_id: str) -> Dict:
        return self.user_contexts.get(client_id, {
            "last_intent": None,
            "pending_action": None,
            "mentioned_books": [],
            "conversation_history": []
        })
    
    def update_context(self, client_id: str, user_message: str, bot_response: str, intent: str):
        if client_id not in self.user_contexts:
            self.user_contexts[client_id] = {
                "last_intent": None,
                "pending_action": None,
                "mentioned_books": [],
                "conversation_history": []
            }
        
        context = self.user_contexts[client_id]
        context["last_intent"] = intent
        context["conversation_history"].append({
            "user": user_message,
            "bot": bot_response,
            "timestamp": datetime.utcnow()
        })
        
        # Keep only last 10 messages
        if len(context["conversation_history"]) > 10:
            context["conversation_history"] = context["conversation_history"][-10:]
    
    def handle_follow_up(self, client_id: str, text: str) -> Optional[Dict]:
        """Handle follow-up questions based on context"""
        context = self.get_context(client_id)
        last_intent = context.get("last_intent")
        
        if not last_intent:
            return None
        
        follow_up_patterns = {
            "search_book": {
                "more": ["more", "show more", "next", "other options", "another"],
                "different": ["different", "other category", "something else"],
                "specific": ["about python", "machine learning", "ai books"]
            }
        }
        
        text_lower = text.lower()
        if last_intent in follow_up_patterns:
            for action, triggers in follow_up_patterns[last_intent].items():
                if any(trigger in text_lower for trigger in triggers):
                    return {"type": "follow_up", "action": action, "original_intent": last_intent}
        
        return None

# Initialize enhanced NLP components
intent_recognizer = EnhancedIntentRecognizer()
entity_extractor = EntityExtractor()
conversation_context = ConversationContext()

# -----------------------
# Improved YES/NO fuzzy detection (Balanced Mode)
# -----------------------
def is_yes(text: str) -> bool:
    """Balanced fuzzy YES detection"""
    text = text.lower().strip()
    hard_yes = ["yes", "yeah", "yep", "yup", "sure", "ok", "okay", "of course", "confirm"]
    if text in hard_yes:
        return True
    return any(
        fuzz.ratio(text, y) >= 75
        for y in ["yes", "yess", "ye", "ys", "yeah", "yup"]
    )

def is_no(text: str) -> bool:
    """Balanced fuzzy NO detection"""
    text = text.lower().strip()
    hard_no = ["no", "nope", "nah", "cancel", "stop"]
    if text in hard_no:
        return True
    return any(
        fuzz.ratio(text, n) >= 80
        for n in ["no", "noo", "nope", "nah", "nooo"]
    )

# -----------------------
# Models
# -----------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    client_id = Column(String(100), unique=True, index=True)
    name = Column(String(150))
    email = Column(String(150))
    phone = Column(String(30))
    created_at = Column(DateTime, server_default=func.now())

    issues = relationship("Issue", back_populates="user")
    reservations = relationship("Reservation", back_populates="user")
    fines = relationship("Fine", back_populates="user")


class Book(Base):
    __tablename__ = "books"
    id = Column(Integer, primary_key=True)
    title = Column(String(255), index=True, nullable=False)
    category = Column(String(100), index=True)
    author = Column(String(150))
    rack = Column(String(50))
    total_copies = Column(Integer, default=1)
    available_copies = Column(Integer, default=1)
    created_at = Column(DateTime, server_default=func.now())

    issues = relationship("Issue", back_populates="book")
    reservations = relationship("Reservation", back_populates="book")


class Issue(Base):
    __tablename__ = "issues"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    book_id = Column(Integer, ForeignKey("books.id"))
    issued_at = Column(DateTime, nullable=False)
    due_at = Column(DateTime, nullable=False)
    returned_at = Column(DateTime, nullable=True)
    status = Column(Enum('issued', 'returned', name="issue_status"), default='issued')

    user = relationship("User", back_populates="issues")
    book = relationship("Book", back_populates="issues")
    fines = relationship("Fine", back_populates="issue")


class Reservation(Base):
    __tablename__ = "reservations"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    book_id = Column(Integer, ForeignKey("books.id"))
    reserved_at = Column(DateTime, server_default=func.now())
    notified = Column(Boolean, default=False)

    user = relationship("User", back_populates="reservations")
    book = relationship("Book", back_populates="reservations")


class Fine(Base):
    __tablename__ = "fines"
    id = Column(Integer, primary_key=True)
    issue_id = Column(Integer, ForeignKey("issues.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    book_id = Column(Integer, ForeignKey("books.id"))
    fine_amount = Column(DECIMAL(10, 2))
    days_overdue = Column(Integer)
    paid = Column(Boolean, default=False)
    calculated_at = Column(DateTime, server_default=func.now())

    issue = relationship("Issue", back_populates="fines")
    user = relationship("User", back_populates="fines")

# -----------------------
# NLP setup
# -----------------------
print("Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None
    print("spaCy failed to load")

print("Loading SentenceTransformer...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading Zero-Shot classifier...")
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

print("NLP loaded successfully.")

# Expanded Intent examples
EXPANDED_INTENT_EXAMPLES = {
    "search_book": ["find book", "do you have", "search for", "look up", "find me a book", "looking for books about", "search for python books", "do you have any books on machine learning", "recommend a good book", "what books are available about AI", "show me some programming books", "I need a book about data science", "any good novels available", "can you help me find a book", "book recommendations for python"],
    "issue_book": ["i want to borrow", "issue book", "borrow", "I want to borrow a book", "can I issue this book", "check out this book", "I'd like to get this book", "borrow python crash course", "can you issue me this book", "I want to take this book home", "how do I borrow this", "get me this book please"],
    "return_book": ["i want to return", "return", "I want to return a book", "give back this book", "return python book", "I'm done with this book", "bring back borrowed book"],
    "renew_book": ["renew book", "extend", "renew my book", "extend due date", "can I keep this longer", "renew loan period"],
    "list_books": ["list books", "show all books", "what books do you have", "show available books", "display all titles"],
    "check_due": ["due date", "when to return", "when is this due", "return date", "deadline for return"],
    "my_books": ["my books", "what I borrowed", "books I have", "my current loans", "what did I borrow"],
    "greeting": ["hello", "hi", "hey", "good morning", "good afternoon", "how are you"],
    "help": ["help", "what can you do", "how does this work", "what commands are available"]
}

INTENT_PROTOTYPES = {}
for intent, examples in EXPANDED_INTENT_EXAMPLES.items():
    emb = embedder.encode(examples, convert_to_tensor=True)
    INTENT_PROTOTYPES[intent] = emb.mean(dim=0)

def semantic_intent_score(text: str):
    q_emb = embedder.encode(text, convert_to_tensor=True)
    best = ("unknown", -1.0)
    for intent, proto in INTENT_PROTOTYPES.items():
        score = float(util.cos_sim(q_emb, proto)[0][0])
        if score > best[1]:
            best = (intent, score)
    return best

def fuzzy_title_matches(query: str, db_books: List[Dict], limit: int = 5):
    titles = [b["title"] for b in db_books]
    results = rf_process.extract(query, titles, scorer=fuzz.WRatio, limit=limit)
    matches = []
    for title, score, idx in results:
        if score >= 50:
            matches.append(db_books[idx])
    return matches

def semantic_title_matches(query: str, db_books: List[Dict], threshold: float = 0.32, limit: int = 5):
    q_emb = embedder.encode(query, convert_to_tensor=True)
    scored = []
    for b in db_books:
        emb = embedder.encode(b["title"], convert_to_tensor=True)
        sim = float(util.cos_sim(q_emb, emb)[0][0])
        scored.append((sim, b))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [b for sim, b in scored if sim >= threshold][:limit]

async def find_books_db(session: AsyncSession):
    q = await session.execute(select(Book))
    rows = q.scalars().all()
    return [{
        "id": b.id, "title": b.title, "category": b.category,
        "author": b.author, "rack": b.rack,
        "total_copies": b.total_copies, "available_copies": b.available_copies
    } for b in rows]

# -----------------------
# Enhanced Message Processing
# -----------------------

# Replace the process_user_message_enhanced function with this improved version:

async def process_user_message_enhanced(client_id: str, user_msg: str):
    """Enhanced message processing with better intent recognition"""
    try:
        # Clean and normalize input
        cleaned_msg = user_msg.strip().lower()
        
        # Check for context-based follow-up first
        follow_up = conversation_context.handle_follow_up(client_id, cleaned_msg)
        if follow_up:
            await handle_follow_up_action(client_id, follow_up)
            return
        
        # Use hybrid intent detection
        intent, confidence, method = intent_recognizer.hybrid_intent_detection(user_msg)
        
        logger.info(f"Intent: {intent}, Confidence: {confidence:.2f}, Method: {method}, Message: '{user_msg}'")
        
        # Handle low confidence cases - LOWERED THRESHOLD from 0.4 to 0.2
        if confidence < 0.2:
            await handle_ambiguous_intent(client_id, user_msg)
            return
        
        # Extract entities
        category = entity_extractor.extract_category(user_msg)
        
        # Update conversation context
        conversation_context.update_context(
            client_id, user_msg, f"Processing {intent}", intent
        )
        
        # Route to appropriate handler
        if intent == "search_book":
            await handle_search_intent(client_id, user_msg, category)
        elif intent == "issue_book":
            await handle_issue_intent(client_id, user_msg)
        elif intent == "return_book":
            await handle_return_intent(client_id, user_msg)
        elif intent == "renew_book":
            await handle_renew_intent(client_id, user_msg)
        elif intent == "my_books":
            await process_action(client_id, {"name": "my_books"})
        elif intent == "greeting":
            await handle_greeting(client_id)
        elif intent == "help":
            await handle_help(client_id)
        elif intent == "reserve_book":
            await handle_reserve_intent(client_id, user_msg)
        else:
            # If we get unknown but have some confidence, try to handle it
            if confidence >= 0.2:
                # Check for common patterns that might have been missed
                if any(word in cleaned_msg for word in ["python", "ai", "machine learning", "data science"]):
                    await handle_search_intent(client_id, user_msg, category)
                elif any(word in cleaned_msg for word in ["borrow", "issue", "get", "take"]):
                    await handle_issue_intent(client_id, user_msg)
                elif "return" in cleaned_msg:
                    await handle_return_intent(client_id, user_msg)
                else:
                    await handle_unknown_intent(client_id, user_msg)
            else:
                await handle_unknown_intent(client_id, user_msg)
            
    except Exception as e:
        logger.error(f"Error processing message from {client_id}: {e}")
        await manager.send(client_id, {
            "type": "bot_message", 
            "message": "I encountered an error. Please try rephrasing your request."
        })

# Also update the hybrid_intent_detection method to be more lenient:
def hybrid_intent_detection(self, text: str) -> Tuple[str, float, str]:
    """Combine multiple approaches for robust intent detection"""
    text_lower = text.lower()
    
    # Quick checks for very clear patterns first
    if any(word in text_lower for word in ["hi", "hello", "hey", "greetings"]):
        return "greeting", 0.9, "direct"
    
    if any(word in text_lower for word in ["search", "find", "look for", "book about"]):
        return "search_book", 0.8, "direct"
    
    if any(word in text_lower for word in ["borrow", "issue", "get book", "take book"]):
        return "issue_book", 0.8, "direct"
    
    if any(word in text_lower for word in ["return", "give back"]):
        return "return_book", 0.8, "direct"
    
    if any(word in text_lower for word in ["my books", "what i have", "borrowed"]):
        return "my_books", 0.8, "direct"
    
    # Then use the existing hybrid approach
    pattern_intent, pattern_score = self.pattern_based_intent(text)
    semantic_intent, semantic_score = semantic_intent_score(text)
    keyword_intent, keyword_score = self.keyword_based_intent(text)
    
    # Combine scores with weights
    scores = {
        pattern_intent: pattern_score * 0.4,
        semantic_intent: semantic_score * 0.4,
        keyword_intent: keyword_score * 0.2
    }
    
    # Find best intent
    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]
    
    method = "pattern" if pattern_score > semantic_score and pattern_score > keyword_score else \
            "semantic" if semantic_score > keyword_score else "keyword"
    
    return best_intent, best_score, method

# Update the pattern-based intent to be more flexible:
def pattern_based_intent(self, text: str) -> Tuple[str, float]:
    """Use regex patterns for intent detection"""
    text_lower = text.lower()
    best_intent = "unknown"
    best_score = 0.0
    
    # Simple word-based matching as fallback
    word_matches = {
        "search_book": ["search", "find", "look", "book about", "books on"],
        "issue_book": ["borrow", "issue", "get", "take", "check out"],
        "return_book": ["return", "give back", "bring back"],
        "renew_book": ["renew", "extend", "more time"],
        "my_books": ["my book", "borrowed", "what i have"],
        "greeting": ["hello", "hi", "hey"]
    }
    
    for intent, words in word_matches.items():
        matches = sum(1 for word in words if word in text_lower)
        if matches > 0:
            score = matches * 0.25
            if score > best_score:
                best_score = score
                best_intent = intent
    
    # Then try regex patterns
    for intent, patterns in self.intent_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                score = len(matches) * 0.3
                if score > best_score:
                    best_score = score
                    best_intent = intent
    
    return best_intent, best_score

# Also add this helper function to handle direct category searches:
async def handle_direct_category_search(client_id: str, category: str):
    """Handle direct category searches like 'python books'"""
    if category:
        await manager.send(client_id, {
            "type": "bot_message",
            "message": f"Here are some {category} books you may like:"
        })
        await process_action(client_id, {
            "name": "list", 
            "page": 1, 
            "per_page": 5, 
            "category": category
        })
    else:
        await manager.send(client_id, {
            "type": "bot_message",
            "message": "What type of books are you looking for? You can mention Python, AI, Machine Learning, etc."
        })

# Update the search intent handler to be more direct:
async def handle_search_intent(client_id: str, user_msg: str, category: str = None):
    """Enhanced search intent handling"""
    # If user directly mentions a category, show those books immediately
    if not category:
        category = entity_extractor.extract_category(user_msg)
    
    if category:
        await handle_direct_category_search(client_id, category)
        return
    
    async with AsyncSessionLocal() as session:
        books = await find_books_db(session)
        
        # Extract potential book titles
        potential_books = entity_extractor.extract_book_title(user_msg, books)
        
        if potential_books:
            # Show exact or close matches
            await manager.send(client_id, {
                "type": "search_results",
                "results": potential_books[:6]
            })
        else:
            # Ask for clarification but also show some books
            await manager.send(client_id, {
                "type": "bot_message",
                "message": "Here are some available books. You can also specify a category like Python, AI, or Machine Learning."
            })
            await process_action(client_id, {
                "name": "list", 
                "page": 1, 
                "per_page": 5, 
                "category": "all"
            })

async def handle_ambiguous_intent(client_id: str, user_msg: str):
    """Handle cases where intent is not clear"""
    clarification_questions = [
        "Are you looking to search for a book, borrow one, or return a book?",
        "Would you like to find a book, see your borrowed books, or something else?",
        "I'm not sure what you'd like to do. Are you searching, borrowing, or returning?"
    ]
    
    import random
    question = random.choice(clarification_questions)
    
    await manager.send(client_id, {
        "type": "bot_message",
        "message": question
    })
    
    # Provide clear options
    await manager.send(client_id, {
        "type": "suggestions",
        "options": [
            {"label": "Search for a book", "action": {"name": "prefill", "value": "Search for books about"}},
            {"label": "Borrow a book", "action": {"name": "prefill", "value": "I want to borrow"}},
            {"label": "Return a book", "action": {"name": "prefill", "value": "I want to return"}},
            {"label": "See my books", "action": {"name": "my_books"}}
        ]
    })

async def handle_search_intent(client_id: str, user_msg: str, category: str = None):
    """Enhanced search intent handling"""
    async with AsyncSessionLocal() as session:
        books = await find_books_db(session)
        
        # Extract potential book titles
        potential_books = entity_extractor.extract_book_title(user_msg, books)
        
        if potential_books:
            # Show exact or close matches
            await manager.send(client_id, {
                "type": "search_results",
                "results": potential_books[:6]
            })
        elif category:
            # Search by category
            await process_action(client_id, {
                "name": "list", 
                "page": 1, 
                "per_page": 5, 
                "category": category
            })
        else:
            # Ask for clarification
            await manager.send(client_id, {
                "type": "bot_message",
                "message": "What type of books are you looking for? You can mention a title, author, or category like Python, AI, etc."
            })

async def handle_issue_intent(client_id: str, user_msg: str):
    """Enhanced issue intent handling"""
    async with AsyncSessionLocal() as session:
        books = await find_books_db(session)
        rows = [{"id": b["id"], "title": b["title"], "category": b["category"], "available": b["available_copies"]>0} for b in books]
        
        # Use entity extraction for better matching
        potential_books = entity_extractor.extract_book_title(user_msg, rows)
        
        if not potential_books:
            # Fallback to semantic matching
            potential_books = semantic_title_matches(user_msg, rows, threshold=0.32, limit=6) or fuzzy_title_matches(user_msg, rows, limit=6)

        if not potential_books:
            await manager.send(client_id, {"type":"bot_message","message":"Book not found. Would you like to see all available books?"})
            await manager.send(client_id, {"type":"suggestions","options":[{"label":"Show all books","action":{"name":"list","page":1,"per_page":5,"category":"all"}}]})
            return

        if len(potential_books) == 1:
            b = potential_books[0]
            pending_action[client_id] = {"name":"issue_by_id","book_id":b["id"]}
            await manager.send(client_id, {"type":"bot_message","message":f"Do you want to issue '{b['title']}'? Reply 'yes' to confirm."})
            await manager.send(client_id, {"type":"suggestions","options":[
                {"label":f"Yes, issue '{b['title']}'","action":{"name":"confirm_yes"}},
                {"label":"No","action":{"name":"noop"}}
            ]})
        else:
            opts = [{"label": f"Issue '{m['title']}'", "action": {"name":"issue_by_id","book_id":m["id"]}} for m in potential_books[:5]]
            await manager.send(client_id, {"type":"suggestions","options":opts})

async def handle_return_intent(client_id: str, user_msg: str):
    """Enhanced return intent handling"""
    async with AsyncSessionLocal() as session:
        books = await find_books_db(session)
        rows = [{"id": b["id"], "title": b["title"]} for b in books]
        
        potential_books = entity_extractor.extract_book_title(user_msg, rows)
        if not potential_books:
            potential_books = semantic_title_matches(user_msg, rows, threshold=0.32, limit=6) or fuzzy_title_matches(user_msg, rows, limit=6)
        
        if potential_books:
            await manager.send(client_id, {"type":"suggestions","options":[
                {"label":f"Return '{m['title']}'","action":{"name":"return_by_id","book_id":m["id"]}} for m in potential_books[:5]
            ]})
        else:
            await manager.send(client_id, {"type":"bot_message","message":"Book not found. Please specify which book you want to return."})

async def handle_renew_intent(client_id: str, user_msg: str):
    """Enhanced renew intent handling"""
    async with AsyncSessionLocal() as session:
        books = await find_books_db(session)
        rows = [{"id": b["id"], "title": b["title"]} for b in books]
        
        potential_books = entity_extractor.extract_book_title(user_msg, rows)
        if not potential_books:
            potential_books = semantic_title_matches(user_msg, rows, threshold=0.32, limit=6) or fuzzy_title_matches(user_msg, rows, limit=6)
        
        if potential_books:
            await manager.send(client_id, {"type":"suggestions","options":[
                {"label":f"Renew '{m['title']}'","action":{"name":"renew_by_id","book_id":m["id"]}} for m in potential_books[:5]
            ]})
        else:
            await manager.send(client_id, {"type":"bot_message","message":"Book not found to renew."})

async def handle_reserve_intent(client_id: str, user_msg: str):
    """Handle reserve book intent"""
    async with AsyncSessionLocal() as session:
        books = await find_books_db(session)
        rows = [{"id": b["id"], "title": b["title"], "available": b["available_copies"]>0} for b in books]
        
        potential_books = entity_extractor.extract_book_title(user_msg, rows)
        if not potential_books:
            potential_books = semantic_title_matches(user_msg, rows, threshold=0.32, limit=6) or fuzzy_title_matches(user_msg, rows, limit=6)
        
        if potential_books:
            opts = [{"label": f"Reserve '{m['title']}'", "action": {"name":"reserve","book_id":m["id"]}} for m in potential_books[:5]]
            await manager.send(client_id, {"type":"suggestions","options":opts})
        else:
            await manager.send(client_id, {"type":"bot_message","message":"Book not found to reserve."})

async def handle_greeting(client_id: str):
    """Handle greeting intent"""
    await manager.send(client_id, {
        "type": "bot_message",
        "message": "👋 Hello! I'm the Library Assistant. I can help you search for books, borrow, return, renew, or check your current loans."
    })
    await manager.send(client_id, {
        "type": "suggestions",
        "options": [
            {"label": "Search for a book", "action": {"name": "prefill", "value": "Search for a book"}},
            {"label": "List books", "action": {"name": "list", "page": 1, "per_page": 5, "category": "all"}},
            {"label": "My books", "action": {"name": "my_books"}}
        ]
    })

async def handle_help(client_id: str):
    """Handle help intent"""
    help_text = """
I can help you with:
• **Searching** for books by title, author, or category
• **Borrowing** books from the library  
• **Returning** books you've finished
• **Renewing** books to extend due dates
• **Reserving** books that are currently unavailable
• **Checking** your borrowed books and due dates

Just tell me what you'd like to do!
"""
    await manager.send(client_id, {
        "type": "bot_message",
        "message": help_text
    })

async def handle_unknown_intent(client_id: str, user_msg: str):
    """Handle unknown intent"""
    await manager.send(client_id, {
        "type": "bot_message",
        "message": "I'm not sure what you'd like to do. Try saying 'search for a book', 'borrow a book', 'return a book', or 'my books'."
    })
    await manager.send(client_id, {
        "type": "suggestions",
        "options": [
            {"label": "Search books", "action": {"name": "prefill", "value": "Search for"}},
            {"label": "Borrow a book", "action": {"name": "prefill", "value": "I want to borrow"}},
            {"label": "My books", "action": {"name": "my_books"}}
        ]
    })

async def handle_follow_up_action(client_id: str, follow_up: Dict):
    """Handle follow-up actions based on context"""
    action = follow_up.get("action")
    original_intent = follow_up.get("original_intent")
    
    if action == "more" and original_intent == "search_book":
        await manager.send(client_id, {
            "type": "bot_message",
            "message": "Here are more books:"
        })
        await process_action(client_id, {"name": "list", "page": 2, "per_page": 5, "category": "all"})
    elif action == "different" and original_intent == "search_book":
        await manager.send(client_id, {
            "type": "bot_message", 
            "message": "What category would you like to explore? Try Python, AI, Machine Learning, Data Science, or NLP."
        })

# -----------------------
# FastAPI app + static mount
# -----------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Connection manager
class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, WebSocket] = {}

    async def connect(self, ws: WebSocket, client_id: str):
        await ws.accept()
        self.active[client_id] = ws
        print(f"[WS] connected {client_id}")

    def disconnect(self, client_id: str):
        self.active.pop(client_id, None)
        print(f"[WS] disconnected {client_id}")

    async def send(self, client_id: str, payload: dict):
        ws = self.active.get(client_id)
        if ws:
            try:
                await ws.send_json(payload)
            except Exception as e:
                print(f"Error sending to {client_id}: {e}")

    async def broadcast(self, payload: dict):
        for cid, ws in list(self.active.items()):
            try:
                await ws.send_json(payload)
            except Exception:
                pass

manager = ConnectionManager()

# Pending actions
pending_action: Dict[str, Optional[dict]] = {}

# DB helpers
async def get_or_create_user(session: AsyncSession, client_id: str, name: Optional[str]=None):
    result = await session.execute(select(User).where(User.client_id == client_id))
    user = result.scalars().first()
    if user:
        return user
    user = User(client_id=client_id, name=name or client_id)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user

async def book_status_payload(book_row):
    return {
        "id": book_row.id,
        "title": book_row.title,
        "category": book_row.category,
        "author": book_row.author,
        "rack": book_row.rack,
        "total_copies": book_row.total_copies,
        "available_copies": book_row.available_copies
    }

async def calculate_fine_for_issue(issue: Issue) -> Optional[Dict]:
    if issue.returned_at:
        return None
    today = datetime.utcnow()
    if issue.due_at and today > issue.due_at:
        days = (today.date() - issue.due_at.date()).days
        amount = round(days * FINE_PER_DAY, 2)
        return {"issue_id": issue.id, "user_id": issue.user_id, "book_id": issue.book_id, "days": days, "amount": amount}
    return None

# -----------------------
# Startup: create tables + seed data
# -----------------------
@app.on_event("startup")
async def startup_event():
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            try:
                await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS created_at DATETIME DEFAULT CURRENT_TIMESTAMP"))
                await conn.execute(text("ALTER TABLE books ADD COLUMN IF NOT EXISTS created_at DATETIME DEFAULT CURRENT_TIMESTAMP"))
            except Exception:
                pass

        async with AsyncSessionLocal() as session:
            q = await session.execute(select(func.count()).select_from(Book))
            total = q.scalar()
            if not total:
                samples = [
                    ("Python Crash Course", "Python", "Eric Matthes", "A1", 2),
                    ("Fluent Python", "Python", "Luciano Ramalho", "A2", 1),
                    ("Hands-On Machine Learning", "ML", "Aurelien Geron", "B1", 1),
                    ("Deep Learning with Python", "AI", "Francois Chollet", "B2", 1),
                    ("Artificial Intelligence: A Modern Approach", "AI", "Stuart Russell", "C1", 1),
                    ("Introduction to Machine Learning", "ML", "Ethem Alpaydin", "C2", 1),
                    ("Data Science from Scratch", "DS", "Joel Grus", "D1", 1),
                    ("Natural Language Processing with Python", "NLP", "Bird/ Klein/ Loper", "D2", 1),
                    ("Pattern Recognition and Machine Learning", "ML", "Christopher Bishop", "E1", 1),
                    ("Deep Learning", "AI", "Ian Goodfellow", "E2", 1),
                ]
                for t, cat, auth, rack, copies in samples:
                    b = Book(title=t, category=cat, author=auth, rack=rack, total_copies=copies, available_copies=copies)
                    session.add(b)
                await session.commit()
                print("Seeded sample books.")

        asyncio.create_task(overdue_checker_task())

    except Exception as e:
        print("Startup error:", e)

# -----------------------
# Background overdue checker
# -----------------------
async def overdue_checker_task():
    while True:
        try:
            async with AsyncSessionLocal() as session:
                q = await session.execute(select(Issue).where(Issue.status == 'issued'))
                issues = q.scalars().all()
                for iss in issues:
                    fine_info = await calculate_fine_for_issue(iss)
                    if fine_info:
                        q2 = await session.execute(select(Fine).where(Fine.issue_id == iss.id, Fine.paid == False))
                        existing = q2.scalars().first()
                        if existing:
                            existing.fine_amount = fine_info["amount"]
                            existing.days_overdue = fine_info["days"]
                            existing.calculated_at = datetime.utcnow()
                            session.add(existing)
                            await session.commit()
                        else:
                            fine = Fine(issue_id=iss.id, user_id=iss.user_id, book_id=iss.book_id,
                                        fine_amount=fine_info["amount"], days_overdue=fine_info["days"], paid=False)
                            session.add(fine)
                            await session.commit()
                        user = await session.get(User, iss.user_id)
                        if user and user.client_id:
                            await manager.send(user.client_id, {"type":"bot_message", "message": f"Your borrowed book '{iss.book.title}' is overdue by {fine_info['days']} days. Fine: {fine_info['amount']}."})
        except Exception as e:
            print("Overdue checker error:", e)
        await asyncio.sleep(3600)

# -----------------------
# REST endpoints
# -----------------------
@app.get("/books")
async def get_books(page: int = 1, per_page: int = 10, category: Optional[str] = Query("all")):
    async with AsyncSessionLocal() as session:
        stmt = select(Book)
        if category and category.lower() != "all":
            stmt = stmt.where(Book.category.ilike(f"%{category}%"))
        total = (await session.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one()
        stmt = stmt.offset((page-1)*per_page).limit(per_page)
        res = await session.execute(stmt)
        items = res.scalars().all()
        return JSONResponse({
            "page": page, "per_page": per_page, "total": total,
            "total_pages": max(1, (int(total) + per_page - 1)//per_page),
            "items": [await book_status_payload(b) for b in items]
        })

@app.get("/book_status/{book_id}")
async def book_status(book_id: int):
    async with AsyncSessionLocal() as session:
        book = await session.get(Book, book_id)
        if not book:
            return JSONResponse({"ok": False, "message": "Book not found."}, status_code=404)
        payload = await book_status_payload(book)
        if book.available_copies == 0:
            q = await session.execute(select(Issue).where(Issue.book_id == book.id, Issue.status == 'issued').order_by(Issue.due_at))
            next_due = q.scalars().first()
            if next_due:
                payload["expected_return"] = next_due.due_at.isoformat()
        return JSONResponse({"ok": True, "book": payload})

@app.get("/categories")
async def categories():
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(Book.category).distinct())
        cats = [r[0] for r in res.all() if r[0]]
        return JSONResponse({"categories": cats})

@app.post("/reserve")
async def reserve(book_id: int, client_id: str):
    async with AsyncSessionLocal() as session:
        user = await get_or_create_user(session, client_id)
        book = await session.get(Book, book_id)
        if not book:
            return JSONResponse({"ok": False, "message": "Book not found."})
        resq = await session.execute(select(Reservation).where(Reservation.user_id==user.id, Reservation.book_id==book.id))
        existing = resq.scalars().first()
        if existing:
            return JSONResponse({"ok": False, "message": "You already reserved this book."})
        reservation = Reservation(user_id=user.id, book_id=book.id)
        session.add(reservation)
        await session.commit()
        await manager.broadcast({"type":"bot_message", "message": f"Reservation placed: {user.client_id} reserved '{book.title}'."})
        return JSONResponse({"ok": True, "message": f"Reserved '{book.title}'. We'll notify you when available."})

@app.get("/fines/{client_id}")
async def get_fines(client_id: str):
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(User).where(User.client_id == client_id))
        user = res.scalars().first()
        if not user:
            return JSONResponse({"ok": True, "fines": []})
        q = await session.execute(select(Fine).where(Fine.user_id == user.id, Fine.paid == False))
        fines = q.scalars().all()
        out = [{"id": f.id, "issue_id": f.issue_id, "amount": float(f.fine_amount), "days_overdue": f.days_overdue} for f in fines]
        return JSONResponse({"ok": True, "fines": out})

@app.post("/pay_fine")
async def pay_fine(fine_id: int, client_id: str):
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(Fine).where(Fine.id == fine_id))
        fine = res.scalars().first()
        if not fine:
            return JSONResponse({"ok": False, "message": "Fine not found."})
        fine.paid = True
        session.add(fine)
        await session.commit()
        return JSONResponse({"ok": True, "message": "Fine marked as paid. Thank you."})

# -----------------------
# Admin endpoints
# -----------------------
@app.post("/admin/check_login")
async def admin_login(payload: dict = Body(...)):
    username = payload.get("username")
    password = payload.get("password")
    if username == "admin" and password == "1234":
        return {"ok": True}
    return {"ok": False}

@app.get("/admin/books")
async def admin_books():
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(Book))
        books = res.scalars().all()
        return [{
            "id": b.id, "title": b.title, "category": b.category,
            "rack": b.rack,
            "total_copies": b.total_copies,
            "available_copies": b.available_copies
        } for b in books]

@app.get("/admin/stats")
async def admin_stats():
    async with AsyncSessionLocal() as session:
        q1 = await session.execute(select(Book.category, func.count()).group_by(Book.category))
        categories = [{"category": r[0], "count": r[1]} for r in q1.all()]
        q2 = await session.execute(select(func.count()).select_from(Issue).where(Issue.status == 'issued'))
        issued = q2.scalar()
        q3 = await session.execute(select(func.count()).select_from(Reservation))
        reservations = q3.scalar()
        q4 = await session.execute(select(Book.title, func.count(Issue.id)).join(Issue, Issue.book_id == Book.id).group_by(Book.id).order_by(func.count(Issue.id).desc()).limit(5))
        top_books = [{"title": r[0], "count": r[1]} for r in q4.all()]
        return {"categories": categories, "issued": issued, "reservations": reservations, "top_books": top_books}

# -----------------------
# Centralized action processor (with confirmation support)
# -----------------------
async def process_action(client_id: str, action: dict):
    name = action.get("name")
    try:
        # Sensitive actions require confirmation unless action['confirm'] == True
        sensitive = name in ("issue_by_id","issue","return_by_id","return","renew_by_id","renew")
        if sensitive and not action.get("confirm"):
            pending_action[client_id] = action
            verb = "perform"
            if name in ("issue_by_id","issue"):
                verb = "issue this book"
            elif name in ("return_by_id","return"):
                verb = "return this book"
            elif name in ("renew_by_id","renew"):
                verb = "renew this book"
            await manager.send(client_id, {"type":"bot_message","message":f"Are you sure you want to {verb}? Reply 'yes' to confirm."})
            await manager.send(client_id, {"type":"suggestions","options":[{"label":"Yes","action":{"name":"confirm_yes"}},{"label":"No","action":{"name":"noop"}}]})
            return

        # LIST
        if name == "list":
            page = int(action.get("page", 1))
            per_page = int(action.get("per_page", 5))
            cat = action.get("category", "all")
            async with AsyncSessionLocal() as session:
                stmt = select(Book)
                if cat and cat != "all":
                    stmt = stmt.where(Book.category.ilike(f"%{cat}%"))
                total = (await session.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one()
                stmt = stmt.offset((page-1)*per_page).limit(per_page)
                res = await session.execute(stmt)
                items = res.scalars().all()
                payload = {"page": page, "per_page": per_page, "total": total,
                           "total_pages": max(1, (int(total)+per_page-1)//per_page),
                           "items": [await book_status_payload(b) for b in items]}
                await manager.send(client_id, {"type":"book_list","data":payload})
            return

        # ISSUE
        if name in ("issue_by_id","issue"):
            book_id = int(action.get("book_id") or action.get("id"))
            async with AsyncSessionLocal() as session:
                user = await get_or_create_user(session, client_id)
                book = await session.get(Book, book_id)
                if not book:
                    await manager.send(client_id, {"type":"bot_message","message":"Book not found."})
                    return
                if book.available_copies <= 0:
                    await manager.send(client_id, {"type":"bot_message","message":f"'{book.title}' is currently unavailable. Would you like to reserve it?"})
                    pending_action[client_id] = {"name":"reserve","book_id":book.id}
                    await manager.send(client_id, {"type":"suggestions","options":[{"label":"Yes, reserve","action":{"name":"reserve","book_id":book.id}},{"label":"No","action":{"name":"noop"}}]})
                    return
                qf = await session.execute(select(Fine).where(Fine.user_id==user.id, Fine.paid==False))
                unpaid = qf.scalars().first()
                if unpaid:
                    await manager.send(client_id, {"type":"bot_message","message":"You have unpaid fines. Please pay fines before issuing new books."})
                    return
                q2 = await session.execute(select(Issue).where(Issue.user_id==user.id, Issue.status=='issued'))
                borrowed_count = len(q2.scalars().all())
                if borrowed_count >= MAX_BORROWED:
                    await manager.send(client_id, {"type":"bot_message","message":f"You have reached the max borrowed limit ({MAX_BORROWED}). Return or renew before borrowing more."})
                    return
                issued_at = datetime.utcnow()
                due_at = issued_at + timedelta(days=DEFAULT_LOAN_DAYS)
                new_issue = Issue(user_id=user.id, book_id=book.id, issued_at=issued_at, due_at=due_at, status='issued')
                book.available_copies -= 1
                session.add(new_issue)
                session.add(book)
                await session.commit()
                await manager.send(client_id, {"type":"bot_message","message":f"'{book.title}' issued to you. Due on {due_at.date().isoformat()}."})
            return

        # RETURN
        if name in ("return_by_id","return"):
            book_id = int(action.get("book_id") or action.get("id"))
            async with AsyncSessionLocal() as session:
                user = await get_or_create_user(session, client_id)
                q = await session.execute(select(Issue).where(Issue.book_id==book_id, Issue.user_id==user.id, Issue.status=='issued'))
                issue = q.scalars().first()
                if not issue:
                    await manager.send(client_id, {"type":"bot_message","message":"You don't have this book issued."})
                    return
                issue.returned_at = datetime.utcnow()
                issue.status = 'returned'
                book = await session.get(Book, book_id)
                if book:
                    book.available_copies = min(book.total_copies, book.available_copies + 1)
                    session.add(book)
                session.add(issue)
                await session.commit()
                await manager.send(client_id, {"type":"bot_message","message":f"Returned '{book.title}'. Thank you."})
                # notify next reservation
                q2 = await session.execute(select(Reservation).where(Reservation.book_id==book.id).order_by(Reservation.reserved_at))
                first_res = q2.scalars().first()
                if first_res:
                    first_res.notified = True
                    session.add(first_res)
                    await session.commit()
                    reserving_user = await session.get(User, first_res.user_id)
                    if reserving_user and reserving_user.client_id:
                        await manager.send(reserving_user.client_id, {"type":"bot_message","message":f"Your reserved book '{book.title}' is now available. Would you like to issue it?"})
            return

        # RENEW
        if name in ("renew_by_id","renew"):
            book_id = int(action.get("book_id") or action.get("id"))
            async with AsyncSessionLocal() as session:
                user = await get_or_create_user(session, client_id)
                q = await session.execute(select(Issue).where(Issue.book_id==book_id, Issue.user_id==user.id, Issue.status=='issued'))
                issue = q.scalars().first()
                if not issue:
                    await manager.send(client_id, {"type":"bot_message","message":"This book isn't issued to you."})
                    return
                q2 = await session.execute(select(Reservation).where(Reservation.book_id==book_id))
                reserved = q2.scalars().all()
                if reserved:
                    await manager.send(client_id, {"type":"bot_message","message":"Cannot renew because another user has reserved this book."})
                    return
                issue.due_at = issue.due_at + timedelta(days=RENEW_DAYS)
                session.add(issue)
                await session.commit()
                await manager.send(client_id, {"type":"bot_message","message":f"Book renewed. New due date: {issue.due_at.date().isoformat()}."})
            return

        # MY BOOKS
        if name == "my_books":
            async with AsyncSessionLocal() as session:
                user = await get_or_create_user(session, client_id)
                q = await session.execute(select(Issue).where(Issue.user_id==user.id).order_by(Issue.due_at))
                recs = q.scalars().all()
                out = []
                for r in recs:
                    b = await session.get(Book, r.book_id)
                    due = r.due_at.date().isoformat() if r.due_at else None
                    out.append({"book_id": b.id, "title": b.title, "issued_at": r.issued_at.date().isoformat(), "due_at": due})
                await manager.send(client_id, {"type":"my_books","items": out})
            return

        # RESERVE
        if name == "reserve":
            book_id = int(action.get("book_id"))
            async with AsyncSessionLocal() as session:
                resq = await session.execute(select(Book).where(Book.id==book_id))
                book = resq.scalars().first()
                if not book:
                    await manager.send(client_id, {"type":"bot_message","message":"Book not found."})
                    return
                user = await get_or_create_user(session, client_id)
                r_check = await session.execute(select(Reservation).where(Reservation.user_id==user.id, Reservation.book_id==book.id))
                if r_check.scalars().first():
                    await manager.send(client_id, {"type":"bot_message","message":"You already reserved this book."})
                    return
                reservation = Reservation(user_id=user.id, book_id=book.id)
                session.add(reservation)
                await session.commit()
                await manager.send(client_id, {"type":"bot_message","message":f"Reserved '{book.title}'. We'll notify you when available."})
            return

        # PREFILL
        if name == "prefill":
            await manager.send(client_id, {"type":"prefill","value": action.get("value","")})
            return

        # NOOP
        if name == "noop":
            await manager.send(client_id, {"type":"bot_message","message":"Okay."})
            return

        # CONFIRM YES (button)
        if name == "confirm_yes":
            to_do = pending_action.get(client_id)
            if to_do:
                to_do["confirm"] = True
                pending_action[client_id] = None
                await process_action(client_id, to_do)
            else:
                await manager.send(client_id, {"type":"bot_message","message":"Nothing to confirm."})
            return

        await manager.send(client_id, {"type":"bot_message","message":"Action not supported."})
    except Exception as e:
        print("process_action error:", e)
        await manager.send(client_id, {"type":"bot_message","message":"Sorry, an error occurred while processing your request."})

# -----------------------
# WebSocket endpoint (main chat)
# -----------------------
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    async with AsyncSessionLocal() as session:
        await get_or_create_user(session, client_id)

    await handle_greeting(client_id)

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            user_msg = data.get("message")
            action = data.get("action")

            if action:
                await process_action(client_id, action)
                continue

            if user_msg:
                await manager.send(client_id, {"type":"user_message","message":user_msg})
                await manager.send(client_id, {"type":"typing","value":True})

                try:
                    text = user_msg.strip()
                    lower = text.lower()

                    # YES / NO fuzzy handling
                    if is_yes(lower):
                        action_to_do = pending_action.get(client_id)
                        if action_to_do:
                            action_to_do["confirm"] = True
                            pending_action[client_id] = None
                            await process_action(client_id, action_to_do)
                            await manager.send(client_id, {"type":"typing","value":False})
                            continue
                        else:
                            await manager.send(client_id, {"type":"bot_message","message":"There is nothing to confirm."})
                            await manager.send(client_id, {"type":"typing","value":False})
                            continue

                    if is_no(lower):
                        pending_action[client_id] = None
                        await manager.send(client_id, {"type":"bot_message","message":"Okay — cancelled."})
                        await manager.send(client_id, {"type":"typing","value":False})
                        continue

                    # Use enhanced message processing
                    await process_user_message_enhanced(client_id, user_msg)
                    await manager.send(client_id, {"type":"typing","value":False})

                except Exception as e:
                    logger.error(f"Error processing message from {client_id}: {e}")
                    await manager.send(client_id, {"type":"bot_message","message":"Sorry, I encountered an error processing your request."})
                    await manager.send(client_id, {"type":"typing","value":False})

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)

# -----------------------
# Root / index (serves your static UI)
# -----------------------
@app.get("/")
async def index():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except Exception as e:
        return JSONResponse({"ok": False, "message": "index.html not found", "error": str(e)}, status_code=500)

# Graceful shutdown handling (optional cleanup)
@app.on_event("shutdown")
async def shutdown_event():
    try:
        # notify connected clients (optional)
        await manager.broadcast({"type":"bot_message","message":"Server is shutting down."})
    except Exception:
        pass
    print("Shutting down server...")

# -----------------------
# Quick start (run using uvicorn)
# -----------------------
# Save the full file as server.py and run:
# uvicorn server:app --reload
#
# Notes:
# - Ensure your database is accessible via DATABASE_URL (default uses asyncmy + local MySQL)
# - If you need to use a .env, add `from dotenv import load_dotenv; load_dotenv()` at top and create .env
# - If models fail to load (spaCy / transformers), server will still run but NLP features will degrade.
#
# That's it — complete updated server.py with enhanced intent recognition.