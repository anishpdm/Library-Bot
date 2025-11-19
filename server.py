import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

import spacy
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from rapidfuzz import process as rf_process, fuzz


print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")  # light and fast

print("Loading sentence-transformers model (this may take a while)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small & fast

print("Loading zero-shot classifier (fallback, may be slower)...")
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

print("NLP models loaded.")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

books = [
    {"id": 1, "title": "Python Fundamentals", "category": "Python", "available": True},
    {"id": 2, "title": "Machine Learning Basics", "category": "ML", "available": False},
    {"id": 3, "title": "Deep Learning with PyTorch", "category": "ML", "available": True},
    {"id": 4, "title": "Advanced Python Patterns", "category": "Python", "available": True},
    {"id": 5, "title": "Introduction to AI", "category": "AI", "available": True},
    {"id": 6, "title": "NLP with Transformers", "category": "ML", "available": True},
    {"id": 7, "title": "Data Structures in Python", "category": "Python", "available": True},
    {"id": 8, "title": "AI Ethics", "category": "AI", "available": True},
    {"id": 9, "title": "Probabilistic ML", "category": "ML", "available": True},
    {"id": 10, "title": "Hands-On Deep Learning", "category": "ML", "available": True},
]

for b in books:
    b["embedding"] = embedder.encode(b["title"], convert_to_tensor=True)

user_books = {}  # {user_id: [{book_id, issued_at, due_at}]}

user_context: Dict[str, Dict[str, Any]] = {}


def serialize_book(book):
    """Return a JSON-serializable version of the book without embeddings"""
    return {
        "id": book["id"],
        "title": book["title"],
        "category": book["category"],
        "available": book["available"]
    }

def serialize_books(book_list):
    """Serialize a list of books for JSON response"""
    return [serialize_book(book) for book in book_list]


INTENT_LABELS = [
    "search_book",
    "issue_book",
    "return_book",
    "renew_book",
    "list_books",
    "check_due",
    "my_books",
    "greeting",
    "unknown"
]

INTENT_EXAMPLES = {
    "search_book": [
        "find book",
        "do you have",
        "search for",
        "look up",
        "find",
        "search",
        "look for books about",
        "show me books on"
    ],
    "issue_book": [
        "i want to borrow",
        "issue book",
        "borrow",
        "issue another book",
        "get book",
        "can i borrow",
        "i'd like to borrow"
    ],
    "return_book": [
        "i want to return",
        "give back the book",
        "return",
        "i'm returning"
    ],
    "renew_book": [
        "renew my book",
        "extend due date",
        "extend loan for",
        "renew"
    ],
    "list_books": [
        "list all books",
        "show me all books",
        "show books",
        "list",
        "show all",
        "display books"
    ],
    "check_due": [
        "when is my due date",
        "when to return",
        "due date for my books",
        "when are my books due"
    ],
    "my_books": [
        "what books i borrowed",
        "my borrowed books",
        "books issued to me",
        "any book issued to me",
        "my books"
    ],
    "greeting": [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    ]
}

INTENT_PROTOTYPES = {}
for intent, examples in INTENT_EXAMPLES.items():
    emb = embedder.encode(examples, convert_to_tensor=True)
    # mean pooling to get single vector
    if hasattr(emb, "mean"):
        proto = emb.mean(dim=0)
    else:
        # fallback if returned numpy
        import numpy as np
        proto = np.mean(emb, axis=0)
    INTENT_PROTOTYPES[intent] = proto


def semantic_intent_score(text: str) -> Tuple[str, float]:
    """Return (best_intent, score) by comparing embedding to prototypes."""
    q_emb = embedder.encode(text, convert_to_tensor=True)
    best = ("unknown", -1.0)
    for intent, proto in INTENT_PROTOTYPES.items():
        score = float(util.cos_sim(q_emb, proto)[0][0])
        if score > best[1]:
            best = (intent, score)
    return best  # (intent_name, similarity_score)


def fuzzy_title_matches(query: str, limit: int = 5) -> List[dict]:
    """Return top fuzzy matches from books using RapidFuzz."""
    titles = [b["title"] for b in books]
    results = rf_process.extract(query, titles, scorer=fuzz.WRatio, limit=limit)
    # results: [(title, score, index), ...]
    matches = []
    for title, score, idx in results:
        if score >= 55:  # threshold
            matches.append({**serialize_book(books[idx]), "score": score})
    return matches

def semantic_title_matches(query: str, threshold: float = 0.35, limit: int = 5) -> List[dict]:
    q_emb = embedder.encode(query, convert_to_tensor=True)
    scored = []
    for b in books:
        sim = float(util.cos_sim(q_emb, b["embedding"])[0][0])
        scored.append((sim, b))
    scored.sort(reverse=True, key=lambda x: x[0])
    results = [serialize_book(b) for sim,b in scored if sim >= threshold][:limit]
    return results

def find_books(query: str, limit: int = 5) -> List[dict]:
    # try semantic first
    sem = semantic_title_matches(query, threshold=0.32, limit=limit)
    if sem:
        return sem
    # try fuzzy
    fuzzy = fuzzy_title_matches(query, limit=limit)
    return fuzzy

# -------------------------
# DETECT INTENT - hybrid with context awareness
# -------------------------
def detect_intent(message: str, context: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Hybrid intent detection with context awareness.
    Returns (intent, updated_context)
    """
    # 1) quick normalization + spelling correction
    mb = TextBlob(message)
    try:
        corrected = str(mb.correct())
    except Exception:
        corrected = message

    text = corrected.strip().lower()
    
    # Initialize context if None
    if context is None:
        context = {}

    # 2) Handle context-aware responses first (highest priority)
    if context.get("awaiting_list_confirmation") and text in ["yes", "yeah", "yep", "sure", "ok", "okay"]:
        # Clear the context and return list_books intent
        context.pop("awaiting_list_confirmation", None)
        return "list_books", context
        
    if context.get("awaiting_list_confirmation") and text in ["no", "nope", "nah"]:
        # Clear the context
        context.pop("awaiting_list_confirmation", None)
        return "unknown", context

    # 3) Quick keyword checks (high precision) - check for exact matches first
    if text in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]:
        return "greeting", context
    
    # 4) Check for book-related keywords (medium priority)
    book_keywords = ["python", "machine learning", "deep learning", "ai", "ml", "data", "programming"]
    if any(keyword in text for keyword in book_keywords):
        # If it contains book keywords, prioritize search/issue intents
        if any(k in text for k in ["search", "find", "look for", "do you have"]):
            return "search_book", context
        elif any(k in text for k in ["issue", "borrow", "take", "get"]):
            return "issue_book", context
        else:
            # Default to search for book-related queries
            return "search_book", context

    # 5) Quick command checks
    if text in ["my books", "borrowed", "issued to me", "what i borrowed", "what have i borrowed", "any book issued to me"]:
        return "my_books", context
    if text in ["list", "list books", "show all", "show all books", "show books"]:
        return "list_books", context

    # 6) semantic prototype matching (lower confidence threshold for short queries)
    intent, score = semantic_intent_score(text)
    
    # Adjust confidence threshold based on query length
    confidence_threshold = 0.65 if len(text.split()) <= 2 else 0.55
    
    # if very confident, return immediately
    if score >= confidence_threshold:
        return intent, context

    # 7) zero-shot fallback when score is uncertain
    try:
        zs = zero_shot(text, list(INTENT_PROTOTYPES.keys()), multi_label=False)
        zs_label = zs["labels"][0] if zs and "labels" in zs else None
        zs_score = zs["scores"][0] if zs and "scores" in zs else 0.0
    except Exception as e:
        print(f"Zero-shot error: {e}")
        zs_label = None
        zs_score = 0.0

    # choose best of semantic prototype vs zero-shot
    if zs_label and zs_score >= 0.7 and zs_score > score:
        return zs_label, context

    # 8) Fallback heuristics for book actions (use keywords)
    if any(k in text for k in ["issue", "borrow", "take", "get"]):
        return "issue_book", context
    if any(k in text for k in ["return", "give back", "give"]):
        return "return_book", context
    if any(k in text for k in ["renew", "extend", "extension"]):
        return "renew_book", context
    if any(k in text for k in ["search", "find", "look for", "do you have"]):
        return "search_book", context
    if any(k in text for k in ["due", "when to return", "deadline"]):
        return "check_due", context

    # 9) Final fallback - be more conservative with short queries
    if len(text.split()) <= 2 and intent == "greeting" and score < 0.7:
        return "unknown", context
        
    return intent if intent != "unknown" else "unknown", context

# -------------------------
# Book ops (same as before)
# -------------------------
def find_book_by_id(book_id: int):
    for b in books:
        if b["id"] == book_id:
            return b
    return None

def issue_book_action(user_id: str, book_id: int):
    b = find_book_by_id(book_id)
    if not b:
        return {"ok": False, "message": "Book not found.", "ask_list": True}
    if not b["available"]:
        return {"ok": False, "message": f"'{b['title']}' is currently unavailable."}
    b["available"] = False
    issued_at = datetime.utcnow()
    due_at = issued_at + timedelta(days=14)
    user_books.setdefault(user_id, []).append({
        "book_id": b["id"],
        "issued_at": issued_at.isoformat(),
        "due_at": due_at.isoformat()
    })
    return {"ok": True, "message": f"'{b['title']}' issued. Due on {due_at.date().isoformat()}."}

def return_book_action(user_id: str, book_id: int):
    if user_id not in user_books:
        return {"ok": False, "message": "You haven't borrowed any books."}
    record = None
    for r in user_books[user_id]:
        if r["book_id"] == book_id:
            record = r
            break
    if not record:
        return {"ok": False, "message": "You don't have this book."}
    user_books[user_id].remove(record)
    b = find_book_by_id(book_id)
    if b:
        b["available"] = True
    return {"ok": True, "message": f"Returned '{b['title']}'."}

def renew_book_action(user_id: str, book_id: int):
    if user_id not in user_books:
        return {"ok": False, "message": "No books to renew."}
    for r in user_books[user_id]:
        if r["book_id"] == book_id:
            due = datetime.fromisoformat(r["due_at"])
            new_due = due + timedelta(days=7)
            r["due_at"] = new_due.isoformat()
            return {"ok": True, "message": f"'{find_book_by_id(book_id)['title']}' renewed to {new_due.date().isoformat()}."}
    return {"ok": False, "message": "This book isn't issued to you."}

# -------------------------
# REST helpers (suggest / books)
# -------------------------
from fastapi import APIRouter
@app.get("/suggest")
async def suggest(q: str = Query("", min_length=0)):
    q = q.strip()
    if not q:
        return JSONResponse({"suggestions": []})
    # use fuzzy and semantic matches combined, return titles
    sem = semantic_title_matches(q, threshold=0.30, limit=6)
    fuzzy = fuzzy_title_matches(q, limit=6)
    # combine uniquely
    seen = set()
    out = []
    for b in (sem + fuzzy):
        if b["id"] not in seen:
            seen.add(b["id"])
            out.append(b["title"])
    return JSONResponse({"suggestions": out[:8]})

@app.get("/books")
async def get_books(page: int = 1, per_page: int = 5, category: Optional[str] = Query("all")):
    # simple pagination
    filtered = books if (not category or category == "all") else [b for b in books if b["category"].lower() == category.lower()]
    total = len(filtered)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page
    return JSONResponse({
        "page": page, "per_page": per_page, "total": total, "total_pages": total_pages,
        "items": serialize_books(filtered[start:end])
    })

# -------------------------
# WebSocket connection manager & endpoint
# -------------------------
class ConnectionManager:
    def __init__(self):
        self.active = {}

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

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)

    # Initialize user context
    user_context[client_id] = {}

    # send greeting + suggestion buttons
    await manager.send(client_id, {"type": "bot_message", "message": "👋 Hello! I'm the NLP-powered Library Assistant. What can I do for you?"})
    await manager.send(client_id, {
        "type": "suggestions",
        "options": [
            {"label": "Search for a book", "action": {"name": "prefill", "value": "Search for a book"}},
            {"label": "List books", "action": {"name": "list", "page": 1, "per_page": 5, "category": "all"}},
            {"label": "My books", "action": {"name": "my_books"}}
        ]
    })

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            user_msg = data.get("message")
            action = data.get("action")

            if action:
                name = action.get("name")
                # action handlers: list / issue_by_id / return_by_id / renew_by_id / my_books / prefill
                if name == "list":
                    page = int(action.get("page", 1))
                    per_page = int(action.get("per_page", 5))
                    cat = action.get("category", "all")
                    # reuse get_books logic
                    filtered = books if (cat == "all") else [b for b in books if b["category"].lower() == cat.lower()]
                    total = len(filtered)
                    total_pages = max(1, (total + per_page - 1) // per_page)
                    page = max(1, min(page, total_pages))
                    start = (page - 1) * per_page
                    end = start + per_page
                    await manager.send(client_id, {"type": "book_list", "data": {
                        "page": page, 
                        "per_page": per_page, 
                        "total": total, 
                        "total_pages": total_pages, 
                        "items": serialize_books(filtered[start:end])
                    }})
                    continue
                if name == "issue_by_id" or name == "issue":
                    book_id = int(action.get("book_id") or action.get("id"))
                    res = issue_book_action(client_id, book_id)
                    await manager.send(client_id, {"type": "bot_message", "message": res["message"]})
                    continue
                if name == "return_by_id" or name == "return":
                    book_id = int(action.get("book_id") or action.get("id"))
                    res = return_book_action(client_id, book_id)
                    await manager.send(client_id, {"type": "bot_message", "message": res["message"]})
                    continue
                if name == "renew_by_id" or name == "renew":
                    book_id = int(action.get("book_id") or action.get("id"))
                    res = renew_book_action(client_id, book_id)
                    await manager.send(client_id, {"type": "bot_message", "message": res["message"]})
                    continue
                if name == "my_books":
                    recs = user_books.get(client_id, [])
                    formatted = []
                    for r in recs:
                        b = find_book_by_id(r["book_id"])
                        if b:
                            formatted.append({"book_id": b["id"], "title": b["title"], "issued_at": r["issued_at"][:10], "due_at": r["due_at"][:10]})
                    await manager.send(client_id, {"type": "my_books", "items": formatted})
                    continue
                if name == "prefill":
                    await manager.send(client_id, {"type": "prefill", "value": action.get("value", "")})
                    continue

            if user_msg:
                # echo and typing indicator
                await manager.send(client_id, {"type": "user_message", "message": user_msg})
                await manager.send(client_id, {"type": "typing", "value": True})

                try:
                    # Get current context for this user
                    current_context = user_context.get(client_id, {})
                    
                    # Detect intent with context awareness
                    intent, updated_context = detect_intent(user_msg, current_context)
                    user_context[client_id] = updated_context
                    
                    print(f"[intent] {user_msg} -> {intent} (context: {current_context})")

                    if intent == "search_book":
                        matches = find_books(user_msg)
                        if not matches:
                            await manager.send(client_id, {"type": "bot_message", "message": "Book not found. Want to list all books?"})
                            await manager.send(client_id, {"type": "suggestions", "options": [{"label":"Show all books","action":{"name":"list","page":1,"per_page":5,"category":"all"}}]})
                            # Set context for follow-up
                            user_context[client_id]["awaiting_list_confirmation"] = True
                        else:
                            # send search_results (client handles formatting)
                            await manager.send(client_id, {"type": "search_results", "results": matches})
                    elif intent == "issue_book":
                        matches = find_books(user_msg)
                        if not matches:
                            await manager.send(client_id, {"type": "bot_message", "message": "Book not found. Show list?"})
                            await manager.send(client_id, {"type": "suggestions", "options": [{"label":"Show all books","action":{"name":"list","page":1,"per_page":5,"category":"all"}}]})
                            # Set context for follow-up
                            user_context[client_id]["awaiting_list_confirmation"] = True
                        elif len(matches) == 1:
                            res = issue_book_action(client_id, matches[0]["id"])
                            await manager.send(client_id, {"type": "bot_message", "message": res["message"]})
                        else:
                            # multiple options — send as suggestions to click
                            opts = [{"label": f"Issue '{m['title']}'", "action": {"name": "issue_by_id", "book_id": m["id"]}} for m in matches]
                            await manager.send(client_id, {"type": "suggestions", "options": opts})
                    elif intent == "return_book":
                        matches = find_books(user_msg)
                        if matches:
                            res = return_book_action(client_id, matches[0]["id"])
                            await manager.send(client_id, {"type": "bot_message", "message": res["message"]})
                        else:
                            await manager.send(client_id, {"type": "bot_message", "message": "Book not found."})
                    elif intent == "renew_book":
                        matches = find_books(user_msg)
                        if matches:
                            res = renew_book_action(client_id, matches[0]["id"])
                            await manager.send(client_id, {"type": "bot_message", "message": res["message"]})
                        else:
                            await manager.send(client_id, {"type": "bot_message", "message": "Book not found."})
                    elif intent == "list_books":
                        filtered = books
                        await manager.send(client_id, {"type": "book_list", "data": {
                            "page": 1,
                            "per_page": 5, 
                            "total": len(filtered),
                            "total_pages": 1,
                            "items": serialize_books(filtered[:5])
                        }})
                    elif intent == "my_books" or intent == "check_due":
                        recs = user_books.get(client_id, [])
                        formatted = []
                        for r in recs:
                            b = find_book_by_id(r["book_id"])
                            if b:
                                formatted.append({"book_id": b["id"], "title": b["title"], "issued_at": r["issued_at"][:10], "due_at": r["due_at"][:10]})
                        await manager.send(client_id, {"type": "my_books", "items": formatted})
                    elif intent == "greeting":
                        await manager.send(client_id, {"type": "bot_message", "message": "Hi there! I can help you search, issue, return, or renew books."})
                    else:
                        await manager.send(client_id, {"type": "bot_message", "message": "Sorry, I didn't understand that. Try 'search <title>' or click suggestions."})

                except Exception as e:
                    print(f"Error processing message: {e}")
                    await manager.send(client_id, {"type": "bot_message", "message": "Sorry, I encountered an error processing your request."})

                finally:
                    await manager.send(client_id, {"type": "typing", "value": False})

    except WebSocketDisconnect:
        user_context.pop(client_id, None)
        manager.disconnect(client_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        user_context.pop(client_id, None)
        manager.disconnect(client_id)

@app.get("/")
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())