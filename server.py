# server.py (full patched)
import os
import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

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
MAX_BORROWED = 4     # max books per user
DEFAULT_LOAN_DAYS = 14
RENEW_DAYS = 14

# -----------------------
# SQLAlchemy Async Setup
# -----------------------
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

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
except Exception as e:
    print("spaCy model load failed:", e)
    nlp = None

print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading zero-shot classifier (facebook/bart-large-mnli)...")
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

print("NLP loaded.")

# Intent prototypes
INTENT_EXAMPLES = {
    "search_book": ["find book", "do you have", "search for", "look up", "show me books on"],
    "issue_book": ["i want to borrow", "issue book", "borrow", "can i borrow"],
    "return_book": ["i want to return", "return", "i'm returning"],
    "renew_book": ["renew my book", "extend due date", "renew"],
    "list_books": ["list all books", "show me all books", "list", "show all"],
    "check_due": ["when is my due date", "when to return", "due date for my books"],
    "my_books": ["what books i borrowed", "my borrowed books", "books issued to me"],
    "greeting": ["hello", "hi", "hey", "good morning"]
}
INTENT_PROTOTYPES = {}
for intent, examples in INTENT_EXAMPLES.items():
    emb = embedder.encode(examples, convert_to_tensor=True)
    try:
        proto = emb.mean(dim=0)
    except Exception:
        import numpy as np
        proto = np.mean(emb, axis=0)
    INTENT_PROTOTYPES[intent] = proto


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
    results = [b for sim,b in scored if sim >= threshold][:limit]
    return results


async def find_books_db(session: AsyncSession, limit:int=9999):
    q = await session.execute(select(Book))
    rows = q.scalars().all()
    return [{"id": b.id, "title": b.title, "category": b.category, "author": b.author,
             "rack": b.rack, "total_copies": b.total_copies, "available_copies": b.available_copies} for b in rows]

# -----------------------
# FastAPI and static mount
# -----------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------
# Connection Manager (WebSocket)
# -----------------------
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

# -----------------------
# In-memory pending actions
# -----------------------
# pending_action[client_id] -> dict(action) or None
pending_action: Dict[str, Optional[dict]] = {}

# -----------------------
# DB helpers
# -----------------------
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
# Startup: create tables + seed data + attempt schema fixes
# -----------------------
@app.on_event("startup")
async def startup_event():
    try:
        async with engine.begin() as conn:
            # create tables if not present
            await conn.run_sync(Base.metadata.create_all)
            # attempt to add created_at columns if older schema exists (graceful)
            try:
                await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS created_at DATETIME DEFAULT CURRENT_TIMESTAMP"))
                await conn.execute(text("ALTER TABLE books ADD COLUMN IF NOT EXISTS created_at DATETIME DEFAULT CURRENT_TIMESTAMP"))
            except Exception:
                # ignore if ALTER not supported
                pass

        # seed sample books if none exist
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

        # start background scheduler
        asyncio.create_task(overdue_checker_task())

    except Exception as e:
        print("Startup error:", e)
        print("If you see 'greenlet' related errors, run: pip install greenlet")

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

# simple admin login
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

@app.post("/admin/add_book")
async def admin_add_book(payload: dict = Body(...)):
    async with AsyncSessionLocal() as session:
        book = Book(title=payload["title"], author=payload.get("author"), category=payload.get("category"), rack=payload.get("rack"), total_copies=payload.get("total_copies", 1), available_copies=payload.get("total_copies", 1))
        session.add(book)
        await session.commit()
        return {"ok": True}

@app.delete("/admin/delete_book/{book_id}")
async def admin_delete_book(book_id: int):
    async with AsyncSessionLocal() as session:
        book = await session.get(Book, book_id)
        if not book:
            return {"ok": False}
        await session.delete(book)
        await session.commit()
        return {"ok": True}

@app.get("/admin/issues")
async def admin_issues():
    async with AsyncSessionLocal() as session:
        q = await session.execute(select(Issue, Book, User).join(Book, Issue.book_id == Book.id).join(User, Issue.user_id == User.id).where(Issue.status == "issued"))
        rows = q.all()
        return [{"title": r[1].title, "user": r[2].client_id, "due": r[0].due_at.date().isoformat()} for r in rows]

@app.get("/admin/reservations")
async def admin_reservations():
    async with AsyncSessionLocal() as session:
        q = await session.execute(select(Reservation, Book, User).join(Book, Reservation.book_id == Book.id).join(User, Reservation.user_id == User.id))
        rows = q.all()
        return [{"title": r[1].title, "user": r[2].client_id, "date": r[0].reserved_at.date().isoformat()} for r in rows]

@app.get("/admin/fines")
async def admin_fines():
    async with AsyncSessionLocal() as session:
        q = await session.execute(select(Fine, Book, User).join(Book, Fine.book_id == Book.id).join(User, Fine.user_id == User.id).where(Fine.paid == False))
        rows = q.all()
        return [{"id": r[0].id, "title": r[1].title, "user": r[2].client_id, "amount": float(r[0].fine_amount)} for r in rows]

@app.post("/admin/pay_fine/{fine_id}")
async def admin_pay_fine(fine_id: int):
    async with AsyncSessionLocal() as session:
        f = await session.get(Fine, fine_id)
        if not f:
            return {"ok": False}
        f.paid = True
        session.add(f)
        await session.commit()
        return {"ok": True}

# -----------------------
# Centralized action processor (with confirmation support)
# -----------------------
async def process_action(client_id: str, action: dict):
    name = action.get("name")
    try:
        # For sensitive actions require explicit confirmation unless action['confirm'] is True
        sensitive = name in ("issue_by_id","issue","return_by_id","return","renew_by_id","renew")
        if sensitive and not action.get("confirm"):
            # store pending action (user must reply 'yes' or click Yes)
            pending_action[client_id] = action
            # craft friendly confirmation message
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

        # If here and sensitive, action.get('confirm')==True -> execute
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

        if name == "prefill":
            await manager.send(client_id, {"type":"prefill","value": action.get("value","")})
            return

        if name == "noop":
            await manager.send(client_id, {"type":"bot_message","message":"Okay."})
            return

        # special confirm_yes action (when Yes button clicked) -> execute pending action
        if name == "confirm_yes":
            to_do = pending_action.get(client_id)
            if to_do:
                # mark confirmed and execute
                to_do["confirm"] = True
                pending_action[client_id] = None
                await process_action(client_id, to_do)
            else:
                await manager.send(client_id, {"type":"bot_message","message":"Nothing to confirm."})
            return

        await manager.send(client_id, {"type":"bot_message","message":"Action not supported."})
    except Exception as e:
        print("process_action error:", e)
        await manager.send(client_id, {"type":"bot_message","message":"Sorry, an error occurred while processing your action."})

# -----------------------
# WebSocket endpoint (main chat)
# -----------------------
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    async with AsyncSessionLocal() as session:
        await get_or_create_user(session, client_id)

    await manager.send(client_id, {"type":"bot_message","message":"👋 Hello! I'm the Library Assistant. How can I help?"})
    await manager.send(client_id, {"type":"suggestions","options":[
        {"label":"Search for a book","action":{"name":"prefill","value":"Search for a book"}},
        {"label":"List books","action":{"name":"list","page":1,"per_page":5,"category":"all"}},
        {"label":"My books","action":{"name":"my_books"}}
    ]})

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            user_msg = data.get("message")
            action = data.get("action")

            # If action arrived (button click), do NOT clear pending_action here — process_action will handle confirm flow
            if action:
                await process_action(client_id, action)
                continue

            if user_msg:
                await manager.send(client_id, {"type":"user_message","message":user_msg})
                await manager.send(client_id, {"type":"typing","value":True})

                try:
                    text = user_msg.strip().lower()

                    # YES / NO handling
                    if text in ["yes", "yeah", "yup", "sure", "ok", "okay", "of course"]:
                        action_to_do = pending_action.get(client_id)
                        if action_to_do:
                            # mark confirm and execute
                            action_to_do["confirm"] = True
                            pending_action[client_id] = None
                            await process_action(client_id, action_to_do)
                            await manager.send(client_id, {"type":"typing","value":False})
                            continue
                        else:
                            await manager.send(client_id, {"type":"bot_message","message":"There is nothing to confirm."})
                            await manager.send(client_id, {"type":"typing","value":False})
                            continue

                    if text in ["no", "nope", "nah"]:
                        pending_action[client_id] = None
                        await manager.send(client_id, {"type":"bot_message","message":"Okay — no problem."})
                        await manager.send(client_id, {"type":"typing","value":False})
                        continue

                    intent, score = semantic_intent_score(user_msg)

                    if intent == "greeting" or text in ["hi","hello","hey"]:
                        await manager.send(client_id, {"type":"bot_message","message":"Hi there! I can help you search, reserve, issue, return, or renew books."})
                    elif intent == "search_book" or any(k in text for k in ["search","find","do you have","look for"]):
                        async with AsyncSessionLocal() as session:
                            simple_list = await find_books_db(session)
                            simple_list2 = [{"id": b["id"], "title": b["title"], "category": b["category"], "available": b["available_copies"] > 0, "rack": b["rack"], "available_copies": b["available_copies"]} for b in simple_list]
                            sem = semantic_title_matches(user_msg, simple_list2, threshold=0.32, limit=6)
                            if sem:
                                await manager.send(client_id, {"type":"search_results","results": sem})
                            else:
                                fuzzy = fuzzy_title_matches(user_msg, simple_list2, limit=6)
                                if fuzzy:
                                    await manager.send(client_id, {"type":"search_results","results": fuzzy})
                                else:
                                    pending_action[client_id] = {"name":"list","page":1,"per_page":5,"category":"all"}
                                    await manager.send(client_id, {"type":"bot_message","message":"No matching books found. Want to list all books?"})
                                    await manager.send(client_id, {"type":"suggestions","options":[{"label":"Show all books","action":{"name":"list","page":1,"per_page":5,"category":"all"}}]})
                    elif intent == "issue_book" or any(k in text for k in ["issue","borrow","i want to borrow","can i borrow"]):
                        async with AsyncSessionLocal() as session:
                            simple_list = await find_books_db(session)
                            simple_list2 = [{"id": b["id"], "title": b["title"], "category": b["category"], "available": b["available_copies"] > 0} for b in simple_list]
                            matches = semantic_title_matches(user_msg, simple_list2, threshold=0.32, limit=6) or fuzzy_title_matches(user_msg, simple_list2, limit=6)
                            if not matches:
                                pending_action[client_id] = {"name":"list","page":1,"per_page":5,"category":"all"}
                                await manager.send(client_id, {"type":"bot_message","message":"Book not found. Show list?"})
                                await manager.send(client_id, {"type":"suggestions","options":[{"label":"Show all books","action":{"name":"list","page":1,"per_page":5,"category":"all"}}]})
                            elif len(matches) == 1:
                                pending_action[client_id] = {"name":"issue_by_id","book_id":matches[0]['id']}
                                await manager.send(client_id, {"type":"bot_message","message": f"Do you want to issue '{matches[0]['title']}'? Reply 'yes' to confirm."})
                                await manager.send(client_id, {"type":"suggestions","options":[{"label":f"Yes, issue '{matches[0]['title']}'","action":{"name":"confirm_yes"}},{"label":"No","action":{"name":"noop"}}]})
                            else:
                                opts = [{"label": f"Issue '{m['title']}'", "action": {"name":"issue_by_id","book_id": m["id"]}} for m in matches]
                                await manager.send(client_id, {"type":"suggestions","options": opts})
                    elif intent == "renew_book" or "renew" in text:
                        async with AsyncSessionLocal() as session:
                            simple_list = await find_books_db(session)
                            simple_list2 = [{"id": b["id"], "title": b["title"]} for b in simple_list]
                            matches = semantic_title_matches(user_msg, simple_list2, threshold=0.32, limit=6) or fuzzy_title_matches(user_msg, simple_list2, limit=6)
                            if matches:
                                await manager.send(client_id, {"type":"suggestions","options":[{"label":f"Renew '{m['title']}'","action":{"name":"renew_by_id","book_id":m["id"]}} for m in matches]})
                            else:
                                await manager.send(client_id, {"type":"bot_message","message":"Book not found to renew."})
                    elif intent == "return_book" or "return" in text:
                        async with AsyncSessionLocal() as session:
                            simple_list = await find_books_db(session)
                            simple_list2 = [{"id": b["id"], "title": b["title"]} for b in simple_list]
                            matches = semantic_title_matches(user_msg, simple_list2, threshold=0.32, limit=6) or fuzzy_title_matches(user_msg, simple_list2, limit=6)
                            if matches:
                                await manager.send(client_id, {"type":"suggestions","options":[{"label":f"Return '{m['title']}'","action":{"name":"return_by_id","book_id":m["id"]}} for m in matches]})
                            else:
                                await manager.send(client_id, {"type":"bot_message","message":"Book not found."})
                    elif intent == "my_books" or any(k in text for k in ["my books","what i borrowed","due date"]):
                        async with AsyncSessionLocal() as session:
                            user = await get_or_create_user(session, client_id)
                            q = await session.execute(select(Issue).where(Issue.user_id==user.id).order_by(Issue.due_at))
                            recs = q.scalars().all()
                            out = []
                            total_fine = 0.0
                            for r in recs:
                                b = await session.get(Book, r.book_id)
                                fine_amt = 0.0
                                if r.due_at:
                                    days_left = (r.due_at.date() - datetime.utcnow().date()).days
                                    if days_left < 0:
                                        fine_amt = round((-days_left) * FINE_PER_DAY, 2)
                                        total_fine += fine_amt
                                out.append({"book_id": b.id, "title": b.title, "issued_at": r.issued_at.date().isoformat(), "due_at": r.due_at.date().isoformat(), "fine": fine_amt})
                            await manager.send(client_id, {"type":"my_books","items": out})
                            if total_fine > 0:
                                await manager.send(client_id, {"type":"bot_message","message":f"You have total pending fines: ₹{round(total_fine,2)}. Check /fines to pay."})
                    else:
                        await manager.send(client_id, {"type":"bot_message","message":"Sorry, I didn't understand that. Try 'search <title>' or click suggestions."})

                except Exception as e:
                    print("Error processing message:", e)
                    await manager.send(client_id, {"type":"bot_message","message":"Sorry, I encountered an error processing your request."})
                finally:
                    await manager.send(client_id, {"type":"typing","value":False})

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        print("WebSocket error:", e)
        manager.disconnect(client_id)

# -----------------------
# Root / index
# -----------------------
@app.get("/")
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# -----------------------
# Run: uvicorn server:app --reload
# -----------------------
