import os
import sys
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai

# ---------------------------------------------------------
# Load .env
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(env_path)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError(f"❌ GOOGLE_API_KEY not found in .env file at: {env_path}")

try:
    client = genai.Client(api_key=api_key)
    print("✅ Gemini client initialized successfully.")
except Exception as e:
    print(f"⚠️ Gemini client initialization failed: {e}", file=sys.stderr)
    client = None

# ---------------------------------------------------------
# Choose a working model
# ---------------------------------------------------------
possible_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
selected_model = None

for model_name in possible_models:
    try:
        client.models.generate_content(model=model_name, contents="Hello")
        selected_model = model_name
        print(f"🤖 Using model: {selected_model}")
        break
    except Exception:
        pass

if not selected_model:
    print("❌ No working model found. Please check your API key or permissions.")
    sys.exit(1)

# ---------------------------------------------------------
# Load personal docs
# ---------------------------------------------------------
DATA_PATH = os.path.join(current_dir, "my_info_data")
docs = []

if not os.path.exists(DATA_PATH):
    print(f"❌ Folder not found: {DATA_PATH}")
    sys.exit()

for file in os.listdir(DATA_PATH):
    if file.endswith(".txt"):
        with open(os.path.join(DATA_PATH, file), "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() != ""]
            text = "\n".join(lines)
            docs.append(f"=== {file} ===\n{text}")

if len(docs) == 0:
    print("❌ No .txt docs found in my_info_data.")
    sys.exit()

print("🔍 Creating embeddings...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(docs, convert_to_numpy=True)

d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
faiss.normalize_L2(embeddings)
index.add(embeddings)
print("✅ FAISS index ready.")

# ---------------------------------------------------------
# Retrieval function
# ---------------------------------------------------------
def retrieve(query, k=None):
    if k is None:
        k = len(docs)
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return [docs[i] for i in I[0]]

# ---------------------------------------------------------
# Ask agent with chat history and summary mode
# ---------------------------------------------------------
chat_history = []  # store conversation history

def ask_agent(question):
    # Append user message to chat history
    chat_history.append(f"You: {question}")

    # Detect summary request
    summary_triggers = ["summary", "introduce yourself", "about yourself", "who are you"]
    if any(word in question.lower() for word in summary_triggers):
        context_docs = retrieve("summary")
        context = "\n\n".join(context_docs)
        prompt = f"""
You are MY INFO AGENT.
Using the information below, provide a concise one-paragraph summary of Zakaria Said.
Do NOT add information not present in the documents.

Context:
{context}

Provide the summary:
"""
    else:
        # Include documents + chat history in context
        context_docs = retrieve(question)
        context = "\n\n".join(context_docs)
        history_text = "\n".join(chat_history[-10:])  # last 10 messages
        prompt = f"""
You are MY INFO AGENT.
Answer ONLY using the information provided below.
- If a question has multiple parts, answer each part clearly.
- If the question asks for a list (skills, projects, certificates, goals), provide bullet points.
- If information is not available, reply: "I don't have that information yet."
- Use previous conversation context to answer follow-up questions naturally.

Context:
{context}

Conversation history:
{history_text}

Question: {question}
Answer:
"""

    try:
        response = client.models.generate_content(
            model=selected_model,
            contents=prompt
        )
        answer = response.text.strip()
        if not answer:
            answer = "I don't have that information yet."
        # Append agent's response to chat history
        chat_history.append(f"My Info Agent: {answer}")
        return answer
    except Exception as e:
        return f"❌ Error querying Gemini: {e}"

# ---------------------------------------------------------
# Interactive chat loop
# ---------------------------------------------------------
def main():
    print("\n=== Welcome to My Info Agent (Gemini) ===")
    print("Ask any question about yourself. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        answer = ask_agent(user_input)
        print(f"My Info Agent: {answer}\n")

if __name__ == "__main__":
    main()
