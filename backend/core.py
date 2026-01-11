'''from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry
from langcodes import Language
import fasttext
import redis
import time

# REDIS CONFIGURATION AND RATE LIMITING
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379)) 

r = None
try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    
    print(f"Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}. Rate limiting is active.")
except redis.exceptions.ConnectionError as e:
    print(f"Error connecting to Redis at {REDIS_HOST}:{REDIS_PORT}: {e}")
    print("Rate limiting will be DISABLED. Ensure Redis is running if needed.")
    r = None

RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 3
RATE_LIMIT_PENALTY_SECONDS = 30

def check_rate_limit(client_ip: str) -> bool:
    if r is None:
        return True

    # Check if user is currently in penalty period
    penalty_key = f"rl:penalty:{client_ip}"
    try:
        if r.exists(penalty_key):
            ttl = r.ttl(penalty_key)
            print(f"IP {client_ip} is in penalty period. {ttl} seconds remaining.")
            return False
    except redis.exceptions.RedisError as re:
        print(f"Redis operation failed checking penalty: {re}. Allowing request for safety.")
        return True

    # Check rate limit
    current_time_window = int(time.time() // RATE_LIMIT_WINDOW_SECONDS)
    redis_key = f"rl:{client_ip}:{current_time_window}"
    
    try:
        pipe = r.pipeline()
        pipe.incr(redis_key)
        pipe.expire(redis_key, RATE_LIMIT_WINDOW_SECONDS, nx=True)
        
        new_count = pipe.execute()[0]
    except redis.exceptions.RedisError as re:
        print(f"Redis operation failed: {re}. Allowing request for safety.")
        return True

    if new_count > RATE_LIMIT_MAX_REQUESTS:
        print(f"RATE LIMIT EXCEEDED for IP: {client_ip}. Count: {new_count}")
        
        # Set penalty period
        try:
            r.setex(penalty_key, RATE_LIMIT_PENALTY_SECONDS, "1")
            print(f"Penalty period of {RATE_LIMIT_PENALTY_SECONDS}s set for IP: {client_ip}")
        except redis.exceptions.RedisError as re:
            print(f"Failed to set penalty: {re}")
        
        return False
    
    return True 
# END OF REDIS RATE LIMITING


# FastText language identification model
try:
    LID_MODEL = fasttext.load_model('./fasttext/lid.176.ftz')
except ValueError as e:
    print(f"Error loading FastText model: {e}")
    print("Please make sure 'lid.176.ftz' is in your script's directory.")
    LID_MODEL = None

def get_language_name(lang_code):
    language = Language.make(language=lang_code).language_name()
    return language

def import_google_api():
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    client = genai.Client(api_key=GOOGLE_API_KEY)
    return client

def embedding_function(client):
    class GeminiEmbeddingFunction(EmbeddingFunction):
        document_mode = True

        def __init__(self, client):
            self.client = client
            self._retry = retry.Retry(predicate=lambda e: isinstance(e, genai.errors.APIError) and e.code in {429, 503})

        def __call__(self, input: Documents) -> Embeddings:
            embedding_task = "retrieval_document" if self.document_mode else "retrieval_query"
            response = self._retry(self.client.models.embed_content)(
                model="models/text-embedding-004",
                contents=input,
                config=types.EmbedContentConfig(task_type=embedding_task),
            )
            return [e.values for e in response.embeddings]

    return GeminiEmbeddingFunction(client)

def persistent_client_hr(embed_fn):
    persist_dir = "./output_hr"
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    DB_NAME = "hrstud-bot-hr"
    collection = chroma_client.get_collection(DB_NAME, embedding_function=embed_fn)
    return collection

def persistent_client_en(embed_fn):
    persist_dir = "./output_en"
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    DB_NAME = "hrstud-bot-en"
    collection = chroma_client.get_collection(DB_NAME, embedding_function=embed_fn)
    return collection

def get_query(
    user_query, 
    embed_fn, 
    collection_hr,
    collection_en,
    client, 
    request_ip: str
):
    if not check_rate_limit(request_ip):
        return """ERROR 429. Previše zahtjeva s Vaše IP adrese. Molimo pričekajte trenutak i pokušajte ponovo.
        \nERROR 429. Too many requests from your IP address. Please wait a moment and try again."""
        
    if not LID_MODEL:
        raise RuntimeError("FastText model is not loaded. Cannot detect language.")
    
    user_language_prediction = LID_MODEL.predict(user_query)
    user_language_code = user_language_prediction[0][0].replace('__label__', '')
    
    if user_language_code in ['hr', 'bs', 'sr']:
        user_language_code = 'hr'
    elif user_language_code == 'en':
        user_language_code = 'en'
    else:
        user_language_code = 'en'

    if user_language_code == 'hr':
        # Pass the correct HR collection
        return get_query_hr(user_query, embed_fn, collection_hr, client, request_ip)
    elif user_language_code == 'en':
        # Pass the correct EN collection
        return get_query_en(user_query, embed_fn, collection_en, client, request_ip)
    else:
        # Fallback (though the logic above forces it to 'hr' or 'en')
        return """ERROR: Unsupported language detected. Please ask your question in Croatian or English.
        \nERROR: Nepodržani jezik. Molimo postavite svoje pitanje na hrvatskom ili engleskom jeziku."""


def get_query_hr(user_query, embed_fn, collection, client, request_ip: str):
    
    # Switch to query mode when generating embeddings
    embed_fn.document_mode = False

    # Retrieve top 1 document (based on your n_results=1 in the original code)
    # The result structure is a dict: {'ids': [[]], 'distances': [[]], 'documents': [[]], 'metadatas': [[]], ...}
    n_results_to_fetch = 7 # Fetch more results for a richer context
    result = collection.query(query_texts=[user_query], n_results=n_results_to_fetch)
    
    # Extract documents (list of passages) and metadatas (list of dicts)
    all_passages = result["documents"][0]
    all_metadatas = result["metadatas"][0]

    query_oneline = user_query.replace("\n", " ")
    print(query_oneline)
    
    # 1. CONSTRUCT THE CONTEXT
    context_list = []
    # Use the metadata from the top result to define the main source link
    # Assuming 'source_path' contains the URL or relevant file path
    #document_link = all_metadatas[0].get("source_path", "Link nije dostupan")
    
    for i, (passage, metadata) in enumerate(zip(all_passages, all_metadatas)):
        # Format the context for the model
        source_name = metadata.get("source", "Nepoznat izvor")
        # I removed the redundant "PASSAGE: " wrapper that was causing issues
        context_list.append(f"--- Izvor: {source_name} (Dio {i+1} od {len(all_passages)}) ---\n{passage.strip()}")

    # Join all context chunks into a single string
    context = "\n\n".join(context_list)
    
    # 2. CONSTRUCT THE PROMPT
    # The document_link is now a defined variable

    embed_fn.document_mode = False
    result = collection.query(query_texts=[user_query], n_results=1)
    [all_passages] = result["documents"]
    query_oneline = user_query.replace("\n", " ")

    prompt = f"""
    Ti si ljubazan, precizan i informativan chatbot **Fakulteta Hrvatskih studija**. Tvoja je glavna zadaća odgovarati na pitanja studenata, potencijalnih studenata i osoblja o fakultetu, uključujući informacije o studijima, nastavi, smjerovima, prijavama, i općenitim informacijama o školi.

    **KRITIČNA PRAVILA:**
    1.  Koristi ISKLJUČIVO informacije iz dostavljene dokumentacije.
    2.  Odgovaraj na **Hrvatskom jeziku**.
    3.  Budi koncizan ali potpun — navedi sve relevantne detalje iz konteksta.
    4.  Ako dokumentacija ne sadrži odgovor, jasno i ljubazno reci da ne možeš pronaći odgovor u bazi znanja i uputi na kontaktiranje odgovarajuće službe.
    5.  **Ne smiješ koristiti fraze poput "Naravno, mogu vam pomoći!" ili "Evo nekoliko informacija o...". Odmah započni s relevantnim odgovorom.**

    **FORMATIRANJE ODGOVORA:**
    * Sve odgovore započni s **Izvorni link je [LINK](url)**, nakon čega slijedi prazan red.
    * Nemoj navoditi izvorni link dokumenta samo URL. npr nemoj navoditi: Izvorni link: ./markdown/fhs.hr_predmet_opsv.md
    * Koristi podebljani tekst za ključne pojmove (npr. **Upisi**, **Filozofija**, **Pročelnik**).
    * Koristi popise (liste) za nabrajanje informacija (studiji, uvjeti, rokovi).
    * Odgovori trebaju biti profesionalni i službeni, ali s ljubaznim tonom.
    **DOSTUPNA DOKUMENTACIJA (Kontekst):**
    {context}

    **KORISNIČKO PITANJE:** {query_oneline}

    **ODGOVOR:**
    """
    
    # 3. Call the model
    answer = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt, # Use the full prompt
        config={
            "max_output_tokens": 2048,
            "temperature": 0.2,
            "top_p": 0.9
        }
    )
    
    # Prepend the link as per your strict instruction, since Gemini might not format the first line perfectly
    #final_response = f"Izvorni link: {document_link}\n\n{answer.text.strip()}"
    
    #return final_response

    return answer.text.strip()


def get_query_en(user_query, embed_fn, collection, client, request_ip: str):
    
    embed_fn.document_mode = False
    n_results_to_fetch = 7
    result = collection.query(query_texts=[user_query], n_results=n_results_to_fetch)
    
    all_passages = result["documents"][0]
    all_metadatas = result["metadatas"][0]

    query_oneline = user_query.replace("\n", " ")
    print(query_oneline)
    
    # Extract the main source link (from the first/most relevant result)
    main_source_link = all_metadatas[0].get("source_path", "Link not available")
    
    # Construct context
    context_list = []
    source_links = []  # Collect all unique source links
    
    for i, (passage, metadata) in enumerate(zip(all_passages, all_metadatas)):
        source_name = metadata.get("source", "Unknown source")
        source_path = metadata.get("source_path", "")
        
        # Collect unique sources for the bottom reference
        if source_path and source_path not in source_links:
            source_links.append(source_path)
        
        context_list.append(f"--- Source: {source_name} (Part {i+1} of {len(all_passages)}) ---\n{passage.strip()}")

    context = "\n\n".join(context_list)
    
    # Format sources for the bottom of the answer
    sources_text = "\n".join([f"- {link}" for link in source_links])
    
    prompt = f"""
    You are a kind, precise, and informative chatbot of the **Faculty of Croatian Studies**. 
    Your main task is to answer questions from students, prospective students, and staff about the faculty, 
    including information about study programs, courses, departments, admissions, and general school information.

    **CRITICAL RULES:**
    1.  Use ONLY the information provided in the supplied documentation.
    2.  Respond **in English**.
    3.  Be concise but complete — **synthesize all relevant details from ALL context sources into ONE cohesive answer**.
    4.  If the documentation does not contain the answer, clearly and politely state that you cannot find the answer 
        in the knowledge base and direct the user to contact the appropriate office.
    5.  Note if some classes are not offered in English.
    6.  **Do not use phrases like "Of course, I can help you!" or "Here is some information about...". 
        Start directly with the relevant answer.**
    7.  **IMPORTANT: Provide ONE unified answer, not multiple separate responses for each source.**

    **ANSWER FORMATTING:**
    * If there is ONE main source, start with: **Source: [Main Source Link]** followed by a blank line.
    * If there are MULTIPLE sources, provide the answer first, then at the end add a "**Sources:**" section listing all source links.
    * Use bold text for key terms (e.g., **Admissions**, **Philosophy**, **Head of Department**).
    * **When listing courses taught by a professor, organize them by level:**
    - **Undergraduate Courses:**
    - **Graduate Courses:**
    - **Doctoral Courses:**
    * For each course, include relevant details like ECTS credits, course hours, and language availability.
    * Responses should be professional and formal, yet polite in tone.
    * **Combine information from all sources into a single, coherent response.**
    * **DO NOT repeat "The source link is..." multiple times. Use it ONCE at the top if there's one main source, OR list all sources at the bottom.**

    **AVAILABLE DOCUMENTATION (Context):**
    {context}

    **USER QUESTION:** {query_oneline}

    **INSTRUCTIONS FOR SOURCE CITATION:**
    Main source link: {main_source_link}
    All source links: {sources_text}

    If the answer comes primarily from ONE source, start with "**Source:** {main_source_link}".
    If the answer uses MULTIPLE sources, end your response with:

    **Sources:**
    {sources_text}

    **ANSWER (provide ONE unified response with courses organized by academic level):**
    """
    
    answer = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "max_output_tokens": 2048,
            "temperature": 0.2,
            "top_p": 0.9
        }
    )

    # Prepend the link as per your strict instruction, since Gemini might not format the first line perfectly
    #final_response = f"Izvorni link: {document_link}\n\n{answer.text.strip()}"
    
    #return final_response

    return answer.text.strip()'''

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry
from langcodes import Language
import fasttext
import redis
import time
import re

# REDIS CONFIGURATION AND RATE LIMITING
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379)) 

r = None
try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    
    print(f"Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}. Rate limiting is active.")
except redis.exceptions.ConnectionError as e:
    print(f"Error connecting to Redis at {REDIS_HOST}:{REDIS_PORT}: {e}")
    print("Rate limiting will be DISABLED. Ensure Redis is running if needed.")
    r = None

RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 3
RATE_LIMIT_PENALTY_SECONDS = 30

def check_rate_limit(client_ip: str) -> bool:
    if r is None:
        return True

    # Check if user is currently in penalty period
    penalty_key = f"rl:penalty:{client_ip}"
    try:
        if r.exists(penalty_key):
            ttl = r.ttl(penalty_key)
            print(f"IP {client_ip} is in penalty period. {ttl} seconds remaining.")
            return False
    except redis.exceptions.RedisError as re:
        print(f"Redis operation failed checking penalty: {re}. Allowing request for safety.")
        return True

    # Check rate limit
    current_time_window = int(time.time() // RATE_LIMIT_WINDOW_SECONDS)
    redis_key = f"rl:{client_ip}:{current_time_window}"
    
    try:
        pipe = r.pipeline()
        pipe.incr(redis_key)
        pipe.expire(redis_key, RATE_LIMIT_WINDOW_SECONDS, nx=True)
        
        new_count = pipe.execute()[0]
    except redis.exceptions.RedisError as re:
        print(f"Redis operation failed: {re}. Allowing request for safety.")
        return True

    if new_count > RATE_LIMIT_MAX_REQUESTS:
        print(f"RATE LIMIT EXCEEDED for IP: {client_ip}. Count: {new_count}")
        
        # Set penalty period
        try:
            r.setex(penalty_key, RATE_LIMIT_PENALTY_SECONDS, "1")
            print(f"Penalty period of {RATE_LIMIT_PENALTY_SECONDS}s set for IP: {client_ip}")
        except redis.exceptions.RedisError as re:
            print(f"Failed to set penalty: {re}")
        
        return False
    
    return True 
# END OF REDIS RATE LIMITING


# FastText language identification model
try:
    LID_MODEL = fasttext.load_model('./fasttext/lid.176.ftz')
except ValueError as e:
    print(f"Error loading FastText model: {e}")
    print("Please make sure 'lid.176.ftz' is in your script's directory.")
    LID_MODEL = None

def get_language_name(lang_code):
    language = Language.make(language=lang_code).language_name()
    return language

def import_google_api():
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    client = genai.Client(api_key=GOOGLE_API_KEY)
    return client

def embedding_function(client):
    class GeminiEmbeddingFunction(EmbeddingFunction):
        document_mode = True

        def __init__(self, client):
            self.client = client
            self._retry = retry.Retry(predicate=lambda e: isinstance(e, genai.errors.APIError) and e.code in {429, 503})

        def __call__(self, input: Documents) -> Embeddings:
            embedding_task = "retrieval_document" if self.document_mode else "retrieval_query"
            response = self._retry(self.client.models.embed_content)(
                model="models/text-embedding-004",
                contents=input,
                config=types.EmbedContentConfig(task_type=embedding_task),
            )
            return [e.values for e in response.embeddings]

    return GeminiEmbeddingFunction(client)

def persistent_client_hr(embed_fn):
    persist_dir = "./output_hr"
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    DB_NAME = "hrstud-bot-hr"
    collection = chroma_client.get_collection(DB_NAME, embedding_function=embed_fn)
    return collection

def persistent_client_en(embed_fn):
    persist_dir = "./output_en"
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    DB_NAME = "hrstud-bot-en"
    collection = chroma_client.get_collection(DB_NAME, embedding_function=embed_fn)
    return collection

def extract_document_filter_hr(query: str) -> dict:
    """
    Extract document/section filters from user query (Croatian).
    Returns ChromaDB where clause if specific document mentioned.
    """
    query_lower = query.lower()
    
    # Common document name patterns
    doc_patterns = {
        "upis": ["upis", "prijav"],
        "studij": ["studij", "program"],
        "raspored": ["raspored", "termin"],
        "ispit": ["ispit", "kolokvij"],
    }
    
    for doc_key, patterns in doc_patterns.items():
        if any(pattern in query_lower for pattern in patterns):
            # Check if asking specifically about a document
            if any(phrase in query_lower for phrase in ["u dokumentu", "na stranici", "dokument o"]):
                return {"doc_name": {"$contains": doc_key}}
    
    return None  # No specific filter

def extract_document_filter_en(query: str) -> dict:
    """
    Extract document/section filters from user query (English).
    Returns ChromaDB where clause if specific document mentioned.
    """
    query_lower = query.lower()
    
    # Common document name patterns
    doc_patterns = {
        "admission": ["admission", "enroll", "apply"],
        "program": ["program", "study", "course"],
        "schedule": ["schedule", "timetable"],
        "exam": ["exam", "test", "assessment"],
    }
    
    for doc_key, patterns in doc_patterns.items():
        if any(pattern in query_lower for pattern in patterns):
            # Check if asking specifically about a document
            if any(phrase in query_lower for phrase in ["in the document", "on the page", "document about"]):
                return {"doc_name": {"$contains": doc_key}}
    
    return None  # No specific filter

def extract_article_link_from_content(content: str) -> str:
    """
    Extract the article link from markdown content.
    Only matches links with the exact text "Article Link".
    """
    # Pattern: Exact match for [Article Link](URL)
    link_pattern = r'\[Article Link\]\((https?://[^\)]+)\)'
    match = re.search(link_pattern, content[:1000])
    
    if match:
        url = match.group(1)
        return f"[Article Link]({url})"
    
    # Fallback: return None if no "Article Link" found
    return None

def get_query(
    user_query, 
    embed_fn, 
    collection_hr,
    collection_en,
    client, 
    request_ip: str
):
    if not check_rate_limit(request_ip):
        return """ERROR 429. Previše zahtjeva s Vaše IP adrese. Molimo pričekajte trenutak i pokušajte ponovo.
        \nERROR 429. Too many requests from your IP address. Please wait a moment and try again."""
        
    if not LID_MODEL:
        raise RuntimeError("FastText model is not loaded. Cannot detect language.")
    
    user_language_prediction = LID_MODEL.predict(user_query)
    user_language_code = user_language_prediction[0][0].replace('__label__', '')
    
    if user_language_code in ['hr', 'bs', 'sr']:
        user_language_code = 'hr'
    elif user_language_code == 'en':
        user_language_code = 'en'
    else:
        user_language_code = 'en'

    if user_language_code == 'hr':
        # Pass the correct HR collection
        return get_query_hr(user_query, embed_fn, collection_hr, client, request_ip)
    elif user_language_code == 'en':
        # Pass the correct EN collection
        return get_query_en(user_query, embed_fn, collection_en, client, request_ip)
    else:
        # Fallback (though the logic above forces it to 'hr' or 'en')
        return """ERROR: Unsupported language detected. Please ask your question in Croatian or English.
        \nERROR: Nepodržani jezik. Molimo postavite svoje pitanje na hrvatskom ili engleskom jeziku."""


def get_query_hr(user_query, embed_fn, collection, client, request_ip: str):
    """
    Optimized retrieval with query expansion and metadata filtering (Croatian).
    Extracts article link from actual document content.
    """
    # 1. QUERY PREPARATION
    embed_fn.document_mode = False
    query_lower = user_query.lower()
    
    # Initial values
    expanded_query = user_query
    n_results_to_fetch = 12
    metadata_filter = extract_document_filter_hr(user_query)

    # 2. QUERY EXPANSION LOGIC
    if any(word in query_lower for word in ["predaje", "tko", "nastavnik", "profesor", "kolegij"]):
        expanded_query = f"{user_query} profesor nositelj zvanje nastava kolegij studij"
    
    elif any(word in query_lower for word in ["doći", "lokacija", "gdje", "kampus", "borongaj", "autobus", "vlak"]):
        expanded_query = f"{user_query} lokacija adresa kampus borongaj autobus 215 236 vlak stanica Trnava"

    elif any(word in query_lower for word in ["studij", "nudite", "program", "upisi", "smjer"]):
        expanded_query = f"{user_query} popis svih studija prijediplomski diplomski doktorski studij kroatologija povijest sociologija psihologija komunikologija filozofija"
        n_results_to_fetch = 20
    
    # 3. VECTOR SEARCH WITH OPTIONAL FILTERING
    query_params = {
        "query_texts": [expanded_query], 
        "n_results": n_results_to_fetch
    }
    
    if metadata_filter:
        query_params["where"] = metadata_filter
        print(f"Applying metadata filter: {metadata_filter}")
    
    result = collection.query(**query_params)
    
    all_passages = result["documents"][0]
    all_metadatas = result["metadatas"][0]
    all_distances = result["distances"][0]

    # 4. EXTRACT ARTICLE LINK FROM TOP RESULT CONTENT
    article_link_markdown = None
    if all_passages:
        # Try to extract from the top result's content
        article_link_markdown = extract_article_link_from_content(all_passages[0])
        
        # Fallback to metadata if extraction fails
        if not article_link_markdown:
            main_url = all_metadatas[0].get("article_link", "")
            if main_url:
                article_link_markdown = f"[{main_url}](https://www.fhs.hr/{main_url})"
            else:
                article_link_markdown = "[Fakultet Hrvatskih studija](https://www.fhs.hr)"

    # 5. DEDUPLICATION & CONTEXT PREPARATION
    context_list = []
    seen_passages = set()
    
    # Dynamic threshold based on query type
    distance_threshold = 0.85 if "studij" in query_lower else 0.90
    
    for p, m, d in zip(all_passages, all_metadatas, all_distances):
        if d < distance_threshold:
            # Use first 200 chars as fingerprint
            fingerprint = p.strip()[:200]
            if fingerprint in seen_passages:
                continue
            seen_passages.add(fingerprint)
            
            chunk_url = m.get("article_link", "")
            header = m.get("header_path", "")
            
            # Include header context if available
            context_entry = f"Izvor URL: {chunk_url}"
            if header:
                context_entry += f"\nSekcija: {header}"
            context_entry += f"\n{p.strip()}"
            
            context_list.append(context_entry)

    if not context_list:
        return "Nažalost, ne mogu pronaći informacije o vašem upitu u bazi znanja. Molimo kontaktirajte studentsku službu za dodatne informacije."

    context = "\n\n".join(context_list)
    query_oneline = user_query.replace("\n", " ")

    # 6. OPTIMIZED PROMPT
    prompt = f"""Ti si ljubazan, precizan i informativan chatbot **Fakulteta Hrvatskih studija**. Tvoja je zadaća odgovarati na pitanja o fakultetu.

**KRITIČNA PRAVILA:**
1. Koristi ISKLJUČIVO dostavljenu dokumentaciju (KONTEKST).
2. Odgovaraj na **Hrvatskom jeziku**.
3. **GRUPIRANJE:** Ako ista osoba predaje više kolegija, navedi ime SAMO JEDNOM. Grupiraj studije po razinama.
4. **POVEZNICE:** Kolegije, emailove i studije prikaži kao Markdown poveznice [Naziv](URL).
5. **BEZ UVODA:** Odmah započni s relevantnim odgovorom bez fraza poput "Naravno...".

**FORMATIRANJE:**
* Prva linija: **Izvor:** Link koji ti se iz dokumentacije čini najrelevantnijim.
* Prazan red nakon izvora.
* **Podebljani tekst** za ključne pojmove.
* Liste (bullet points) za nabrajanje.
* Ako je navedeni link u ovom formatu npr. u ovom slučaju za Email: E-mail: [idzinic@fhs.hr](javascript:startMail('qvvmva@pus.feu');)
    - prikaži kao običnu poveznicu bez javascript dijela, npr. E-mail: idzinic@fhs.hr
    - makni javascript dio iz poveznice za svaku poveznicu
**DOSTUPNA DOKUMENTACIJA:**
{context}

**KORISNIČKO PITANJE:** {query_oneline}

**ODGOVOR:**"""
    
    # 7. GENERATE RESPONSE
    answer = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=prompt,
        config={
            "max_output_tokens": 2048,
            "temperature": 0.1,
            "top_p": 0.9
        }
    )
    
    return answer.text.strip()


def get_query_en(user_query, embed_fn, collection, client, request_ip: str):
    """
    Optimized retrieval with query expansion and metadata filtering (English).
    Extracts article link from actual document content.
    """
    # 1. QUERY PREPARATION
    embed_fn.document_mode = False
    query_lower = user_query.lower()
    
    # Initial values
    expanded_query = user_query
    n_results_to_fetch = 12
    metadata_filter = extract_document_filter_en(user_query)

    # 2. QUERY EXPANSION LOGIC
    if any(word in query_lower for word in ["teaches", "who", "professor", "instructor", "course", "class"]):
        expanded_query = f"{user_query} professor instructor title teaching course class program study"
    
    elif any(word in query_lower for word in ["get to", "location", "where", "campus", "borongaj", "bus", "train"]):
        expanded_query = f"{user_query} location address campus borongaj bus 215 236 train station Trnava"

    elif any(word in query_lower for word in ["program", "offer", "study", "admission", "major"]):
        expanded_query = f"{user_query} list of programs undergraduate graduate doctoral study croatology history sociology psychology communication philosophy"
        n_results_to_fetch = 20
    
    # 3. VECTOR SEARCH WITH OPTIONAL FILTERING
    query_params = {
        "query_texts": [expanded_query], 
        "n_results": n_results_to_fetch
    }
    
    if metadata_filter:
        query_params["where"] = metadata_filter
        print(f"Applying metadata filter: {metadata_filter}")
    
    result = collection.query(**query_params)
    
    all_passages = result["documents"][0]
    all_metadatas = result["metadatas"][0]
    all_distances = result["distances"][0]

    # 4. EXTRACT ARTICLE LINK FROM TOP RESULT CONTENT
    article_link_markdown = None
    if all_passages:
        # Try to extract from the top result's content
        article_link_markdown = extract_article_link_from_content(all_passages[0])
        
        # Fallback to metadata if extraction fails
        if not article_link_markdown:
            main_url = all_metadatas[0].get("article_link", "")
            if main_url:
                article_link_markdown = f"[{main_url}](https://www.fhs.hr/{main_url})"
            else:
                article_link_markdown = "[Faculty of Croatian Studies](https://www.fhs.hr)"

    # 5. DEDUPLICATION & CONTEXT PREPARATION
    context_list = []
    seen_passages = set()
    
    # Dynamic threshold based on query type
    distance_threshold = 0.85 if "program" in query_lower or "study" in query_lower else 0.90
    
    for p, m, d in zip(all_passages, all_metadatas, all_distances):
        if d < distance_threshold:
            # Use first 200 chars as fingerprint
            fingerprint = p.strip()[:200]
            if fingerprint in seen_passages:
                continue
            seen_passages.add(fingerprint)
            
            chunk_url = m.get("article_link", "")
            header = m.get("header_path", "")
            
            # Include header context if available
            context_entry = f"Source URL: {chunk_url}"
            if header:
                context_entry += f"\nSection: {header}"
            context_entry += f"\n{p.strip()}"
            
            context_list.append(context_entry)

    if not context_list:
        return "Unfortunately, I cannot find information about your query in the knowledge base. Please contact the student office for additional information."

    context = "\n\n".join(context_list)
    query_oneline = user_query.replace("\n", " ")

    # 6. OPTIMIZED PROMPT
    prompt = f"""You are a kind, precise, and informative chatbot for the **Faculty of Croatian Studies**. Your task is to answer questions about the faculty.

**CRITICAL RULES:**
1. Use ONLY the provided documentation (CONTEXT).
2. Respond in **English**.
3. **GROUPING:** If the same person teaches multiple courses, mention their name ONLY ONCE. Group programs by level.
4. **LINKS:** Display courses, emails, and programs as Markdown links [Name](URL).
5. **NO INTRODUCTION:** Start directly with the relevant answer without phrases like "Of course...".

**FORMATTING:**
* First line: **Source:** The link that seems most relevant from the documentation.
* Blank line after source.
* **Bold text** for key terms.
* Lists (bullet points) for enumeration.
* If the provided link is in this format, e.g., for Email: E-mail: [idzinic@fhs.hr](javascript:startMail('qvvmva@pus.feu');)
    - display as a plain link without the javascript part, e.g., E-mail: idzinic@fhs.hr
    - remove the javascript part from the link for every link
**AVAILABLE DOCUMENTATION:**
{context}

**USER QUESTION:** {query_oneline}

**ANSWER:**"""
    
    # 7. GENERATE RESPONSE
    answer = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=prompt,
        config={
            "max_output_tokens": 2048,
            "temperature": 0.1,
            "top_p": 0.9
        }
    )
    
    return answer.text.strip()