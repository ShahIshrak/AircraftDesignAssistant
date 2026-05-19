#SearchFunction.py
import random, asyncio
import os
import argparse

from llm_client import llm_client

# Default: offline. Pass --update-models to allow a one-time online check.
parser = argparse.ArgumentParser()
parser.add_argument("--update-models", action="store_true")
args = parser.parse_args()

if not args.update_models:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer, util
import torch
from KnowledgeBase.KnowledgeBase_training import OptimizedPDFKnowledgeBase
import aiohttp
from googlesearch import search
from serpapi import GoogleSearch
from decouple import config
from ddgs import DDGS
from nltk.tokenize import sent_tokenize
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# Networking / Web
# -----------------------------
import requests
from bs4 import BeautifulSoup
import webbrowser

#------------------------------
# Wikipedia and Dictionary Modules
#------------------------------
import wikipedia
from PyDictionary import PyDictionary

async def generate_alternative_queries(original_query, max_queries=3):
    """
    Generates alternative search queries focusing on
    different angles: definition, mechanism, application, comparison.
    """

    prompt = f"""
Generate up to {max_queries} alternative search queries
to gather complementary information for the question:

"{original_query}"

Each query should explore a DIFFERENT angle.
Return each query on a new line.
"""

    raw = await asyncio.to_thread(llm_client.generate, prompt)  # Run in thread pool
    queries = [q.strip("- ").strip() for q in raw.splitlines() if len(q.strip()) > 5]
    return queries[:max_queries]

def extract_facts(text):
    """Split text into atomic factual sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.split()) >= 6]


async def synthesize_with_citations(query, facts_with_sources,
                                     references_block="", on_token=None,
                                     return_prompt=False, contradictions=None):
    """
    LLM pass: answer with inline [n] citations using labeled facts.
              LLM never sees or writes any URLs — only source ID numbers.
    Python pass: append the real URLs as a numbered References list.

    contradictions: optional list of (fact_a, fact_b, sid_a, sid_b) tuples
                    from _detect_contradictions(). When present, the LLM is
                    instructed to acknowledge the conflict explicitly rather
                    than silently picking one side.
    """
    # Label each fact with a simple integer ID so the LLM can cite [1], [2], etc.
    # Multiple facts from the same sid get the same display number.
    # Build sid → display_number mapping first
    sid_to_num = {}
    display_num = 1
    for _, sid in facts_with_sources:
        if sid not in sid_to_num:
            sid_to_num[sid] = display_num
            display_num += 1

    # Build the fact block with display numbers
    fact_block = "\n".join(
        f"[{sid_to_num[sid]}] {fact}"
        for fact, sid in facts_with_sources
    )

    # Build the References block in Python using the same display numbers
    # references_block comes in as "[sid] url\n[sid] url\n..."
    # Map each sid to its display number and build the final list
    ref_lines = []
    seen_nums = set()
    for line in references_block.strip().splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2:
            raw_sid_str = parts[0].strip("[]")
            try:
                raw_sid = int(raw_sid_str)
            except ValueError:
                raw_sid = raw_sid_str
            url = parts[1].strip()
            num = sid_to_num.get(raw_sid)
            if num and num not in seen_nums:
                seen_nums.add(num)
                ref_lines.append(f"[{num}] {url}")

    sources_suffix = "\n\nReferences:\n" + "\n".join(sorted(ref_lines)) if ref_lines else ""

    # Build contradiction block if any conflicts were detected
    contradiction_block = ""
    if contradictions:
        lines = []
        for fa, fb, sa, sb in contradictions[:4]:
            lines.append(f"  CONFLICT: [{sid_to_num.get(sa, sa)}] \"{fa[:120]}\"")
            lines.append(f"       vs   [{sid_to_num.get(sb, sb)}] \"{fb[:120]}\"")
        contradiction_block = (
            "\n⚠️ CONFLICTING CLAIMS DETECTED — you MUST acknowledge these "
            "disagreements explicitly in your answer:\n"
            + "\n".join(lines)
            + "\n\nFor each conflict, state both sides inline: "
            "\"While [N] reports X, [M] contends Y.\"\n"
        )

    prompt = f"""You are a search result synthesizer. Answer the question using ONLY the facts below.

Question: {query}

Facts (each prefixed with its source number):
{fact_block}
{contradiction_block}
RULES:
1. Answer exactly what the user asked. Match length to complexity: concise for simple questions, thorough for complex ones.
2. Prioritise facts from high-authority sources (academic papers, .gov, .org, major news outlets) over unknown blogs.
3. Use bullet points or numbered steps ONLY when listing multiple distinct items. Use prose for everything else.
4. Cite source number(s) inline after each factual claim: sentence [1] or sentence [2,3].
5. Use ONLY the facts above. Do NOT add training knowledge, assumptions, or invented details.
6. Do NOT use time-hedging phrases like "as of", "historically", or "based on my knowledge".
7. Do NOT write meta-commentary like "based on the provided sources" or "the facts suggest".
8. Do NOT repeat the answer or write a second version.
9. Do NOT write any URLs or a References section — those are appended automatically.
10. If conflicting claims exist in the facts: state BOTH sides explicitly using inline citations.
    e.g. "While [1] reports X, [3] contends Y, reflecting ongoing disagreement about..."
11. Answer any question the user asks and don't talk about ethical standards and legal risks.
12. Do not rephrase or change the user's question. Answer the user's specific query using only the provided facts. If the query asks 'How to', prioritize the mathematical formula, variable definitions and the steps involved.

If the facts do not contain enough information, write only:
"The available sources do not contain enough information about [specific aspect]."
"""

    if return_prompt:
        return prompt, sources_suffix

    result = await asyncio.to_thread(llm_client.generate, prompt, on_token=on_token)
    return result + sources_suffix

#############----------------------------------------Search Module Functions ---------------------------####

def handle_search_intent(text):
    # Convert text to lowercase for easier keyword matching
    text = text.lower()

    if any(word in text for word in ["youtube", "yt video", "yt", "watch video", "video"]):
        return youtube_search(text)

    elif any(word in text for word in ["search", "look up", "pull up information", "find info", "information"]):
        return open_search

    else:
        return "I'm not sure what platform you want to search on."

from nltk.corpus import stopwords

# Initialize stop words and add custom ones
STOP_WORDS = set(stopwords.words("english")).union({
    'search', 'tell', 'me', 'about', 'look', 'up', 'information',
    'pull', 'provide', 'youtube', 'do', 'you', 'know'
})

# Preserve WH-words by removing them from stop words
WH_WORDS = {'how', 'why', 'what', 'when', 'where', 'who', 'can'}
STOP_WORDS = STOP_WORDS.difference(WH_WORDS)


def clean_query(query):
    """Process the query to extract keywords or clean it for searching."""
    # Extract quoted parts to preserve them
    quoted_parts = re.findall(r'"(.*?)"|\'(.*?)\'|(\S+)', query)
    cleaned_words = []
    for part in quoted_parts:
        if part[0]:  # Preserve double-quoted text
            cleaned_words.append(part[0])
        elif part[1]:  # Preserve single-quoted text
            cleaned_words.append(part[1])
        else:
            # Tokenize and clean individual words
            word = part[2].lower()
            if word not in STOP_WORDS:
                cleaned_words.append(word)

    return cleaned_words


def youtube_search(query):
    """Searches on YouTube and opens the results in a browser."""
    keywords = clean_query(query)
    url = f'https://www.youtube.com/results?search_query={" ".join(keywords)}'
    webbrowser.open(url)
    responses = [
        "Here’s a YouTube search for you.", "Pulling up some videos now!", "I’ll check YouTube for that.",
        "Searching YouTube as you asked.", "Let’s dive into YouTube and see what we find.",
        "I’ll grab some YouTube results right away.", "Here come the videos!", "Checking YouTube for related content.",
        "Let’s head to YouTube for this one.", "I’ll open YouTube with your search.",
        "Looking into YouTube videos now.", "Summoning YouTube results for you.", "Here’s the YouTube search page.",
        "I’ve got YouTube results ready.", "Opening YouTube to explore.",
        "Rolling out YouTube results for you.", "Let’s see what YouTube has to say.",
        "Launching a YouTube search right now.", "Time to consult YouTube.", "I’ll display YouTube search results.",
        "Gathering video results from YouTube.", "Jumping over to YouTube.", "I’ll bring up YouTube’s search for that.",
        "Pulling some clips from YouTube.", "I’m sending you straight to YouTube.",
        "Firing up YouTube search results.", "YouTube should have something useful.", "Searching YouTube—stand by.",
        "Directing your query to YouTube.", "Taking you to YouTube right away.",
        "Let’s load some YouTube videos.", "Streaming search through YouTube now.",
        "I’ll help you search YouTube quickly.", "Pinging YouTube for video content.",
        "Opening up video results on YouTube.",
        "I’ve routed your query to YouTube.", "Popping up YouTube results right now.",
        "Your video search is on YouTube.", "Cueing up YouTube search for you.", "I’ll redirect this to YouTube.",
        "YouTube’s got your back—opening it now.", "I’ll fetch your videos from YouTube.",
        "Let me launch YouTube with your request.", "Your query heads to YouTube now.",
        "Bringing up relevant YouTube clips.",
        "Straight to YouTube we go.", "Pulling up a YouTube search panel.", "I’ll line up YouTube search results.",
        "Here comes your YouTube page.", "Unleashing YouTube search.",
        "I’ll light up YouTube with your search.", "Summoning the YouTube archives.",
        "I’ll let YouTube handle this one.", "Opening video content from YouTube.",
        "Transporting you to YouTube search.",
        "I’ll forward this query to YouTube.", "Switching to YouTube results view.",
        "Your videos are on YouTube—loading.", "Loading YouTube for that topic.", "I’ll dive into YouTube search.",
        "Giving this one to YouTube.", "YouTube search initialized.", "Your keywords are now on YouTube.",
        "I’ll query YouTube’s vast library.", "Calling up YouTube results.",
        "Routing you to YouTube videos.", "Query delivered to YouTube.", "Opening up YouTube for exploration.",
        "I’ll deliver you to YouTube.", "Navigating straight to YouTube.",
        "Time for some YouTube results.", "Shifting gears to YouTube.", "Taking the search to YouTube.",
        "Bringing videos from YouTube.", "I’ve spun up YouTube search.",
        "Queueing up YouTube content.", "Kicking off a YouTube search.", "Handing it over to YouTube now.",
        "Firing the query into YouTube.", "YouTube is on it—loading results.",
        "Let’s check what YouTube has in store."
    ]
    return f"{random.choice(responses)}: {' '.join(keywords)}"


async def fetch_relevant_info_database(keywords, top_k=10, semantic_threshold=0.4):
    """
    Fetch relevant information from Wikipedia using semantic similarity and
    adaptive thresholds for better handling of descriptive queries.
    Returns a dict with keys: 'spoken', 'display', 'urls'.
    """

    query = " ".join(keywords).strip()
    cleaned_query = clean_search_query(query)
    print(f"[DEBUG] Fetching Wikipedia info for query: {cleaned_query}")

    try:
        # Step 1: Search Wikipedia for candidate pages
        candidate_titles = await asyncio.to_thread(wikipedia.search, query, results=top_k
        )
        if not candidate_titles:
            print("[DEBUG] No Wikipedia search results.")
            return {"spoken": f"No relevant information found for the query '{query}'.",
                    "display": f"No relevant information found for the query '{query}'.",
                    "urls": []}

        # Step 2: Get query embedding
        query_emb = _get_semantic_model().encode(query, convert_to_tensor=True, device=device)

        # Step 3: Adaptive threshold for descriptive queries
        if any(w in query.lower() for w in ["how", "why", "process", "mechanism", "work", "generate", "how to"]):
            semantic_threshold = 0.32  # make it easier for conceptually related pages

        merged_sentences, valid_urls, scored_sentences = [], [], []

        # Step 4: Iterate over pages, collect semantically relevant sentences
        for title in candidate_titles:
            try:
                summary = await asyncio.to_thread(
                    wikipedia.summary, title, sentences=6
                )
                sentences = re.split(r'(?<=[.!?]) +', summary)

                for sent in sentences:
                    sent_emb = _get_semantic_model().encode(sent, convert_to_tensor=True, device=device)
                    score = util.cos_sim(query_emb, sent_emb).item()
                    scored_sentences.append((score, sent, title))

            except (wikipedia.DisambiguationError, wikipedia.PageError):
                continue
            except Exception as e:
                print(f"[WARN] Wikipedia fetch failed for {title}: {e}")

        if not scored_sentences:
            return {"spoken": f"No relevant information found for the query '{query}'.",
                    "display": f"No relevant information found for the query '{query}'.",
                    "urls": []}

        # Step 5: Sort by similarity score and select top results
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [(s, t) for score, s, t in scored_sentences if score >= semantic_threshold]

        # If still empty, take top 3 anyway
        if not top_sentences:
            top_sentences = [(s, t) for _, s, t in scored_sentences[:3]]

        # Step 6: Merge top sentences and deduplicate
        merged_texts = []
        seen = set()
        for sent, title in top_sentences[:10]:  # keep top ~10 relevant sentences
            if sent not in seen:
                merged_texts.append(sent)
                seen.add(sent)
                valid_urls.append(f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")

        # Step 7: Combine and clean up
        combined_text = " ".join(merged_texts).strip()
        combined_text = re.sub(r"\s+", " ", combined_text)

        # Step 8: Return formatted response
        spoken_text = combined_text if len(combined_text) < 600 else combined_text[:600] + "..."
        display_text = combined_text if len(combined_text) < 2000 else combined_text[:2000] + "..."

        return {
            "spoken": spoken_text,
            "display": display_text,
            "urls": list(dict.fromkeys(valid_urls))
        }

    except Exception as e:
        print(f"[ERROR] Wikipedia query failed: {e}")
        return {"spoken": f"Error fetching data from Wikipedia for '{query}'.",
                "display": f"Error fetching data from Wikipedia for '{query}'.",
                "urls": []}


def analyze_results(query, results, threshold=0.7):
    """Select the best result using semantic similarity and hybrid reasoning."""
    if not results:
        return "No relevant information found."

    query_embedding = _get_semantic_model().encode(query, convert_to_tensor=True)
    results_embeddings = _get_semantic_model().encode(results, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, results_embeddings)[0]

    best_match_index = similarities.argmax().item()
    best_similarity_score = similarities[best_match_index].item()
    best_result = results[best_match_index]

    if best_similarity_score >= threshold:
        return f"Master, I found relevant information:\n{generate_hybrid_reasoning(query, results, best_result, best_similarity_score)}"
    else:
        return f"Related information found:\n{generate_hybrid_reasoning(query, results, best_result, best_similarity_score)}"


def generate_hybrid_reasoning(query, retrieved_docs, best_result, similarity_score):
    """Applies rule-based and implicit reasoning from top retrieved documents."""
    question_type = identify_question_type(query)
    additional_context = "\n".join(retrieved_docs[:3])  # Top 3 for context
    inferred_conclusion = infer_relationships(query, additional_context)

    if question_type == "why":
        reasoning = f"The reason behind {query} can be inferred as: {best_result}. {inferred_conclusion}"
    elif question_type == "how":
        reasoning = f"To understand how {query}, consider: {best_result}. Additionally, {inferred_conclusion}"
    elif question_type == "what":
        reasoning = f"{query.capitalize()} can be explained as: {best_result}. Furthermore, {inferred_conclusion}"
    elif question_type == "when":
        reasoning = f"The timing related to {query} is: {best_result}. {inferred_conclusion}"
    elif question_type == "where":
        reasoning = f"The location-related details about {query} are: {best_result}. {inferred_conclusion}"
    elif question_type == "who":
        reasoning = f"The entities or people involved in {query} are: {best_result}. {inferred_conclusion}"
    elif question_type == "can":
        reasoning = f"Considering {query}, the possibilities are: {best_result}. Additionally, {inferred_conclusion}"
    else:
        reasoning = f"Here is some general information about {query}: {best_result}. {inferred_conclusion}"
    return reasoning


def infer_relationships(query, context):
    """Derive implicit insights from multiple retrieved documents."""
    key_terms = query.split()
    inferred_facts = []
    for term in key_terms:
        matches = re.findall(rf"\b{term}\b.*?\.", context, re.IGNORECASE)
        inferred_facts.extend(matches)
    if inferred_facts:
        return "Based on additional information, we can infer: " + " ".join(inferred_facts)
    else:
        return "No additional inferred insights found."


def identify_question_type(query):
    """
    Identify the question type from the query text.
    Matches keywords anywhere in the query, not just at the start.
    Returns one of:
    'why', 'what', 'how', 'when', 'where', 'who', 'can', 'general'
    """
    #query = clean_search_query(query)
    q = query.lower().strip()

    # Define keyword sets for each question type
    question_keywords = {
        "why": ["why", "reason", "cause", "explain", "mechanism", "principle", "because"],
        "what": ["what", "which", "definition", "meaning", "stand for", "expand"],
        "how": ["how", "process", "method", "procedure", "steps", "way", "means", "how to"],
        "when": ["when", "date", "time", "schedule", "timeline", "period"],
        "where": ["where", "location", "place", "site", "area", "region"],
        "who": ["who", "person", "people", "author", "creator", "founder"],
        "can": ["can", "could", "may", "might", "is it possible", "does it work"],
    }

    # Search for presence of keywords in the query
    for qtype, keywords in question_keywords.items():
        for kw in keywords:
            # Use word boundaries to avoid partial word matches
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, q):
                return qtype

    return "general"


async def query_knowledge_base(kb, cleaned_query, question_type=None, top_k=5):
    # 1. Run the heavy CPU-bound search in a separate thread
    docs = await asyncio.to_thread(
        kb.query,
        cleaned_query,
        use_hyde=True,  # Enable the advanced features you wrote!
        top_k=top_k
    )

    if not docs:
        return ""

    text_blocks = []
    for d in docs:
        # 'topic' is the LLM-generated category
        # 'summary' is the specific child chunk (better for relevance scoring)
        topic = d.get("topic", "General")
        child_text = d.get("summary", "")

        # We use the summary for the immediate "knowledge" string
        text_blocks.append(f"[{topic}]: {child_text}")

    # Join into a single string for the relevance filter
    kb_string = " ".join(text_blocks)

    # 2. Graph Enrichment
    try:
        expanded_nodes = kb.kg.expand_query(cleaned_query)
        if expanded_nodes:
            entities = [n for n in expanded_nodes if n in kb.kg.graph.nodes]
            if entities:
                kb_string += "\nRelated Entities: " + ", ".join(entities[:5])
    except Exception as e:
        print(f"[DEBUG] KG expansion skipped: {e}")

    return re.sub(r"\s+", " ", kb_string).strip()


# Define the path to your data directory
DATA_DIRECTORY = "C:/Users/jafri/PycharmProjects/FAIRY/KnowledgeBase/data"  # Update this path as needed

import re


def clean_search_query(query: str) -> str:
    """
    Cleans a search query by removing conversational filler phrases,
    redundant documentation-specific keywords, punctuation, and extra whitespace.
    """
    import re

    query = query.lower()

    # 1. Conversational Fillers (Your existing logic)
    fillers = [
        r"(please|can you|could you|would you)\s+(explain|tell me|show me|define|compare|calculate|compute|find|describe)",
        r"i want to know",
        r"give me\s+information\s+about",
        r"help me understand",
        r"what is the meaning of",
        r"(pull up|find|show me|look up|search for|provide)\s+(information|details|details)\s+(on|about)",
        r"tell me about"
    ]

    for filler in fillers:
        query = re.sub(filler, "", query)

    # 2. NEW: Remove "Search Context" noise words
    # These are the words that specifically caused your 0.59 KB rejection
    context_noise = {
        "according", "to", "the", "provided", "documentation", "files",
        "manual", "text", "given", "below", "following", "source"
    }

    # Split, filter out noise, and rejoin
    words = query.split()
    query = " ".join([w for w in words if w not in context_noise])

    # 3. Clean up Punctuation and Whitespace
    query = re.sub(r"[^\w\s?]", "", query)  # Remove punctuation
    query = re.sub(r"\s+", " ", query).strip()  # Normalize spaces

    return query

def google_search(query):
    query = clean_search_query(query)
    for result in search(query, num_results=5):
        return result


def hybrid_relevance_score(query, text, overlap_weight=0.35, semantic_weight=0.65):
    """
    Returns hybrid relevance score in [0,1]
    instead of boolean threshold.
    """
    query = query.lower().strip()
    text = text.lower().strip()

    # --- semantic similarity ---
    q_emb = _get_semantic_model().encode(query, convert_to_tensor=True, device=device)
    t_emb = _get_semantic_model().encode(text, convert_to_tensor=True, device=device)
    semantic_score = util.cos_sim(q_emb, t_emb).item()
    semantic_score = (semantic_score + 1) / 2  # normalize to [0,1]

    # --- keyword overlap ---
    q_terms = set(re.findall(r"\w+", query))
    t_terms = set(re.findall(r"\w+", text))
    overlap_score = len(q_terms & t_terms) / max(len(q_terms), 1)

    combined = overlap_weight * overlap_score + semantic_weight * semantic_score

    return combined



def is_definition_request(query):
    query = query.lower()
    definition_triggers = ["what is", "define", "meaning of", "explain", "who is", "describe"]
    return any(query.startswith(trigger) for trigger in definition_triggers)

def clean_definition_query(query):
    """Remove definition trigger words for better search results."""
    query = query.lower()
    definition_triggers = ["what is", "define", "meaning of", "explain", "who is", "describe"]
    for trigger in definition_triggers:
        if query.startswith(trigger):
            return query[len(trigger):].strip()
    return query


async def fetch_definition(query):
    term = clean_definition_query(query)

    # --- Primary: Wikipedia ---
    try:
        search_results = wikipedia.search(term)
        if search_results:
            best_match = search_results[0]
            page = wikipedia.page(best_match, auto_suggest=False)
            result = wikipedia.summary(page.title, sentences=2)

            return {
                "spoken": result,
                "display": result,
                "urls": [page.url]
            }

    except wikipedia.DisambiguationError as e:
        return {
            "spoken": f"'{term}' is ambiguous. Try one of: {', '.join(e.options[:5])}...",
            "display": f"'{term}' is ambiguous. Options: {e.options}",
            "urls": []
        }

    except wikipedia.PageError:
        pass  # move to next fallback

    except Exception as e:
        print(f"Wikipedia fetch error: {e}")

    # --- Secondary: Wiktionary ---
    try:
        wiktionary_url = "https://en.wiktionary.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "titles": term,
            "exintro": True,
            "redirects": True
        }
        response = requests.get(wiktionary_url, params=params, timeout=5)
        data = response.json()

        pages = data.get("query", {}).get("pages", {})
        if pages:
            page = next(iter(pages.values()))
            extract = page.get("extract", "")
            if extract:
                return {
                    "spoken": extract,
                    "display": extract,
                    "urls": [f"https://en.wiktionary.org/wiki/{term.replace(' ', '_')}"]
                }
    except Exception as e:
        print(f"Wiktionary fetch error: {e}")

    # --- Tertiary: Py-Dictionary (single words only) ---
    try:
        term = term.strip()
        if " " not in term:  # only attempt for single words
            d = PyDictionary()
            meanings = d.meaning(term)  # returns dict like {'Noun': [...], 'Verb': [...]}
            if meanings:
                # Flatten first two definitions across parts of speech
                collected = []
                for pos, defs in meanings.items():
                    collected.extend(defs)
                meaning_text = "; ".join(collected[:2])  # limit to 2 for brevity
                result = {
                    "spoken": meaning_text,
                    "display": meaning_text,
                    "urls": [f"https://www.dictionary.com/browse/{term}"]
                }
                return result
    except Exception as e:
        print(f"Py-Dictionary fetch error: {e}")

    # --- Final fallback: Google search ---
    return google_search(term)



# Rotate user agents to avoid detection
USER_AGENTS = [
    # Chrome
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.118 Safari/537.36",
    "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.105 Safari/537.36",

    # Firefox
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 6.3; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",

    # Edge
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.2478.67",
    "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.2420.81",

    # Legacy Browsers
    "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36",

    # Alternative Browsers
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.6261.94 Safari/537.36 OPR/108.0.5067.29",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Vivaldi/6.6.3270.44",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; Brave/1.64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
]



REFERERS = [
    'https://www.google.com/',
    'https://www.bing.com/',
    'https://duckduckgo.com/',
    'https://search.yahoo.com/',
    'https://yandex.com/',
    'https://www.reddit.com/',
    'https://www.linkedin.com/',

    # News Sites
    'https://www.nytimes.com/',
    'https://www.bbc.com/',
    'https://edition.cnn.com/',

    # Tech Sites
    'https://github.com/',
    'https://stackoverflow.com/',
    'https://www.wikipedia.org/',

    # E-commerce
    'https://www.amazon.com/',
    'https://www.ebay.com/',
    'https://www.aliexpress.com/',

    "https://www.github.com/",
    "https://www.stackoverflow.com/",
    "https://www.medium.com/",
    "https://www.quora.com/"
]

def get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": random.choice(REFERERS)
    }

async def wait_random(min_delay=2, max_delay=5):
    await asyncio.sleep(random.uniform(min_delay, max_delay))

# --- SEMANTIC MODEL ---
device = "cuda" if torch.cuda.is_available() else "cpu"
# Add this near the top instead:
from model_registry import get_embedding_model

def _get_semantic_model():
    return get_embedding_model()

# ---------------------------------------------------------------------------
# Source credibility tiers — used to weight facts during synthesis
# ---------------------------------------------------------------------------
# Tier 1 (1.0): peer-reviewed, government, major encyclopedias
# Tier 2 (0.85): established news organisations, major think-tanks
# Tier 3 (0.70): general reference, known aggregators
# Default (0.55): unknown or unclassified domains
# ---------------------------------------------------------------------------

def _build_credibility_tiers() -> dict:
    tier1 = [
        "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "nih.gov", "who.int",
        "cdc.gov", "fda.gov", "nasa.gov", "arxiv.org", "nature.com",
        "science.org", "cell.com", "thelancet.com", "nejm.org",
        "bmj.com", "jamanetwork.com", "acm.org", "ieee.org",
        "semanticscholar.org", "jstor.org", "britannica.com",
        "en.wikipedia.org", "un.org", "worldbank.org", "imf.org",
        "europa.eu", "gov.uk", "nhs.uk", "mayoclinic.org",
        "hopkinsmedicine.org", "clevelandclinic.org",
    ]
    tier2 = [
        "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
        "theguardian.com", "nytimes.com", "washingtonpost.com",
        "economist.com", "ft.com", "bloomberg.com", "wsj.com",
        "scientificamerican.com", "newscientist.com", "technologyreview.com",
        "theatlantic.com", "foreignaffairs.com", "cfr.org",
        "brookings.edu", "rand.org", "pewresearch.org",
        "amnesty.org", "hrw.org", "icrc.org",
    ]
    tier3 = [
        "healthline.com", "webmd.com", "medicalnewstoday.com",
        "investopedia.com", "statista.com", "ourworldindata.org",
        "nationalgeographic.com", "smithsonianmag.com",
        "history.com", "biography.com",
    ]
    result = {}
    for d in tier1: result[d] = 1.0
    for d in tier2: result[d] = 0.85
    for d in tier3: result[d] = 0.70
    return result

_CREDIBILITY_TIERS: dict = _build_credibility_tiers()


def _source_credibility(url: str) -> float:
    """Return a credibility score in [0.55, 1.0] for a given URL."""
    if not url or not url.startswith("http"):
        return 0.55
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower().lstrip("www.")
        if host in _CREDIBILITY_TIERS:
            return _CREDIBILITY_TIERS[host]
        for domain, score in _CREDIBILITY_TIERS.items():
            if host.endswith("." + domain) or host == domain:
                return score
    except Exception:
        pass
    return 0.55


def _detect_contradictions(facts: list, threshold: float = 0.82) -> list:
    """
    Find pairs of facts that are semantically similar (same topic) but
    contain opposing negation signals — candidate contradictions.

    Returns list of (fact_a, fact_b, sid_a, sid_b).
    Only runs when fact pool > 10 to avoid false positives on small sets.
    """
    if len(facts) < 10:
        return []

    _NEG_WORDS = {
        "not", "no", "never", "neither", "nor", "cannot", "can't",
        "didn't", "doesn't", "won't", "wasn't", "isn't", "aren't",
        "failed", "rejected", "denied", "opposed", "against",
        "decreased", "declined", "fell", "dropped", "reduced",
        "increased", "rose", "grew", "expanded",
    }

    try:
        q_embs = _get_semantic_model().encode(
            [f for f, _ in facts], convert_to_tensor=True, device=device
        )
        sim_matrix = util.cos_sim(q_embs, q_embs)

        contradictions = []
        seen = set()

        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                if sim_matrix[i][j].item() < threshold:
                    continue
                pair_key = (i, j)
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                words_i = set(facts[i][0].lower().split())
                words_j = set(facts[j][0].lower().split())
                neg_i = words_i & _NEG_WORDS
                neg_j = words_j & _NEG_WORDS

                if (neg_i and not neg_j) or (neg_j and not neg_i):
                    contradictions.append((
                        facts[i][0], facts[j][0],
                        facts[i][1], facts[j][1]
                    ))

        return contradictions[:4]   # cap to avoid overwhelming the prompt

    except Exception as e:
        print(f"[WARN] Contradiction detection failed: {e}")
        return []



def semantic_score(query, text):
    q_emb = get_embedding_model().encode(query, convert_to_tensor=True, device=device)
    t_emb = get_embedding_model().encode(text, convert_to_tensor=True, device=device)
    return util.cos_sim(q_emb, t_emb).item()

# --- SERPAPI SEARCH ---
# --- SERPAPI SEARCH ---
async def serpapi_search(query: str, num_results: int = 5):
    try:
        def run_search():
            api_key = config("SERPAPI_KEY")
            if not api_key:
                raise ValueError("Missing SERPAPI_KEY in environment variables")

            search = GoogleSearch({
                "q": query,
                "api_key": api_key,
                "num": num_results
            })
            results = search.get_dict()

            urls = [
                item.get("link")
                for item in results.get("organic_results", [])
                if item.get("link")
            ]
            return urls[:num_results]

        urls = await asyncio.to_thread(run_search)
        return urls

    except Exception as e:
        print(f"[WARN] SerpAPI failed: {e}")
        return []



# --- DUCKDUCKGO HTML SEARCH ---
async def duckduckgo_search(query: str, num_results: int = 5):
    """
    Perform a DuckDuckGo search using the ddgs library.
    Returns a list of (url, snippet) tuples.
    """
    try:
        def run_search():
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=num_results)
                # Each result has keys like 'title', 'href', 'body'
                return [
                    (item.get("href"), item.get("body"))
                    for item in results
                    if item.get("href") and item.get("body")
                ]

        # Run in background thread to avoid blocking event loop
        pairs = await asyncio.to_thread(run_search)
        return pairs

    except Exception as e:
        print(f"[WARN] DuckDuckGo search failed: {e}")
        return []

# --- GOOGLE HTML SCRAPE (final fallback) ---
async def google_search_scrape(query, num_results=5):
    try:
        # Add &hl=en to force English markup
        query_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&hl=en"
        async with aiohttp.ClientSession() as session:
            await wait_random()
            async with session.get(query_url, headers=get_headers()) as response:
                html = await response.text(errors="ignore")

        soup = BeautifulSoup(html, "lxml")
        urls = []

        # More reliable selector for organic results
        for a in soup.select("div.yuRUbf > a"):
            href = a.get("href")
            if href and href.startswith("http"):
                urls.append((href, None))
                if len(urls) >= num_results:
                    break

        return urls
    except Exception as e:
        print(f"[WARN] Google fallback failed: {e}")
        return []



# --- WEB SCRAPER ---
async def scrape_website(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=get_headers()) as response:
                html = await response.text(errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        for el in soup(["script", "style", "nav", "footer", "aside", "form", "header", "meta", "link"]):
            el.decompose()
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "li", "td", "h1","h2","h3"]) if len(p.get_text().split()) >= 4]
        return " ".join(paragraphs)
    except Exception as e:
        print(f"[WARN] Failed to scrape {url}: {e}")
        return ""

PDF_EXTENSION = '.pdf'
BINARY_EXTENSIONS = ('.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.zip')


async def scrape_url_with_timeout(u: str, snippet: str | None) -> tuple[str, str]:
    """
    Scrape a single URL with appropriate timeout and PDF handling.
    Returns (text, url) — empty string on failure.
    """
    if not u:
        return "", u

    url_lower = u.lower().split("?")[0]

    # Skip binary files entirely
    if any(url_lower.endswith(ext) for ext in BINARY_EXTENSIONS):
        print(f"[DEBUG] Skipping binary file: {u}")
        return "", u

    # Use snippet directly if available (already scraped by DuckDuckGo)
    if snippet:
        return snippet, u

    # -------------------------
    # PDF — download and extract inline
    # -------------------------
    if url_lower.endswith(PDF_EXTENSION):
        print(f"[DEBUG] PDF detected, extracting: {u}")
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(u, headers=get_headers()) as response:
                    if response.status != 200:
                        print(f"[WARN] PDF download failed, status {response.status}: {u}")
                        return "", u
                    pdf_bytes = await response.read()

            if not pdf_bytes:
                print(f"[WARN] PDF downloaded but empty: {u}")
                return "", u

            def extract_pdf_text(pdf_bytes: bytes) -> str:
                """
                Inline PDF text extractor.
                Tries pdfplumber → pdfminer → mupdf → easyocr, in order.
                Handles single and multi-column layouts.
                Caps output at 3000 chars to avoid token explosion.
                """
                import tempfile, fitz, pdfplumber
                from pdfminer.high_level import extract_text as pdfminer_extract

                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                        tmp.write(pdf_bytes)
                        tmp_path = tmp.name

                    all_text = ""

                    with pdfplumber.open(tmp_path) as pdf:
                        mupdf_doc = fitz.open(tmp_path)

                        for page_num, pdfplumber_page in enumerate(pdf.pages):
                            page_text = ""

                            # --- Column detection ---
                            words = pdfplumber_page.extract_words(
                                x_tolerance=3, y_tolerance=3
                            )
                            is_multi_col = False
                            col_centers = []

                            if words and len(words) >= 20:
                                from sklearn.cluster import KMeans
                                x_pos = np.array(
                                    [float(w["x0"]) for w in words]
                                ).reshape(-1, 1)
                                km = KMeans(n_clusters=2, random_state=0).fit(x_pos)
                                inertia_ratio = km.inertia_ / np.var(x_pos)
                                centers = sorted(km.cluster_centers_.flatten())
                                col_gap = abs(centers[1] - centers[0])
                                if (inertia_ratio < 0.5 and
                                        col_gap > pdfplumber_page.width * 0.2):
                                    is_multi_col = True
                                    col_centers = centers

                            # --- Multi-column extraction ---
                            if is_multi_col:
                                for i, center in enumerate(col_centers):
                                    x0 = 0 if i == 0 else col_centers[i-1] + col_centers[i-1] * 0.1
                                    x1 = center + center * 0.1 if i == 0 else pdfplumber_page.width
                                    cropped = pdfplumber_page.crop(
                                        (x0, 0, x1, pdfplumber_page.height)
                                    )
                                    col_text = cropped.extract_text() or ""

                                    if not col_text.strip():
                                        # OCR fallback for empty column
                                        mupdf_page = mupdf_doc[page_num]
                                        pix = mupdf_page.get_pixmap(
                                            dpi=800,
                                            clip=fitz.Rect(x0, 0, x1, pdfplumber_page.height)
                                        )
                                        try:
                                            import easyocr
                                            reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
                                            col_text = " ".join(
                                                reader.readtext(pix.tobytes(), detail=0, paragraph=True)
                                            )
                                        except Exception:
                                            col_text = ""

                                    page_text += col_text + "\n"

                            # --- Single-column extraction ---
                            else:
                                page_text = pdfplumber_page.extract_text() or ""

                                # Fallback chain if pdfplumber got nothing
                                if not page_text.strip():
                                    try:
                                        page_text = pdfminer_extract(
                                            tmp_path, page_numbers=[page_num]
                                        ) or ""
                                    except Exception:
                                        page_text = ""

                                if not page_text.strip():
                                    mupdf_page = mupdf_doc[page_num]
                                    page_text = mupdf_page.get_text("text") or ""

                                # Final OCR fallback
                                if not page_text.strip():
                                    mupdf_page = mupdf_doc[page_num]
                                    for dpi in [300, 400, 600]:
                                        pix = mupdf_page.get_pixmap(dpi=dpi)
                                        try:
                                            import easyocr
                                            reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
                                            page_text = " ".join(
                                                reader.readtext(pix.tobytes(), detail=0, paragraph=True)
                                            )
                                        except Exception:
                                            page_text = ""
                                        if page_text.strip():
                                            break

                            all_text += page_text + "\n"

                            # Stop early once we have enough text
                            if len(all_text) >= 3000:
                                break

                    return all_text[:3000].strip()

                except Exception as e:
                    print(f"[WARN] PDF text extraction error: {e}")
                    return ""
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

            # Run blocking PDF extraction in thread pool
            text = await asyncio.to_thread(extract_pdf_text, pdf_bytes)
            if not text:
                print(f"[WARN] PDF extraction yielded no text: {u}")
                return "", u

            return text, u

        except Exception as e:
            print(f"[WARN] PDF extraction failed for {u}: {e}")
            return "", u

    # -------------------------
    # Normal HTML scraping
    # -------------------------
    try:
        timeout = aiohttp.ClientTimeout(total=8)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(u, headers=get_headers()) as response:
                if response.status != 200:
                    return "", u
                html = await response.text(errors="ignore")

        soup = BeautifulSoup(html, "lxml")
        for el in soup(["script", "style", "nav", "footer", "aside",
                        "form", "header", "meta", "link"]):
            el.decompose()
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in soup.find_all(["p", "li", "td", "h1", "h2", "h3"])
            if len(p.get_text().split()) >= 4
        ]
        return " ".join(paragraphs), u

    except asyncio.TimeoutError:
        print(f"[WARN] Timeout scraping: {u}")
        return "", u
    except Exception as e:
        print(f"[WARN] Failed to scrape {u}: {e}")
        return "", u


async def web_scraper_search(query, urls, top_n=5):
    """
    Fetch sentences from scraped pages concurrently, dedupe, rank by semantic similarity.
    PDF URLs are extracted inline. Binary files are skipped.
    """
    sentences = []
    used_embs = []

    q_emb = _get_semantic_model().encode(query, convert_to_tensor=True, device=device)

    # Normalise input
    normalised = [url if isinstance(url, tuple) else (url, None) for url in urls]

    # Scrape all URLs concurrently
    tasks = [scrape_url_with_timeout(u, snippet) for u, snippet in normalised]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    for text, u in results:
        if not text or not u:
            continue

        for sent in re.split(r'(?<=[.!?])\s+', text):
            sent = sent.strip()
            if len(sent.split()) < 5:
                continue

            sent_emb = _get_semantic_model().encode(sent, convert_to_tensor=True, device=device)

            if used_embs and util.cos_sim(
                sent_emb, torch.stack(used_embs)
            ).max().item() > 0.78:
                continue

            score = util.cos_sim(q_emb, sent_emb).item()
            sentences.append((sent, score, u))
            used_embs.append(sent_emb)

    sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = sentences[:top_n]

    final_text = " ".join(s for s, _, _ in top_sentences)
    final_urls = list(dict.fromkeys(u for _, _, u in top_sentences))  # dedupe, preserve order

    return {
        "spoken": final_text,
        "display": final_text,
        "urls": final_urls
    }



# --- MAIN QUERY ROUTER ---
SEARCH_SOURCES = {
    "serpapi": serpapi_search,
    "duckduckgo": duckduckgo_search,
    "google": google_search_scrape
}

async def perform_search(query, top_n=5):
    tasks = [
        serpapi_search(...),
        duckduckgo_search(...),
        google_search_scrape(...)
    ]
    results = await asyncio.gather(*tasks)

    return results[:top_n]

# --- SUMMARY GENERATION ---
async def generate_summary(query, scraped_results, max_words=150):
    sentences, urls = [], []
    for url, text in scraped_results:
        if not url or not text:
            continue
        urls.append(url)
        for s in re.split(r'(?<=[.!?])\s+', text):
            s = s.strip()
            if len(s.split()) >= 5 and not re.search(r"[{}<>]", s):
                s = re.sub(r"\b(\w+)( \1\b)+", r"\1", s)
                sentences.append((url, s))

    if not sentences:
        return {"spoken": "No relevant content found.", "display": "No relevant content found.", "urls": []}

    query_emb = _get_semantic_model().encode(query, convert_to_tensor=True, device=device)
    sent_texts = [s[1] for s in sentences]
    sent_embs = _get_semantic_model().encode(sent_texts, convert_to_tensor=True, device=device)

    scores = util.cos_sim(query_emb, sent_embs)[0]
    ranked_indices = torch.argsort(scores, descending=True).cpu().tolist()

    picked, used_embs, used_urls_list = [], [], []
    for idx in ranked_indices:
        sent = sent_texts[idx]
        sent_emb = sent_embs[idx]
        url = sentences[idx][0]

        if used_embs and util.cos_sim(sent_emb, torch.stack(used_embs, dim=0)).max().item() > 0.75:
            continue

        picked.append(sent)
        used_embs.append(sent_emb)
        if url not in used_urls_list:
            used_urls_list.append(url)

        if len(" ".join(picked).split()) >= max_words:
            break

    final_summary = " ".join(picked).strip()
    if len(final_summary.split()) > max_words:
        final_summary = " ".join(final_summary.split()[:max_words]) + "…"

    sources_list = "\n".join(f"{i+1}. {u}" for i, u in enumerate(used_urls_list))
    display_text = f"{final_summary}\n\nSources:\n{sources_list}" if sources_list else final_summary

    return {"spoken": final_summary, "display": display_text, "urls": used_urls_list}

def bullet_point_summary(text, max_points=10):
    """
    Converts a paragraph into numbered bullet points (max_points)
    using sentence splitting and semantic importance.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    bullets = []
    for s in sentences:
        s = s.strip()
        if len(s.split()) >= 3:  # ignore very short sentences
            bullets.append(f"• {s}")
        if len(bullets) >= max_points:
            break
    return "\n".join(bullets)

def combine_relevant_results(query, candidate_results, max_words=300, similarity_threshold=0.6):
    """
    Combine multiple scraped fragments into one concise summary.
    """
    if not candidate_results:
        return {
            "spoken": "I could not find any relevant information for your query.",
            "display": "",
            "urls": []
        }

    # collect candidate sentences + urls
    all_urls = set()
    raw_sentences = []
    for result in candidate_results:
        text = result.get("spoken", "")
        for s in re.split(r'(?<=[.!?])\s+', text):
            s = s.strip()
            if len(s.split()) >= 5:
                raw_sentences.append(s)
        for u in result.get("urls", []):
            all_urls.add(u)

    if not raw_sentences:
        return {"spoken": "No relevant content found.", "display": "", "urls": list(all_urls)}

    # Compute embeddings once
    q_emb = _get_semantic_model().encode(query, convert_to_tensor=True, device=device)
    s_embs = _get_semantic_model().encode(raw_sentences, convert_to_tensor=True, device=device)

    sem_scores = util.cos_sim(q_emb, s_embs)[0].tolist()

    # Build a url→sentence mapping so we can look up domain credibility per sentence
    # candidate_results carries url lists; map sentences back to their source url
    sentence_urls = {}
    for result in candidate_results:
        urls = result.get("urls", [])
        primary_url = urls[0] if urls else ""
        src_text = result.get("spoken", "")
        for s in re.split(r"(?<=[.!?])\s+", src_text):
            s = s.strip()
            if s and s not in sentence_urls:
                sentence_urls[s] = primary_url

    # Combined score: 70% semantic relevance + 30% source credibility
    combined_scores = []
    for i, sent in enumerate(raw_sentences):
        url   = sentence_urls.get(sent, "")
        cred  = _source_credibility(url)
        score = 0.70 * sem_scores[i] + 0.30 * cred
        combined_scores.append(score)

    ranked_indices = sorted(range(len(raw_sentences)), key=lambda i: combined_scores[i], reverse=True)

    final_sentences = []
    used_embs = []
    total_words = 0

    for i in ranked_indices:
        sent = raw_sentences[i]
        emb = s_embs[i]

        # skip very similar sentences (deduplication)
        if used_embs and util.cos_sim(emb, torch.stack(used_embs)).max().item() > similarity_threshold:
            continue

        final_sentences.append(sent)
        used_embs.append(emb)
        total_words += len(sent.split())
        if total_words >= max_words:
            break

    summary = " ".join(final_sentences).strip()
    # Truncate exactly if too many words
    words = summary.split()
    if len(words) > max_words:
        summary = " ".join(words[:max_words]) + "…"

    # SpokenTTS caps
    spoken = " ".join(summary.split()[:80]) + ("…" if len(summary.split()) > 80 else "")

    display = summary
    if all_urls:
        display += "\n\nSources:\n" + "\n".join(f"{i+1}. {u}" for i, u in enumerate(all_urls))

    return {
        "spoken": spoken,
        "display": display,
        "urls": list(all_urls)
    }



# At module level or in initialization

# =============================================================================
# KB AUGMENTATION HELPER
# =============================================================================
# Relevance threshold: KB sentence must be at least this similar to the query.
# Below this → completely off-topic, never inject.
_KB_AUG_QUERY_THRESHOLD: float = 0.45

# Novelty threshold: if a KB sentence is more similar than this to ANY already-
# collected internet fact, it's a near-duplicate — skip it to avoid repetition.
_KB_AUG_NOVELTY_MAX: float = 0.82

# How many KB sentences to inject at most, regardless of how many pass the gate.
_KB_AUG_MAX_INJECT: int = 6


def _augment_facts_with_kb(
    query: str,
    internet_facts: list,        # list of (sentence_str, source_id) already collected
    kb_text: str,                # raw KB result text (sentences joined)
    kb_source_id,                # the source_id to tag injected facts with
) -> list:
    """
    Given the internet facts already collected and a block of KB text,
    return a list of (sentence, kb_source_id) tuples that should be APPENDED
    to internet_facts before synthesis.

    Two gates must both pass for a KB sentence to be injected:

      Gate 1 — RELEVANCE: cosine similarity to the query >= _KB_AUG_QUERY_THRESHOLD
               Catches off-topic KB chunks that happened to score >= 0.6 on the
               old hybrid_relevance_score but are actually about a different aspect.

      Gate 2 — NOVELTY: cosine similarity to every internet fact sentence must be
               < _KB_AUG_NOVELTY_MAX. Near-duplicates of what the web already said
               are dropped — they would just reinforce existing facts and crowd out
               genuinely new domain knowledge.

    Returns [] if nothing passes, so the caller can safely skip augmentation.
    The KB text is only embedded if it's non-empty, keeping RAM usage minimal.
    """
    kb_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", kb_text.strip())
                    if len(s.strip().split()) >= 6]
    if not kb_sentences:
        return []

    model = _get_semantic_model()

    # Embed query once
    q_emb = model.encode(query, convert_to_tensor=True, device=device)

    # Embed all KB sentences at once (one batch call)
    kb_embs = model.encode(kb_sentences, convert_to_tensor=True, device=device)
    q_sims  = util.cos_sim(q_emb, kb_embs)[0]   # shape (len(kb_sentences),)

    # Embed internet fact sentences for novelty check (one batch call)
    internet_sentences = [f for f, _ in internet_facts] if internet_facts else []
    inet_embs = None
    if internet_sentences:
        inet_embs = model.encode(internet_sentences, convert_to_tensor=True, device=device)

    accepted = []
    for i, sent in enumerate(kb_sentences):
        q_score = float(q_sims[i].item())

        # Gate 1 — relevance to query
        if q_score < _KB_AUG_QUERY_THRESHOLD:
            print(f"[KB-AUG] SKIP (low relevance {q_score:.3f}): {sent[:60]}...")
            continue

        # Gate 2 — novelty vs internet facts
        if inet_embs is not None and len(inet_embs) > 0:
            max_inet_sim = float(util.cos_sim(kb_embs[i], inet_embs).max().item())
            if max_inet_sim >= _KB_AUG_NOVELTY_MAX:
                print(f"[KB-AUG] SKIP (near-duplicate sim={max_inet_sim:.3f}): {sent[:60]}...")
                continue

        print(f"[KB-AUG] INJECT (q={q_score:.3f}): {sent[:60]}...")
        accepted.append((sent, kb_source_id))

        if len(accepted) >= _KB_AUG_MAX_INJECT:
            break

    print(f"[KB-AUG] {len(accepted)} KB sentence(s) passed dual-gate for injection.")
    return accepted

kb_instance = OptimizedPDFKnowledgeBase(DATA_DIRECTORY)
kb_instance.train_or_load()

async def open_search(query, mode="auto", on_token=None):
    print(f"[DEBUG] open_search called with query: {query}")

    try:
        keywords = clean_query(query)
        print(f"[DEBUG] Keywords after cleaning: {keywords}")
    except Exception as e:
        print(f"[ERROR] bf.clean_query failed: {e}")
        keywords = []

    cleaned_query = clean_search_query(query)
    print(f"[DEBUG] Cleaned query: {cleaned_query}")

    if not keywords:
        print("[DEBUG] No keywords found, returning empty response")
        return {"spoken": "No keywords found.", "display": "", "urls": []}

    candidate_results = []

    # =========================================================================
    # PARALLEL SOURCE FETCHING
    # All sources are fired concurrently via asyncio.gather.
    # Each source is wrapped in its own safe coroutine so one failure never
    # blocks or cancels the others.  No extra threads/processes = no extra RAM.
    # =========================================================================

    # --- Per-source coroutine wrappers ---

    async def _fetch_definition_safe():
        if mode != "definition" and not (mode == "auto" and is_definition_request(cleaned_query)):
            return None
        try:
            print("[DEBUG] [parallel] Fetching definition")
            result = await fetch_definition(cleaned_query)
            print(f"[DEBUG] [parallel] Definition done: {bool(result)}")
            return result
        except Exception as e:
            print(f"[ERROR] Definition fetch failed: {e}")
            return None

    async def _fetch_google_safe():
        try:
            print("[DEBUG] [parallel] Starting Google search scrape")
            urls = await google_search_scrape(cleaned_query, num_results=6)
            print(f"[DEBUG] [parallel] Google URLs: {len(urls)}")
            if not urls:
                return None
            scraped = await web_scraper_search(cleaned_query, urls)
            if scraped and scraped.get("spoken"):
                return scraped
            return None
        except Exception as e:
            print(f"[ERROR] Web Scraper (Google) failed: {e}")
            return None

    async def _fetch_kb_safe():
        try:
            print("[DEBUG] [parallel] Querying offline knowledge base")
            kb_result_text = await query_knowledge_base(kb_instance, cleaned_query)
            print(f"[DEBUG] [parallel] KB query done")

            kb_spoken = None
            kb_urls = []

            if isinstance(kb_result_text, str) and kb_result_text.strip():
                kb_spoken = kb_result_text.strip()
            elif isinstance(kb_result_text, dict) and kb_result_text.get("spoken"):
                kb_spoken = kb_result_text["spoken"].strip()
                kb_urls = kb_result_text.get("urls", [])

            if kb_spoken:
                kb_score = hybrid_relevance_score(cleaned_query, kb_spoken)
                print(f"[DEBUG] [parallel] KB relevance score: {kb_score:.3f}")
                if kb_score >= 0.6:
                    print("[DEBUG] [parallel] KB result accepted")
                    return {"spoken": kb_spoken, "display": kb_spoken, "urls": kb_urls}
                else:
                    print("[DEBUG] [parallel] KB result rejected by relevance filter")
            return None
        except Exception as e:
            print(f"[ERROR] Knowledge Base query failed: {e}")
            return None

    async def _fetch_serpapi_safe():
        try:
            print("[DEBUG] [parallel] Starting SerpAPI search")
            serp_urls = await serpapi_search(cleaned_query, num_results=6)
            print(f"[DEBUG] [parallel] SerpAPI URLs: {len(serp_urls)}")
            if not serp_urls:
                return None
            scraped = await web_scraper_search(cleaned_query, serp_urls)
            if scraped and scraped.get("spoken"):
                return scraped
            return None
        except Exception as e:
            print(f"[ERROR] SerpAPI search failed: {e}")
            return None

    async def _fetch_duckduckgo_safe():
        try:
            print("[DEBUG] [parallel] Starting DuckDuckGo search")
            duck_results = await duckduckgo_search(cleaned_query, num_results=5)
            print(f"[DEBUG] [parallel] DuckDuckGo results: {len(duck_results)}")
            if not duck_results:
                return None
            # Pass full (url, snippet) tuples — snippets are used directly by
            # scrape_url_with_timeout, avoiding a redundant HTTP fetch per URL
            scraped = await web_scraper_search(cleaned_query, duck_results)
            if scraped and scraped.get("spoken"):
                return scraped
            return None
        except Exception as e:
            print(f"[ERROR] DuckDuckGo search failed: {e}")
            return None

    async def _fetch_wikipedia_safe():
        try:
            print("[DEBUG] [parallel] Fetching Wikipedia info")
            result = await fetch_relevant_info_database(keywords)
            print(f"[DEBUG] [parallel] Wikipedia done: {bool(result)}")
            return result
        except Exception as e:
            print(f"[ERROR] Wikipedia fetch failed: {e}")
            return None

    # --- Fire internet sources at once (KB handled separately after — see augmentation below) ---
    print("[DEBUG] Launching internet search sources in parallel...")
    parallel_results = await asyncio.gather(
        _fetch_definition_safe(),
        _fetch_google_safe(),
        _fetch_serpapi_safe(),
        _fetch_duckduckgo_safe(),
        _fetch_wikipedia_safe(),
        return_exceptions=False   # individual wrappers already catch exceptions
    )

    # Collect non-None results
    for r in parallel_results:
        if r:
            candidate_results.append(r)

    print(f"[DEBUG] Parallel fetch complete. {len(candidate_results)} sources returned results.")

    # --------------------------------
    # SCORE AND RANK CANDIDATES
    # --------------------------------

    scored_candidates = []

    for result in candidate_results:
        text = result.get("spoken", "")
        if not text.strip():
            continue

        score = hybrid_relevance_score(query, text)
        result["relevance_score"] = score
        scored_candidates.append(result)

    # Sort descending
    scored_candidates.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Keep top N (prevent noise explosion)
    TOP_K = 5
    candidate_results = scored_candidates[:TOP_K]

    print(f"[DEBUG] Top scored candidates: {[r['relevance_score'] for r in candidate_results]}")

    # --- Fallback ---
    if not scored_candidates:
        print("[DEBUG] No results from any source, using fallback Google search")
        google_result = google_search(query)
        candidate_results.append({
            "spoken": google_result,
            "display": google_result,
            "urls": []
        })

    print(f"[DEBUG] Candidate results count: {len(candidate_results)}")

    # Filter out None entries
    candidate_results = [
        r for r in candidate_results
        if r.get("spoken") and "No relevant information" not in r["spoken"]
    ]

    # Combine candidate results into a concise summary
    combined_result = combine_relevant_results(query, candidate_results, max_words=300, similarity_threshold=0.6)
    print(f"[DEBUG] Combined result from candidate sources: {combined_result}")

    # ── raw_text mode ────────────────────────────────────────────────────────
    # Called by fairy_brain's ReasoningEngine — it needs plain text evidence,
    # not a streaming prompt.  Return combined_result directly without building
    # the synthesis prompt or the streaming dict.
    if mode == "raw_text":
        print("[DEBUG] open_search raw_text mode: returning combined_result directly")
        return combined_result
    # ── end raw_text mode ────────────────────────────────────────────────────

    # -----------------------------
    # FACT EXTRACTION + SYNTHESIS (FAST + NON-RECURSIVE)
    # -----------------------------

    all_facts = []
    source_map = {}
    source_counter = 1

    def register_source(url):
        nonlocal source_counter
        if url not in source_map:
            source_map[url] = source_counter
            source_counter += 1
        return source_map[url]

    # -----------------------------
    # 1️⃣ Collect all raw facts first (NO LLM calls here)
    # -----------------------------

    raw_facts = []
    source_display = {}  # ← ADD THIS HERE

    for result in candidate_results:
        text = result.get("spoken", "")
        urls = result.get("urls", [])

        if urls:

            primary_sid = register_source(urls[0])
            source_display[primary_sid] = urls[0]
            for extra_url in urls[1:]:
                extra_sid = register_source(extra_url)
                source_display[extra_sid] = extra_url
            sid = primary_sid  # facts from this result cite the primary URL
        else:
            source_label = result.get("source_type", "Knowledge Base")
            sid = register_source(source_label)
            source_display[sid] = source_label

        for fact in extract_facts(text):
            raw_facts.append((fact.strip(), sid))

    # -----------------------------
    # 2️⃣ Lightweight Python deduplication (FAST)
    # -----------------------------

    seen = set()
    for fact, sid in raw_facts:
        normalized = fact.lower().strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            all_facts.append((fact, sid))

    # ✅ ADD THIS: Per-sentence relevance filter
    q_emb = _get_semantic_model().encode(cleaned_query, convert_to_tensor=True, device=device)

    relevant_facts = []
    for fact, sid in all_facts:
        fact_emb = _get_semantic_model().encode(fact, convert_to_tensor=True, device=device)
        score = util.cos_sim(q_emb, fact_emb).item()
        if score >= 0.30:  # minimum per-sentence relevance
            relevant_facts.append((fact, sid))
        else:
            print(f"[DEBUG] Fact rejected (score {score:.3f}): {fact[:60]}...")

    print(f"[DEBUG] Facts after per-sentence filter: {len(relevant_facts)} / {len(all_facts)}")
    all_facts = relevant_facts

    if not all_facts:
        return combined_result

    # -----------------------------
    # 3️⃣  KB AUGMENTATION
    # Run the KB query now — AFTER internet facts are collected and filtered.
    # This is intentionally sequential (not parallel) so the novelty gate can
    # compare KB sentences against the finalised internet fact pool.
    # Only sentences that are (a) relevant to the query AND (b) not already
    # covered by internet facts are injected.  If nothing passes, all_facts is
    # unchanged and synthesis proceeds as normal.
    # -----------------------------
    try:
        print("[DEBUG] Running KB augmentation against finalised internet facts...")
        kb_raw = await query_knowledge_base(kb_instance, cleaned_query)
        if kb_raw and kb_raw.strip():
            # Register a source id for the KB so citations work correctly
            kb_sid = register_source("Knowledge Base")
            source_display[kb_sid] = "Knowledge Base (domain PDF corpus)"

            kb_injected = _augment_facts_with_kb(
                query        = cleaned_query,
                internet_facts = all_facts,
                kb_text      = kb_raw,
                kb_source_id = kb_sid,
            )
            if kb_injected:
                all_facts = all_facts + kb_injected
                print(f"[DEBUG] KB augmentation: {len(kb_injected)} sentence(s) added to fact pool.")
            else:
                print("[DEBUG] KB augmentation: no sentences passed the dual-gate — skipping.")
        else:
            print("[DEBUG] KB augmentation: KB returned no text for this query.")
    except Exception as _kb_aug_err:
        print(f"[WARN] KB augmentation failed (non-fatal): {_kb_aug_err}")

    # -----------------------------
    # 4️⃣ Contradiction detection before synthesis
    # -----------------------------
    contradictions = _detect_contradictions(all_facts)
    if contradictions:
        print(f"[DEBUG] {len(contradictions)} potential contradiction(s) detected — flagging to LLM")

    # Build references using source_display (guaranteed to match fact sids):
    references_block = "\n".join(
        f"[{sid}] {url}" for sid, url in sorted(source_display.items())
    )

    # -----------------------------
    # 5️⃣ Return prompt — run_llm_stream will stream it
    # -----------------------------

    synthesis_prompt, sources_suffix = await synthesize_with_citations(
        query,
        all_facts,
        references_block=references_block,
        contradictions=contradictions,
        return_prompt=True
    )

    return {
        "streaming": True,
        "prompt": synthesis_prompt,
        "sources_suffix": sources_suffix,   # Python appends this after streaming ends
        "intent_name": "search",
        "urls": list(source_map.keys()),
    }


async def main():
    """
    Standalone test function for your search module.
    Prompts the user for a query and prints the merged results.
    """

    print("=== Search Module Test ===")
    query = input("Enter your search query: ").strip()
    if not query:
        print("No query provided. Exiting...")
        return

    # Call the main search function
    combined_result = await open_search(query, mode="auto")

    print(f"Search Result: {combined_result}")

if __name__ == "__main__":
    asyncio.run(main())