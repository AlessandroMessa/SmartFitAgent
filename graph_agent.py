import os
import requests
import time
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from typing import TypedDict, Optional
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
import json

load_dotenv()
client = Groq()

# --- Configurazione ---
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
MAX_RESULTS = 3  # quanti paper recuperare
MAX_RETRY = 3         # tentativi massimi in caso di 429
RETRY_DELAY = 3       # secondi di attesa iniziale (raddoppia ad ogni tentativo)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# 1. Definiamo lo Stato del Grafo
class AgentState(TypedDict):
    user_input: str
    clean_keywords: str # Parole chiave estratte dall'input dell'utente
    kb_data: Optional[str] # Evidenza scientifica trovata o None se non disponibile
    missing_info: bool # Indica se l'input dell'utente richiede dati non presenti nella KB
    final_program: Optional[str] # Contiene il programma finale generato dall'agente, se disponibile

# 2. Definiamo i Nodi del Workflow

# ---------------------------------- NODO 0: Estrazione Focus ----------------- -----------------
def estrai_focus(state: AgentState) -> AgentState:
    print("\n[Nodo 0] -> Pulizia dell'input ed estrazione keyword scientifiche...")
    
    prompt_sistema = """
    Sei un analista dati specializzato in scienze motorie. Il tuo compito è leggere la richiesta dell'utente ed estrarre ESCLUSIVAMENTE i termini scientifici, i pattern di movimento o gli esercizi core (es. dips, pull-ups, squat, forza, ipertrofia, rpe).
    Rimuovi saluti, preamboli, stati d'animo e parole inutili. 
    Rispondi SOLO con le parole chiave separate da virgola, senza nient'altro.
    
    Esempio: "Vorrei fare forza nelle dips con zavorra ma ho i polsi deboli"
    Output: dips, zavorra, forza, polsi
    """
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": state["user_input"]}
        ],
        model="llama-3.1-8b-instant", # Veloce ed economico per task di estrazione
    )
    
    keywords_pulite = response.choices[0].message.content.lower().strip()
    print(f"Keyword estratte per la ricerca: [{keywords_pulite}]")
    
    return {"clean_keywords": keywords_pulite}

# ---------------------------------- NODO 1: Controllo KB Locale ----------------- -----------------
def controlla_kb(state: AgentState) -> AgentState:
    print("\n[Nodo 1] -> Controllo reale della Knowledge Base locale...")
    
    user_input_lower = state["user_input"].lower()
    script_dir = Path(__file__).parent
    kb_path = script_dir / 'knowledge_base.json'
    
    # 1. Controlliamo se il file JSON esiste
    if not kb_path.exists():
        print("Il file knowledge_base.json non esiste ancora. Deviazione su ricerca paper.")
        return {"missing_info": True, "kb_data": None}
        
    try:
        # 2. Leggiamo il database JSON
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb_content = json.load(f)
            
        # 3. Scansione delle chiavi nel JSON per trovare corrispondenze
        dati_trovati = []
        for argomento, dettagli in kb_content.items():
            # Se il nome dell'argomento (es. "linee_guida_forza" o "zavorre") è nell'input dell'utente
            # O se l'utente menziona parole chiave interne
            if argomenti_correlati(argomento, user_input_lower):
                print(f"Corrispondenza trovata nella KB per l'argomento: '{argomento}'")
                dati_trovati.append(f"--- {argomento.upper()} ---\n{json.dumps(dettagli, indent=2, ensure_ascii=False)}")
        
        # 4. Gestione del risultato del bivio logico
        if dati_trovati:
            # Uniamo tutte le regole scientifiche trovate in un'unica stringa nel blocco note (stato)
            contesto_kb = "\n\n".join(dati_trovati)
            return {"missing_info": False, "kb_data": contesto_kb}
        else:
            print("Nessuna linea guida specifica trovata nel database locale per questa richiesta.")
            return {"missing_info": True, "kb_data": None}
            
    except Exception as e:
        print(f"Errore durante la lettura della KB: {e}")
        return {"missing_info": True, "kb_data": None}

# Funzione di utility per il nodo controlla_kb
def argomenti_correlati(chiave_db: str, testo_utente: str) -> bool:
    # Mappiamo i sinonimi per rendere l'agente più elastico
    mappa_sinonimi = {
        "forza": ["forza", "zavorre", "weighted", "dips", "pull up", "pull-up", "massimali"],
        "ipertrofia": ["massa", "ipertrofia", "muscolo", "crescita", "volume"],
        "dimagrimento": ["definizione", "cut", "dimagrire", "cardio", "deficit"]
    }
    
    # Controllo diretto sul nome della chiave
    if chiave_db in testo_utente:
        return True
        
    # Controllo sui sinonimi mappati
    for radice, sinonimi in mappa_sinonimi.items():
        if radice in chiave_db:
            return any(sinonimo in testo_utente for sinonimo in sinonimi)
            
    return False

# ---------------------------------- NODO 2: Ricerca Paper Scientifici Online ----------------- -----------------
def cerca_paper(state: AgentState) -> AgentState:
    print("\n[Nodo 2] -> Dati non trovati in KB. Avvio ricerca paper scientifici online...")
    
    # Nodo LangGraph: cerca paper su Semantic Scholar e sintetizza
    # le evidenze con un LLM prima di salvarle in kb_data.
    
    # Recupera la domanda originale dallo state
    domanda = state.get("user_input", "")
    if not domanda:
        return {"kb_data": "Errore: nessuna domanda disponibile nello state."}
 
    try:
        # 1. Cerca i paper
        print(f"  -> Ricerca su Semantic Scholar per: '{domanda}'")
        papers = _cerca_su_semantic_scholar(domanda)
 
        if not papers:
            return {"kb_data": "Nessun paper scientifico trovato per questa domanda."}
 
        print(f"  -> Trovati {len(papers)} paper con abstract. Avvio sintesi LLM...")
 
        # 2. Riassumi con LLM
        sintesi = _riassumi_con_llm(domanda, papers)
 
        print("  -> Sintesi completata.")
        return {"kb_data": sintesi}
 
    except requests.RequestException as e:
        print(f"  -> Errore nella chiamata a Semantic Scholar: {e}")
        return {"kb_data": f"Errore durante la ricerca online: {str(e)}"}

# Funzione di utility per il nodo cerca_paper
def _cerca_su_semantic_scholar(query: str) -> list[dict]:
    #Chiama l'API di Semantic Scholar e restituisce una lista di paper
    params = {
        "query": "dips, pull up, strength", #andra sostituito con query più specifica in futuro
        "limit": MAX_RESULTS,
        "fields": "title,abstract,year,authors,url",
    }
    headers = {
        "User-Agent": "GymAgent/1.0",  # buona pratica verso l'API
        "x-api-key": os.getenv("SEMANTIC_SCHOLAR_API_KEY", ""),  
    }
 
    #serve per riprovare la richiesta in caso di 429 (rate limit)
    delay = RETRY_DELAY
    for tentativo in range(1, MAX_RETRY + 1):
 
        response = requests.get( SEMANTIC_SCHOLAR_URL, params=params, headers=headers, timeout=15 )
 
        if response.status_code == 429:
            # Rispetta il Retry-After se presente nell'header, altrimenti backoff
            retry_after = int(response.headers.get("Retry-After", delay))
            print(f"  -> 429 Rate limit. Attendo {retry_after}s (tentativo {tentativo}/{MAX_RETRY})...")
            time.sleep(retry_after)
            delay *= 2  # backoff esponenziale
            continue
 
        response.raise_for_status()  # altri errori HTTP vengono lanciati subito
 
        data = response.json()
        papers = data.get("data", [])
 
        risultati = []
        for p in papers:
            abstract = p.get("abstract") or ""
            if not abstract:
                continue  # scarta paper senza abstract
            risultati.append({
                "titolo": p.get("title", "N/D"),
                "anno": p.get("year", "N/D"),
                "abstract": abstract,
                "url": p.get("url", ""),
            })
 
        return risultati
 
    raise requests.RequestException(
        f"Semantic Scholar ha risposto con 429 per {MAX_RETRY} tentativi consecutivi."
    )

# Funzione di utility per il nodo cerca_paper 
def _riassumi_con_llm(domanda: str, papers: list[dict]) -> str:
    #Usa l'LLM per sintetizzare le evidenze dai paper trovati.
    testi_paper = "\n\n".join([
        f"[Paper {i+1}] {p['titolo']} ({p['anno']})\n{p['abstract']}\nFonte: {p['url']}"
        for i, p in enumerate(papers)
    ])
 
    prompt = f"""Sei un esperto di scienze dello sport e allenamento.
                L'utente ha fatto questa domanda: "{domanda}"
 
                Ecco gli abstract di {len(papers)} paper scientifici pertinenti:
 
                {testi_paper}
 
                Sintetizza le evidenze scientifiche trovate in modo chiaro e pratico,
                citando i paper per numero (es: [Paper 1]).
                Sii conciso ma preciso. Rispondi in italiano."""
 
    risposta = llm.invoke([HumanMessage(content=prompt)])
    return risposta.content

# ---------------------------------- NODO 3: Aggiornamento Knowledge Base ----------------- -----------------
def aggiorna_kb(state: AgentState) -> AgentState:
    print("\n[Nodo 3] -> Scrittura delle nuove scoperte scientifiche nel file JSON...")
    # Ritorniamo un dizionario valido per non mandare in crash lo stream
    return {"missing_info": False}

# ---------------------------------- NODO 4: Generazione Scheda ----------------- -----------------
def genera_scheda(state: AgentState) -> AgentState:
    print("\n[Nodo 4] -> Generazione della scheda con Groq usando i dati scientifici...")
    
    # Chiamata a Groq usando il contesto accumulato nello stato
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"Sei un PT d'élite. Crea una scheda basandoti SOLO su questi dati scientifici: {state['kb_data']}"
            },
            {
                "role": "user", 
                "content": state["user_input"]}
        ],
        model="llama-3.3-70b-versatile",
    )
    
    return {"final_program": chat_completion.choices[0].message.content}

# 3. Definiamo la Funzione Condizionale (Router)
def decidi_percorso(state: AgentState):
    if state["missing_info"]:
        return "cerca_paper"
    return "genera_scheda"

# 4. Costruzione del Grafo
workflow = StateGraph(AgentState)

# Aggiungiamo i nodi
workflow.add_node("estrai_focus", estrai_focus)
workflow.add_node("controlla_kb", controlla_kb)
workflow.add_node("cerca_paper", cerca_paper)
workflow.add_node("aggiorna_kb", aggiorna_kb)
workflow.add_node("genera_scheda", genera_scheda)

# Impostiamo il punto di ingresso
workflow.set_entry_point("estrai_focus")

# Aggiungiamo il bivio condizionale dopo il controllo KB
workflow.add_conditional_edges(
    "controlla_kb",
    decidi_percorso,
    {
        "cerca_paper": "cerca_paper",
        "genera_scheda": "genera_scheda"
    }
)

# Colleghiamo i nodi della ricerca fino alla generazione
workflow.add_edge("estrai_focus", "controlla_kb")
workflow.add_edge("cerca_paper", "aggiorna_kb")
workflow.add_edge("aggiorna_kb", "genera_scheda")
workflow.add_edge("genera_scheda", END)

# Inizializza il salvataggio in memoria
memory = MemorySaver()

# Compiliamo il grafo
app = workflow.compile()

'''
# --- GENERAZIONE E SALVATAGGIO DEL GRAFICO (VERSIONE WINDOWS) ---
print("\n--- GENERAZIONE MAPPA DEL GRAFO ---")
try:
    # Genera la stringa di testo in formato Mermaid
    codice_mermaid = app.get_graph().draw_mermaid()
    
    print("\nCopia TUTTO il blocco di testo qui sotto (comprese le linee con i trattini):\n")
    print("--------------------------------------------------")
    print(codice_mermaid)
    print("--------------------------------------------------")
    
except Exception as e:
    print(f"⚠️ Impossibile generare il codice Mermaid: {e}")
# ----------------------------------------------------------------
'''

# 5. Esecuzione di Test
if __name__ == "__main__":
    print("=== SmartFitAgent V1.0 (LangGraph Test) ===")
    
    # 1. Prepariamo l'input dell'utente
    inputs = {"user_input": "Vorrei una scheda su 3 giorni per migliorare la forza in dips e pull up zavorrate"}
    
    # 2. Definiamo la configurazione con il thread_id obbligatorio per MemorySaver
    config = {
        "configurable": {
            "thread_id": "sessione_allenamento_01"
        }
    }
    
    print("\n--- AVVIO WORKFLOW IN CORSO ---")
    
    # Variabile d'appoggio per catturare il risultato finale durante lo streaming
    risultato_finale = None

    # 3. Eseguiamo lo stream passando SEMPRE sia gli input che la config
    for output in app.stream(inputs, config=config):
        for node_name, node_state in output.items():
            print(f"✅ Nodo completato: [{node_name}]")
            
            # PROTEZIONE: Controlliamo che il node_state sia valido e non None
            if node_state and "final_program" in node_state:
                risultato_finale = node_state["final_program"]

    print("\n--- WORKFLOW COMPLETATO ---")
    print("\n=== RISPOSTA FINALE DELL'AGENTE PT ===")
    
    if risultato_finale:
        print(risultato_finale)
    else:
        # Nel caso estremo in cui get_state dia ancora problemi, usiamo questo fallback sicuro
        try:
            state_snapshot = app.get_state(config)
            print(state_snapshot.values.get("final_program", "Nessun programma trovato nello stato."))
        except Exception as e:
            print(f"Non è stato possibile recuperare lo stato: {e}")