import os
import requests
import time
import json
import requests
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from typing import TypedDict, Optional
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from Bio import Entrez

load_dotenv()
client = Groq()

# --- Configurazione ---
'''
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
MAX_RESULTS = 3       # quanti paper recuperare
MAX_RETRY = 3         # tentativi massimi in caso di 429
RETRY_DELAY = 3       # secondi di attesa iniziale (raddoppia ad ogni tentativo)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
'''
# NOTA CRITICA: PubMed richiede obbligatoriamente un indirizzo email per identificare chi usa le loro API pubbliche gratuite
Entrez.email = "ale.emme01@gmail.com"

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
    
    clean_keywords_lower = state["clean_keywords"].lower()
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
            if argomenti_correlati(argomento, clean_keywords_lower):
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

def argomenti_correlati(chiave_db: str, testo_utente: str) -> bool:
    # Normalizziamo tutto in minuscolo per evitare problemi di maiuscole
    chiave_db = chiave_db.lower().strip()
    testo_utente = testo_utente.lower().strip()
    
    # 1. Match diretto bidirezionale (Intercetta "zavorre" se l'utente scrive "zavorre")
    if chiave_db in testo_utente or testo_utente in chiave_db:
        return True
        
    # 2. Cluster semantici di sinonimi, TO-DO : potremmo usare un embedding o un modello semantico per fare match più sofisticati
    mappa_sinonimi = {
        "forza": ["forza", "zavorre", "zavorrate", "weighted", "dips", "pull up", "pull-up", "trazioni", "massimali"],
        "ipertrofia": ["massa", "ipertrofia", "muscolo", "crescita", "volume", "ipertrofico"],
        "dimagrimento": ["definizione", "cut", "dimagrire", "cardio", "deficit", "dimagrimento"]
    }
    
    # Controlliamo se sia la chiave del DB sia l'input dell'utente appartengono allo stesso "mondo"
    for radice, sinonimi in mappa_sinonimi.items():
        # La chiave del DB appartiene a questo cluster? (es: è la radice "forza" o la parola "zavorre"?)
        chiave_nel_cluster = (radice in chiave_db) or any(s in chiave_db for s in sinonimi)
        
        # L'input dell'utente appartiene a questo stesso cluster? (es: l'utente ha scritto "dips" o "forza"?)
        utente_nel_cluster = (radice in testo_utente) or any(s in testo_utente for s in sinonimi)
        
        # Se entrambi sono nello stesso cluster, abbiamo un match scientifico!
        if chiave_nel_cluster and utente_nel_cluster:
            return True
            
    return False

# ---------------------------------- NODO 2: Ricerca Paper Scientifici Online ----------------- -----------------
def cerca_paper(state: AgentState) -> AgentState:
    print("\n[Nodo 2] -> Keyword mancanti nella KB. Ottimizzazione della query per PubMed...")
    
    # 1. Recuperiamo il focus in italiano (es. "zavorre dips forza")
    focus_italiano = state.get("clean_keywords", "")
    if not focus_italiano:
        print("  -> Errore: nessuna keyword disponibile nello state.")
        return {"kb_data": "Errore: parole chiave non disponibili."}
        
    # 2. CHIAMATA LIGHT A GROQ: Trasformiamo l'input in una query scientifica inglese
    # Usiamo llama-3.1-8b-instant perché è fulmineo, gratuito e perfetto per la traduzione
    prompt_ottimizzazione = (
        "Sei un assistente specializzato in ricerca scientifica sportiva. "
        "Il tuo compito è convertire i termini di allenamento in italiano forniti dall'utente "
        "in una singola stringa di ricerca ottimizzata per il database PubMed (in inglese).\n"
        "Usa termini scientifici corretti (es. 'resistance training' invece di 'pesi', 'hypertrophy' invece di 'ipertrofia').\n\n"
        "REQUISITI RIGIDI:\n"
        "1. Restituisci SOLO la stringa di ricerca finale in inglese, senza introduzioni, commenti o virgolette.\n"
        "2. Mantieni la query focalizzata: usa al massimo 2 o 3 termini chiave inglesi uniti (se necessario) dall'operatore AND.\n\n"
        f"Keywords italiane da ottimizzare: {focus_italiano}"
    )
    
    try:
        print(f"  -> Focus originale: '{focus_italiano}'")
        traduzione_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_ottimizzazione}],
            model="llama-3.1-8b-instant", 
        )
        # La stringa pulita (es. "weighted dips AND strength training")
        query_pubmed = traduzione_completion.choices[0].message.content.strip()
        print(f"  -> 🎯 Query PubMed generata (Inglese): '{query_pubmed}'")
        
        # 3. FASE 1 PUBMED (ESearch): Cerchiamo i 2 paper più attinenti usando la query inglese
        print("  -> Interrogazione database PubMed via Biopython...")
        handle = Entrez.esearch(db="pubmed", term=query_pubmed, retmax=2)
        risultato_ricerca = Entrez.read(handle)
        handle.close()
        
        id_list = risultato_ricerca.get("IdList", [])
        if not id_list:
            print(f"  -> Nessun paper trovato su PubMed per la query: '{query_pubmed}'")
            return {"kb_data": "Nessun paper scientifico trovato per questa domanda."}
            
        print(f"  -> Trovati {len(id_list)} ID di paper ({id_list}). Scaricamento abstract...")
        
        # 4. FASE 2 PUBMED (EFetch): Scarichiamo gli abstract reali
        id_stringa = ",".join(id_list)
        handle = Entrez.efetch(db="pubmed", id=id_stringa, retmode="xml")
        dati_completi = Entrez.read(handle)
        handle.close()
        
        testi_estratti = []
        articoli = dati_completi.get("PubmedArticle", [])
        
        for i, articolo in enumerate(articoli, start=1):
            try:
                medline = articolo["MedlineCitation"]
                titolo = medline["Article"]["ArticleTitle"]
                abstract_data = medline["Article"].get("Abstract", {})
                abstract_text_list = abstract_data.get("AbstractText", [])
                
                abstract_completo = " ".join([str(p) for p in abstract_text_list])
                if abstract_completo:
                    testi_estratti.append(f"--- Paper {i}: {titolo} ---\nAbstract: {abstract_completo}\n")
            except KeyError:
                continue
                
        if not testi_estratti:
            return {"kb_data": "Impossibile recuperare abstract validi dai paper."}
            
        contesto_scientifico = "\n".join(testi_estratti)
        print("  -> Abstract recuperati con successo! Invio a llama per l'estrazione dati strutturata...")
        
        # 5. FASE 3 SINTESI (llama): Elaboriamo il JSON finale basandoci sui paper veri
        prompt_sistema = (
            "Sei un assistente di ricerca specializzato in biomeccanica. Analizza gli abstract in inglese "
            "e convertili in uno schema JSON rigoroso scritto in ITALIANO.\n\n"
            "REQUISITI DI OUTPUT:\n"
            "1. Rispondi ESCLUSIVAMENTE con un oggetto JSON valido, senza blocchi markdown (NO a ```json).\n"
            "2. Usa come chiave principale esattamente la parola chiave italiana che ha originato la ricerca.\n\n"
            "Struttura richiesta:\n"
            "{\n"
            f"  \"{focus_italiano}\": " "{\n"
            "    \"recuperi\": \"linee guida sui tempi di recupero\",\n"
            "    \"intensita_rpe\": \"linee guida su intensità o RPE\",\n"
            "    \"note_scientifiche\": \"sintesi in italiano delle evidenze del paper\"\n"
            "  }\n"
            "}"
        )
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": contesto_scientifico}
            ],
            model="llama-3.3-70b-versatile", 
            response_format={"type": "json_object"},
        )
        
        sintesi_json_string = chat_completion.choices[0].message.content
        return {"kb_data": json.loads(sintesi_json_string)}
        
    except Exception as e:
        print(f"  -> Errore nel Nodo 2: {e}")
        return {"kb_data": f"Errore durante la ricerca: {str(e)}"}


# ---------------------------------- NODO 3: Aggiornamento Knowledge Base ----------------- -----------------
def aggiorna_kb(state: AgentState) -> AgentState:
    print("\n[Nodo 3] -> Scrittura delle nuove scoperte scientifiche nel file JSON...")
    
    # 1. Recuperiamo il dizionario strutturato prodotto dall'LLM nel Nodo 2
    nuovi_dati = state.get("kb_data")
    file_path = "knowledge_base.json"
    
    # Verifichiamo che ci siano dati reali da salvare e che sia un dizionario
    if nuovi_dati and isinstance(nuovi_dati, dict):
        # 2. Leggiamo il file esistente per fare un "append logico" ed evitare sovrascritture
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    kb = json.load(f)
                except json.JSONDecodeError:
                    kb = {} # Se il file fosse corrotto o vuoto, ripartiamo da un dizionario pulito
        else:
            kb = {}
        
        # 3. Aggiorniamo la Knowledge Base locale fondendo i vecchi dati con i nuovi
        kb.update(nuovi_dati)
        
        # 4. Scriviamo il file aggiornato su disco
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(kb, f, indent=4, ensure_ascii=False)
            
        print(f"  -> 💾 File '{file_path}' aggiornato con successo con le nuove evidenze!")
    else:
        print("  -> Avviso: nessun dato dizionario valido trovato in 'kb_data'. Salto la scrittura.")
        
    # Ritorniamo il flag impostato su False: ora i dati ci sono, 
    # il bivio condizionale devierà il flusso direttamente al Nodo 4 (genera_scheda)
    return {"missing_info": False, "kb_data": json.dumps(nuovi_dati, indent=4, ensure_ascii=False)}

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