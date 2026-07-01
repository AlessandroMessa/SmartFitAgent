import json
from typing import Dict
from Bio import Entrez
from state import AgentState

# Nota: PubMed richiede un indirizzo email per identificare l'applicazione
Entrez.email = "ale.emme01@gmail.com"

class CercaPaper:

    def __init__(self, client_or_api_key):
        self.client = client_or_api_key

    def __call__(self, state: AgentState) -> Dict:
        print("\n[Nodo 2] -> Keyword mancanti nella KB. Ottimizzazione della query per PubMed...")
        
        # 1. Recuperiamo il focus in italiano
        focus_italiano = state.get("clean_keywords", "")
        if not focus_italiano:
            print("  -> Errore: nessuna keyword disponibile nello state.")
            return {"kb_data": None, "missing_info": True}
            
        # 2. Ottimizzazione della query per PubMed tramite Groq (Llama-3.1-8b)
        prompt_ottimizzazione = (
            "Sei un assistente specializzato in ricerca scientifica sportiva. "
            "Il tuo compito è convertire i termini di allenamento in italiano forniti dall'utente "
            "in una singola stringa di ricerca ottimizzata per il database PubMed (in inglese).\n"
            "Usa termini scientifici corretti (es. 'resistance training' invece di 'pesi', 'hypertrophy' invece di 'ipertrofia').\n\n"
            "REQUISITI RIGIDI:\n"
            "1. Restituisci SOLO la stringa di ricerca finale in inglese, senza introduzioni, commenti o virgolette.\n"
            "2. Mantieni la query focalizzata: usa al massimo 2 o 3 termini chiave inglesi uniti dall'operatore AND.\n\n"
            f"Keywords italiane da ottimizzare: {focus_italiano}"
        )
        
        try:
            print(f"  -> Focus originale: '{focus_italiano}'")
            
            # CORREZIONE: Usiamo self.client invece di client globale
            traduzione_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_ottimizzazione}],
                model="llama-3.1-8b-instant", 
            )
            
            query_pubmed = traduzione_completion.choices[0].message.content.strip()
            print(f"  -> 🎯 Query PubMed generata (Inglese): '{query_pubmed}'")
            
            # 3. FASE 1 PUBMED (ESearch): Cerchiamo i 2 paper più attinenti
            print("  -> Interrogazione database PubMed via Biopython...")
            handle = Entrez.esearch(db="pubmed", term=query_pubmed, retmax=2)
            risultato_ricerca = Entrez.read(handle)
            handle.close()
            
            id_list = risultato_ricerca.get("IdList", [])
            if not id_list:
                print(f"  -> Nessun paper trovato su PubMed per la query: '{query_pubmed}'")
                return {"kb_data": None, "missing_info": True}
                
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
                # Se non ci sono abstract reali (es. solo titoli), usiamo una Clausola di Salvaguardia
                print("  -> Avviso: Trovati paper ma nessun abstract testuale disponibile.")
                return {"kb_data": None, "missing_info": True}
                
            contesto_scientifico = "\n".join(testi_estratti)
            print("  -> Abstract recuperati con successo! Estrazione dati strutturata...")
            
            # 5. FASE 3 SINTESI (Llama-3.3-70b): Generazione del dizionario strutturato
            prompt_sistema = (
                "Sei un assistente di ricerca specializzato in biomeccanica. Analizza gli abstract in inglese "
                "e convertili in uno schema JSON rigoroso scritto in ITALIANO.\n\n"
                "REGOLA CRITICA SE MANCANO DATI REALI:\n"
                "Se l'abstract è teorico e non contiene indicazioni numeriche specifiche su recuperi o RPE, "
                "scrivi nel campo: 'Non specificato nell'abstract, fare riferimento ai principi generali'.\n\n"
                "REQUISITI DI OUTPUT:\n"
                "1. Rispondi ESCLUSIVAMENTE con un oggetto JSON valido, senza blocchi markdown (NO a ```json).\n"
                "2. Usa come chiave principale esattamente la parola chiave italiana che ha originato la ricerca.\n\n"
                "Struttura richiesta:\n"
                "{\n"
                f"  \"{focus_italiano}\": {{\n"
                "    \"recuperi\": \"linee guida sui tempi di recupero o fallback\",\n"
                "    \"intensita_rpe\": \"linee guida su intensità o RPE o fallback\",\n"
                "    \"note_scientifiche\": \"sintesi delle evidenze emesse dal paper\"\n"
                "  }}\n"
                "}"
            )
            
            # CORREZIONE: Usiamo self.client invece di client globale
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": contesto_scientifico}
                ],
                model="llama-3.3-70b-versatile", 
                response_format={"type": "json_object"},
            )
            
            sintesi_json_string = chat_completion.choices[0].message.content
            
            # Ritorniamo il dizionario decodificato (sarà memorizzato in kb_data)
            return {"kb_data": json.loads(sintesi_json_string)}
            
        except Exception as e:
            print(f"  -> Errore nel Nodo 2: {e}")
            return {"kb_data": None, "missing_info": True}