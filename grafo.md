```mermaid
graph TD;
    __start__([START]):::first
    estrai_focus(estrai_focus)
    controlla_kb(controlla_kb)
    cerca_paper(cerca_paper)
    aggiorna_kb(aggiorna_kb)
    genera_scheda(genera_scheda)
    __end__([END]):::last

    __start__ --> estrai_focus;
    estrai_focus --> controlla_kb;
    
    %% Il resto rimane uguale ma usa i dati puliti
    cerca_paper --> aggiorna_kb;
    aggiorna_kb --> genera_scheda;
    genera_scheda --> __end__;
    controlla_kb -.-> cerca_paper;
    controlla_kb -.-> genera_scheda;

    classDef default fill:#D1C4E9,stroke:#512DA8,stroke-width:3px,color:#000000,font-weight:bold
    classDef first fill:#B3E5FC,stroke:#0288D1,stroke-width:3px,color:#000000,font-weight:bold
    classDef last fill:#C8E6C9,stroke:#388E3C,stroke-width:3px,color:#000000,font-weight:bold