# Relazione Esercizio 4

In questa esercitazione, abbiamo scritto ispirandoci al Text Tiling un algoritmo di Document Segmentation. I passi iniziali di tale algoritmo sono molto semplici, poiché abbiamo voluto rimanere il più possibile fedeli all'implementazione del Text Tilling di NLTK. L'algoritmo si comporta nel seguente modo:

1. Importato e inizializzato nasari all'interno di un dizionario. In una stuttura di configurazione è possibile impostare gli n primi elementi da salvare in dizionario per ogni word.

2. Tramite NLTK abbiamo mappato il corpus in una lista di sentences e generato due strutture. una con le sentences come lista di stringhe e una con le sentences come lista di lista di tokens.

3. Per ogni token (parola) in ogni frase è stata calcolata la similarietà tra quest'ultimo e tutti gli altri token prima nella frase precedente e poi in quella successiva. Per calcolare tale similarietà si è ricorsi prima all'uso della metrica  del Weighted Overlap per computare la similarietà tra i due vettori Nasari dei due token, e poi successivamente, si è fatto ricorso allo Square Root Weighhted Overlap per calcolare la similarietà finale delle due parole a partire dal Weighted Overlap calcolato in precedenza.

4. 

## Osservazioni

Una volta che lo sviluppo dell'esercizio è stato terminato, ci siamo concentrati sullo sperimentare l'efficacia dell'algoritmo descritto poc'anzi cambiando dataset.
