# Relazione Esercizio 2

Nello sviluppo di questo esercizio, dopo una serie di prove abbiamo implementato il seguente algoritmo:

1. Per ogni concetto (riga), esploriamo una definizione per volta (colonna)

2. Data la singola definizione, prendiamo tutti i nomi tramite un'analisi dei Pos Tag della frase, e tra questi estraiamo il nome più frequente e lo impostiamo come Genus della definizione

3. Per ogni definizione, ci salviamo una lista di iponimi calcolati a partire dal suo Genus

4. Sfruttando l'oggetto [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer) di Scikit-Learn otteniamo un vettore di frequenze degli iponimi presenti in tutte le definizioni del concetto elaborato

5. Definiamo come concetto risultante l'iponimo più frequente tra tutti gli iponimi per quella definizione. Quindi il max del CountVectorizer.

HYPER -------------------------------------------------------------------------

- nel punto 2 facciamo prima disambiguazione con Lesk della parola prima di trattarla per recuperare tutti gli iperonimi della parola disambiguata.

- prendiamo come genus l'iperonimo più frequente

- prendiamo tutti i possibili sensi synset del genus

- per ogni synset prendiamo gli iponimi

## Risultati

Sviluppi futuri:

- Se scendiamo troppo di livello con gli iponimi tendiamo a generalizzare a partire da un certo livello

- anzichè restituire il genus più frequente restituire il genus che ha generato l'iponimo puù frequente

- è stato difficile capire quando salire e quanto scendere nell'albero dei synset.

- risultati migliori (più simili alla soluzione) con gli iperonimi
