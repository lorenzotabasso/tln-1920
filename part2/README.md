# Parte 2 (prof. Radicioni):
Questa parte di progetto consiste nello sviluppare 3 esercitazioni.

## Esercitazione 1: WordNet 

1. implementare l’algoritmo di Lesk (**non usare** implementazione esistente, come quella di NLTK).
2. Estrarre 50 frasi dal corpus _SemCor_ (corpus annotato con i synset di WornNet) e disambiguare
 almeno un sostantivo per frase. Calcolare l’accuratezza del sistema implementato sulla base dei 
 sensi annotati in SemCor. SemCor è disponibile all’URL http://web.eecs.umich.edu/~mihalcea/downloads.html

## Esercitazione 2: Mapping di Frame in WN Synsets
### 1. individuazione di un set di frame
Come prima operazione ciascuno deve individuare un insieme di frame (nel seguito riferito come FrameSet) su cui lavorare.
A tale fine utilizzare la funzione getFrameSetForStudent(cognome); nel caso il gruppo sia costituito da 2 o 3 
componenti, utilizzare la funzione per trovare il set di frame per ciascuno dei componenti del gruppo. La funzione 
restituisce, dato un cognome in input, l'elenco di frame da elaborare. Gli studenti Mario Rossi e Marta Verdi 
eseguirebbero quindi due chiamate

```python
getFrameSetForStudent('Rossi')
getFrameSetForStudent('Verdi')
```

ottenendo in output, rispettivamente,

```python
student: Rossi
  ID: 2562 frame: Manner_of_life
  ID: 1302 frame: Response
  ID: 1700 frame: Knot_creation_scenario ID: 2380 frame: Popularity
  ID: 1602 frame: Abundance
```

e

```python
student: Verdi
  ID: 2481 frame: Erasing
  ID: 1790 frame: Means
  ID: 2916 frame: Distributed_abundance ID: 37 frame: Hearsay
  ID: 1816 frame: Removing_scenario
```

Questi saranno pertanto i frame utilizzati dai componenti del gruppo Rossi-Verdi.

### 2. Consegna
Per ogni frame nel FrameSet è necessario assegnare un WN synset ai seguenti elementi:
- Frame name (nel caso si tratti di una multiword expression, come per esempio 'Religious_belief', disambiguare il 
termine principale, che in generale è il sostantivo se l'espressione è composta da NOUN+ADJ, e il verbo se 
l'espressione è composta da VERB+NOUN; in generale l'elemento fondamentale è individuato come il reggente dell'espressione.
- Frame Elements (FEs) del frame;
- Lexical Units (LUs).

I contesti di disambiguazione possono essere creati utilizzando le definizioni disponibili (sia quella del frame, sia 
quelle dei FEs), ottenendo ```Ctx(w)``` , il contesto per FN terms ```w```.

Per quanto riguarda il contesto dei sensi presenti in WN è possibile selezionare glosse ed esempi dei sensi, e dei loro
rispettivi iponimi e iperonimi, in modo da avere più informazione, ottenendo quindi il contesto di disambiguazione 
```Ctx(s)``` .