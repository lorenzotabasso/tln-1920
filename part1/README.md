# Parte 1 (prof. Mazzei):
Per questa parte di progetto, si deve realizzare un **traduttore interlingua** che traduca le seguenti frasi 
dall'inglese all'italiano. In modo particolare, questo traduttore dovrà parsificare e interpretare logicamente 
(almeno) le frasi inglesi

1. "You are imagining things",
2. "There is a price on my head",
3. "Your big opportunity is flying out of here".
 
In seguito alla parsificazione, esso dovrò poi **trasformare** la **formula logica** ottenuta in un **sentence plan** 
per SimpleNLG, e infine **generare** la **traduzione** mediante SimpleNLG-it.

Di seguito è presente il dettaglio dei tre passi da svolgere:

## 1. Parsificare le frasi inglesi 
Scrivere una grammatica CF G1 (con semantica!) per le frasi in ingresso e usare [NLTK](https://www.nltk.org).
Suggerimento: ispirarsi alla grammatica “simple-sem.fcfg”, disponibile al seguente [link](https://github.com/nltk/nltk_teach/blob/master/examples/grammars/book_grammars/simple-sem.fcfg)
e a quest'altro [link](http://www.nltk.org/book/ch10.html), contenente una guida a NLTK.

## 2. Dalla Logica al Sentence Plan 
Per passare dalla logica FOL al Sentence Plan è necessario costruire un sentence planner che per ogni formula logica 
prodotta dalla grammatica __G1__ produca un sentence plan valido, da passare in input a SimpleNLG-it. Inoltre va usata 
una lessicalizzazione il più semplice possibile, in modo da traformare le costanti e i predicati in parole italiane.

## 3. Generare le frasi in SimpleNLG-it
Come ultimo passo, bisogna implementare un semplice realizer che trasformi i sentence plans in frasi italiane.
Questo passo può essere implementato usando la libreria SimpleNLG-IT (eventualmente come server attraverso socket o via 
pipe) scritta dal professor Mazzei in java e reperibile al seguente [link](https://github.com/alexmazzei/SimpleNLG-IT/blob/master/docs/Testsimplenlgit.java).
Se si vuole prediligere l'uso di Python, è possibile usare la libreria installandola tramite PiP. Il link in questo 
caso è [questo](https://pypi.org/project/simplenlg/).

### Consegna
Gli esercizi si possono fare in gruppi formati da un massimo di 2 persone. Inoltre, Bisogna consegnare il codice e una 
breve relazione (massimo 10 pagine) almeno due giorni prima della data dell'esame dell'orale concordata. 