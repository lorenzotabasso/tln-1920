# Relazione Esercizio 3

In questa esercitazione, abbiamo creato i cluster semantici basandoci sul concetto della valenza dei verbi di Hanks.

1. Abbiamo lavorato sul Brown Corpus e previsto di dare in input il verbo (con valenza  = 2) su cui eseguire l'algoritmo.

2. Sono state estratte dal corpus tutte le frasi contenenti il verbo di input.

3. Allo scopo di estrarre i filler di ogni frase, abbiamo effettuato: PoS Tagging (tramite _Stanford CoreNLP_), analisi sintattica creando per ogni frase un grafo a dipendenze (si veda la classe OurDependencyGraph), e lemmatizzato tutti i verbi presenti all'interno del grafo, infine convertendo il grafo in lista di adiacenza è stato possibile estrarre i filler per ogni verbo che rispettasse la valenza specificata (nel nostro caso 2).

4. Durante la fase di aggregazione dei risultati, abbiamo recuperato i Supersensi dai Wordnet Synset delle parole disambiguate tramite due implementazioni dell' algoritmmo di, Lesk, una versione nostra (OurLesk) e la versione di NLTK, poiché abbiamo riscontrato dei risultati diversi nell'aggregazione dei Semantic Type.

## Risultati

A fine esecuzione, l'algoritmo descritto, ha dato il seguente output:

```shell script
Enter a verb to search in the Brown corpus: meet
[1] - Extracting sentences...
	307 sentences in which the verb 'meet' appears.

[2] - Extracting fillers...

[3] - Total of 7 Fillers
	('himself', 'person', "one way to do this is by `` proxy sittings '' , wherein the person seeking a message does not himself meet with the medium but is represented by a substitute , the proxy sitter ")
	('i', 'it', 'hiroshima is a better city than it was before -- in the minds of the people i met was a strong determination for peace and understanding ')
	('i', 'literature', "after all , the large ( and probably unreliable ) reader's digest literature on the `` most unforgettable character i ever met '' deals with village grocers , country doctors , favorite if illiterate aunts , and so forth ")
	('founder', 'pallavicini', 'and general the count pallavicini , founder of the austrian branch of that celebrated italian house , a courtier littlepage could have met at madrid in december , 1780 ')
	('people', 'we', 'this of course was not true of the educated and sophisticated people we met , who loved their pets , but kindness is not a basic human instinct ')
	('officers', 'she', 'if tommy sat long enough , she would be sure to see all the young officers she had met in san diego and long beach ')
	('husbands', 'i', 'the husbands of these women and others i had met in catatonia were distinguished only in that they were , to me at least , indistinguishable ')

[4.1] - "Our Lesk":
	Finding Semantic Clusters (percentage, count of instances, semantic cluster):
	[14.29%] - 1 - ('noun.person', 'noun.substance')
	[28.57%] - 2 - ('noun.substance', 'noun.cognition')

[4.2] - "NLTK Lesk":
	Finding Semantic Clusters (percentage, count of instances, semantic cluster):
	[14.29%] - 1 - ('adj.all', 'noun.cognition')
	[14.29%] - 1 - ('adj.all', 'noun.act')
	[14.29%] - 1 - ('noun.person', 'adj.all')
```
