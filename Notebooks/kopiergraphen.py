import pandas as pd
import numpy as np
from random import random,randrange,choice
import matplotlib.pyplot as plt 
import itertools
from IPython.display import HTML
import pygraphviz as pgv
import networkx as nx
from collections import Counter
from scipy.misc import factorial

from pandas.tools.plotting import scatter_matrix

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score

def show2images(pnglist,width):
    """
    Zeige zwei oder mehr Bilddateien nebeneinander an.
    Parameter: 
    pnglist: Liste der Bilder
    width: Breite pro Bild
    """
    res = ''.join(['<iframe src={} width={}% height=400></iframe>'.format(png,width) for png in pnglist])
    return HTML(res)

def MWurzel(merkmale,anzahlNullen=False):
    """
    Erzeuge Wurzelvektor mit N=merkmale Einträgen und k=anzahlNullen Nullen.
    """
    if anzahlNullen:
        vec = [1]*merkmale
        for i in range(anzahlNullen):
            vec[randrange(merkmale)] = 0
    else:
        vec = [randrange(2) for i in range(merkmale)]
    return vec

def Binaerkopie(m,p,q):
    """
    Zufällige Änderung einer Null oder Eins 
    """
    rd=random()
    if m==0:
        if rd<p:
            w=1
        else:
            w=0
    else:
        if rd<q:
            w=0
        else:
            w=1
    return(w)

def ModelKopie(M,p,q):
    """
    Zufällige Umwandlung eines Merkmalsvektors.
    """
    copy=[Binaerkopie(M[i],p,q) for i in range(len(M))]
    return(copy)

def Make_Kopiergraph(p,q,IniVektor,NTexte):
    """
    Erzeuge Kopiergraphen und dictionary aus Wurzelvektor mit Wahrscheinlichkeiten p (Umwandlung einer Null) 
    und q (Umwandlung einer Eins) und NTexte Kopien.
    """
    # dictionary mit allen objekten ID:Merkmalsvektor
    d={0:IniVektor}
    G=nx.DiGraph()
    for i in range(1,NTexte):
        # Zufallsauswahl eines Textes
        pos=randrange(len(d))
        Tsel=d[pos]
        # Kopie des ausgewählen Textes, d[i] ist Merkmalsvektor des i-ten Objekts
        d[i]=ModelKopie(Tsel,p,q)
        # Kopierpaare werden in einen Graph eingetragen
        G.add_edge(pos,i)
    return(G,d)

def diffv(d,i,j): 
    """Vergleich zweier Merkmalsvektoren i und j des dictionarys d
    """
    d1=d[i] # vektor des ersten objekts
    d2=d[j] # vektor des zweiten objekts
    diffv1=[]
    for u in range(len(d1)): # vergleich der Vektoren
        if d1[u]==0 and d2[u]==0:
            dv=0
        if d1[u]==1 and d2[u]==1:
            dv=1
        if d1[u]==1 and d2[u]==0:
            dv=2
        if d1[u]==0 and d2[u]==1:
            dv=3
        diffv1.append(dv)
    return(diffv1)

def categorie(i,j,G):
    """Festlegung der Kategorie (keine Kopierbeziehung, A->B, oder B->A """
    if G.has_edge(i,j):
        res = 1
    elif G.has_edge(j,i):
        res = 2
    else:
        res = 0
    return res

def Vergleich(d,i,j,G): # vergleich zweier Objekte
    """ Erzeuge dictionary der Merkmalsvektoren, Differenzen und Kategorie der Beziehung."""
    dd=(i,j,diffv(d,i,j),categorie(i,j,G))
    return(dd)

def datensatz_erzeugen(p,q,inputListe):
    """Erzeuge Trainingsdatensatz mit Zielvektor und Ursprungs-Dataframe."""
    # Dataframe mit festgelegten Spaltennamen erzeugen
    dfTemp = pd.DataFrame(inputListe,columns=['dia1','dia2','diff','kopie'])
    # Dataframe mit Anzahl 0/1/2/3 erzeugen,drehen,Spalten umbennenen und NaN durch 0 ersetzen
    dfTemp_diffcount = pd.DataFrame(dict(dfTemp['diff'].apply(lambda r: Counter(r))))\
            .transpose()\
            .rename(columns={0:'0->0',1:'1->1',2:'1->0',3:'0->1'})\
            .fillna(0)
    # Dataframe mit aufgespalteten Merkmalsvektoren
    dfTemp_diff = pd.DataFrame(dict(dfTemp['diff'])).transpose().fillna(0)
    # Alles zusammenfassen
    dfTempFull = dfTemp.join(dfTemp_diffcount)
    ###
    # Gesamt Produkt-Wahrscheinlichkeit angeben
    ###
    dfTempProd = dfTempFull.copy()
    dfTempProd['0->0'] = dfTempProd['0->0'].apply(lambda row: (1-p)**row)
    dfTempProd['1->1'] = dfTempProd['1->1'].apply(lambda row: (1-q)**row)
    dfTempProd['0->1'] = dfTempProd['0->1'].apply(lambda row: p**row)
    dfTempProd['1->0'] = dfTempProd['1->0'].apply(lambda row: q**row)
    #
    dfTempFull['wahrProdukt'] = np.log(dfTempProd['0->0'] * dfTempProd['1->1'] * dfTempProd['0->1'] * dfTempProd['1->0'])
    # Trainings-Datensätze:
    # Label-Vektor
    y_Temp = dfTempFull['kopie']
    # Feature-Matrix
    X_Temp = dfTempFull.drop('diff',axis=1).drop('kopie',axis=1).drop('dia1',axis=1).drop('dia2',axis=1)
    # Reihenfolge beachten: X,y, Dataframe, Index-dictionary
    return (X_Temp,y_Temp,dfTempFull)    

def bewertung_klassifizierung(classifier, X_test,y_test):
    """Bewerte den Klassifizierer anhand der Test-Daten. """
    scores = cross_val_score(classifier, X_test, y_test, cv=5,scoring='accuracy')
    y_pred = classifier.predict(X_test)
    print("Accuracy: %0.2f (+/- %0.2f) \n" % (scores.mean(), scores.std() * 2))
    print('Scores: {} \n'.format(scores))
    print('Summe vorhergesagter Links (Eintrag kann 1 oder 2 sein): {}'.format(y_pred.sum()))
    
def plot_confusion_matrix(classifier,X_test,y_test, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """Zeige die Confusions-Matrix."""
    fig = plt.figure(figsize=(12,4))
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def klassifizierter_Graph(classifier,X_test,y_test,dataframe):
    """ Rekonstruiere klassifizierten Graph aus Test-Daten"""
    # Erzeuge leeren Graph
    predictedGraph=pgv.AGraph(compound=True,directed=True)  
    # Quelle für Vorhersage
    predictionSource = X_test
    # Klassifizierung
    comp = classifier.predict(predictionSource)
    for i in range(len(comp)):
        order = [dataframe.dia1.iloc[i],dataframe.dia2.iloc[i]]
        if comp[i] == 1:
            # direkte Kopie: A->B
            predictedGraph.add_edge(order)
        elif comp[i] == 2:
            # inverse Kopie: B <- A, daher wird für add_edge Reihenfolge umgekehrt 
            predictedGraph.add_edge(list(reversed(order)))
        else:
            pass
    return predictedGraph

def delDoubles(inputGraph):
    """ Entferne doppelte Kanten."""
    edgeList = inputGraph.edges()
    dbs = [x for x in itertools.combinations(edgeList,2) if x[0][0]==x[1][1] and x[0][1]==x[1][0]]
    rmEdge = [x[0] if int(x[0][0]) > int(x[0][1]) else x[1] for x in dbs]
    try:
        inputGraph.delete_edges_from(rmEdge)
    except:
        pass
    return print('Entfernte Kanten:',rmEdge)