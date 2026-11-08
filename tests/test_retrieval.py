from stateful_safety.retrieval import select_history,bm25_search

def test_never_uses_future_history():
    prior=[(0,'paper boat'),(1,'blue skies'),(2,'paper boat design')]
    assert select_history('current','paper boat',prior,3)==[]
    assert select_history('recent','paper boat',prior,2)==prior[-2:]
    assert select_history('full','paper boat',prior,2)==prior
    assert bm25_search('paper boat',prior,1)==[(0,'paper boat')]

def test_k_zero_and_chronology():
    prior=[(0,'red apple'),(1,'purple melon'),(2,'red fig')]
    assert bm25_search('red',prior,0)==[]
    assert [i for i,_ in bm25_search('red',prior,2)]==[0,2]
